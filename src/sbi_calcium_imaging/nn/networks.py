# %%
from json import encoder
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from sbi_calcium_imaging.nn.s4 import FFTConv
from sbi_calcium_imaging.nn.s4 import S4Block as S4


# TODO clean all of this up
class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        layer_lr=0.001,
        prenorm=False,
        bidirectional=False,
        d_state=64,
        lin_reg_dim=20,
        norm="batch",
    ):
        super().__init__()

        self.norm = norm
        self.prenorm = prenorm

        self.encoder = nn.Linear(d_input, d_model)

        self.linear = nn.Parameter(torch.randn(d_input, lin_reg_dim))
        self.lin_reg_dim = lin_reg_dim

        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(n_layers):
            self.s4_layers.append(
                S4(
                    d_model,
                    dropout=dropout,
                    transposed=True,
                    bidirectional=bidirectional,
                    lr=min(0.001, layer_lr),
                    d_state=d_state,
                    final_act="gelu",
                )
            )
            if norm == "layer":
                self.norms.append(nn.LayerNorm(d_model))
            else:
                raise ValueError(f"Unknown norm type: {norm}")
            self.dropouts.append(nn.Dropout1d(dropout))

        d_output = d_input
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, d_input, L)
        """
        x = x.transpose(-1, -2)

        x = self.encoder(x)

        x = x.transpose(-1, -2)
        for layer, norm in zip(self.s4_layers, self.norms):

            z = x
            if self.prenorm:
                if self.norm == "layer":
                    z = norm(z.transpose(-1, -2)).transpose(-1, -2)
                else:
                    raise ValueError(f"Unknown norm type: {self.norm}")

            z, _ = layer(z)

            x = z + x

            if not self.prenorm:
                if self.norm == "layer":
                    x = norm(x.transpose(-1, -2)).transpose(-1, -2)
                else:
                    raise ValueError(f"Unknown norm type: {self.norm}")

        x = x.transpose(-1, -2)
        x = self.decoder(x)
        x = x.transpose(-1, -2)

        # return F.softplus(x)
        return x


class AutoEncoderBlock(nn.Module):
    def __init__(
        self,
        C,
        L,
        kernel="s4",
        bidirectional=True,
        kernel_params=None,  # not used
        num_lin_per_mlp=2,
        use_act2=False,
    ):
        super().__init__()
        self.C = C
        self.L = L
        self.bidirectional = bidirectional
        self.kernel_params = kernel_params
        self.time_mixer = self.get_time_mixer(kernel)
        # channel-wise scale for post-act:
        self.post_tm_scale = nn.Conv1d(C, C, 1, bias=True, groups=C, padding="same")
        self.channel_mixer = self.get_channel_mixer(num_lin_per_mlp=num_lin_per_mlp)
        self.norm1 = nn.InstanceNorm1d(C, affine=False)  # make sure input is [B, C, L]!
        self.norm2 = nn.InstanceNorm1d(C, affine=False)  # we will use adaLN
        self.act1 = nn.GELU()
        self.act2 = nn.GELU() if use_act2 else nn.Identity()

        # 3 for each mixer, shift, scale, gate. gate remains unused for now
        self.ada_ln = nn.Parameter(torch.zeros(1, C * 6, 1), requires_grad=True)

    @staticmethod
    def affine_op(x_, shift, scale):
        # x is [B, C, L], shift and scale are [B, C, 1]
        assert len(x_.shape) == len(shift.shape), f"{x_.shape} != {shift.shape}"
        return x_ * (1 + scale) + shift

    def get_time_mixer(self, kernel):
        if kernel == "s4":
            time_mixer = FFTConv(
                self.C,
                bidirectional=self.bidirectional,
                activation=None,
            )
        elif kernel == "attention":
            time_mixer = MHAWrapper(embed_dim=self.C, num_heads=self.C // 64)
        elif kernel == "mamba":
            raise NotImplementedError("Mamba kernel not implemented in time mixer")
        else:
            raise ValueError(f"Unknown kernel type: {kernel}")

        return time_mixer

    def get_channel_mixer(self, num_lin_per_mlp=2):
        layers = [
            Rearrange("b c l -> b l c"),
            nn.Linear(self.C, self.C * 2, bias=False),
        ]
        for _ in range(max(num_lin_per_mlp - 2, 0)):
            layers.extend(
                [
                    nn.GELU(),
                    nn.Linear(self.C * 2, self.C * 2, bias=False),
                ]
            )
        layers.extend(
            [
                nn.GELU(),
                nn.Linear(self.C * 2, self.C, bias=False),
            ]
        )
        layers.append(Rearrange("b l c -> b c l"))
        return nn.Sequential(*layers)

    # TODO figure out if this all makes sense
    def forward(self, x):
        y = x  # x is residual stream
        y = self.norm1(y)
        ada_ln = repeat(self.ada_ln, "1 d c -> b d c", b=x.shape[0])
        shift_tm, scale_tm, gate_tm, shift_cm, scale_cm, gate_cm = ada_ln.chunk(
            6, dim=1
        )
        y = self.affine_op(y, shift_tm, scale_tm)
        y = self.time_mixer(y)
        y = y[0]  # get output not state for gconv and fftconv
        # y = x + gate_tm.unsqueeze(-1) * self.act1(y)
        y = x + self.post_tm_scale(self.act1(y))

        x = y  # x is again residual stream from last layer
        y = self.norm2(y)
        y = self.affine_op(y, shift_cm, scale_cm)
        # y = x + gate_cm.unsqueeze(-1) * self.act2(self.channel_mixer(y))
        y = x + self.act2(self.channel_mixer(y))
        return y


class AutoEncoder(nn.Module):
    def __init__(
        self,
        C_in,
        C,
        L,
        max_spikes=3,
        encoder_mode="linear",  # or "embed"
        kernel="s4",
        bidirectional=True,
        kernel_params=None,
        num_blocks=4,
        num_lin_per_mlp=2,
        latent_size=32,
        rate_rectifier="softplus",
        out_mode="poisson",  # or mulinomial
    ):
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.L = L
        self.bidirectional = bidirectional
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.latent_size = latent_size
        self.max_spikes = max_spikes
        self.encoder_mode = encoder_mode
        self.out_mode = out_mode

        if rate_rectifier == "softplus":
            self.rate_rectifier = lambda x: F.softplus(x)
        elif rate_rectifier == "exp":
            self.rate_rectifier = lambda x: torch.exp(x)
        elif rate_rectifier == "softplus_squared":
            self.rate_rectifier = lambda x: F.softplus(x) * 2
        else:
            raise ValueError(f"Unknown rate rectifier: {rate_rectifier}")

        if self.encoder_mode == "linear":
            self.encoder_in = nn.Conv1d(C_in, C, 1)
        elif self.encoder_mode == "embed":
            assert C % C_in == 0, "C must be multiple of C_in for embedding"
            self.encoder_in = nn.Embedding(max_spikes + 2, C // C_in)

        self.encoder = nn.ModuleList(
            [
                AutoEncoderBlock(
                    C,
                    L,
                    kernel,
                    bidirectional,
                    kernel_params,
                    num_lin_per_mlp=num_lin_per_mlp,
                )
                for _ in range(num_blocks)
            ]
        )
        self.bottleneck = nn.Conv1d(C, latent_size, 1)
        # self.decode = nn.Conv1d(latent_size, C_in, 1)
        self.decode = self.get_decoder(num_lin_per_mlp=3)

        if self.kernel == "attention":
            self.time_embedder = TimestepEmbedder(C)

    def encode(self, x):

        if self.encoder_mode == "linear":
            z = self.encoder_in(x.float())
        elif self.encoder_mode == "embed":
            # -1 is my mask token, so add 1 to everything
            x = torch.clamp(x, max=self.max_spikes)
            z = rearrange(self.encoder_in(x.long() + 1), "b c l e -> b (c e) l")
        else:
            raise ValueError(f"Unknown encoder mode: {self.encoder_mode}")

        if self.kernel == "attention":
            t = torch.arange(self.L, device=x.device)
            t_emb = self.time_embedder(t).transpose(0, 1)
            z = z + t_emb.unsqueeze(0)

        for block in self.encoder:
            z = block(z)
        z = self.bottleneck(z)

        return z

    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)
        y = self.rate_rectifier(y) if self.out_mode == "poisson" else y
        return y

    def get_decoder(self, num_lin_per_mlp=2):
        out_multiplier = 1 if self.out_mode == "poisson" else self.max_spikes + 1
        layers = [
            Rearrange("b c l -> b l c"),
            nn.Linear(self.latent_size, self.C * 2),
        ]
        for _ in range(max(num_lin_per_mlp - 2, 0)):
            layers.extend(
                [
                    nn.GELU(),
                    nn.Linear(self.C * 2, self.C * 2),
                ]
            )
        layers.extend(
            [
                nn.GELU(),
                nn.Linear(self.C * 2, self.C_in * out_multiplier),
            ]
        )
        layers.append(
            Rearrange("b l c -> b c l")
            if self.out_mode == "poisson"
            else Rearrange("b l (c k) -> b c l k", k=out_multiplier)
        )
        return nn.Sequential(*layers)


class MHAWrapper(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x):
        # x is (B, L, D)
        x = x.transpose(1, 2)
        x, _ = self.mha(x, x, x)
        x = x.transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_emb = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_emb)
        return t_emb
