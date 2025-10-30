import numpy as np
import torch
from torch import Tensor
from sbi.utils import BoxUniform

# NOTE: mainly taken from Peter's repo, slighly adapted to support many different spike trains at once


def bi_exp_kernel_torch(t: Tensor, tau_rise: Tensor, tau_decay: Tensor) -> Tensor:
    """
    Vectorized bi-exponential kernel for batched computation.
    t: [T] or scalar
    tau_rise, tau_decay: [N] batch dimension
    Returns: [N, T]
    """
    tau_rise = tau_rise.clamp(min=1e-6)
    tau_decay = tau_decay.clamp(min=tau_rise + 1e-6)

    # Reshape for broadcasting: tau [N, 1], t [1, T]
    if t.dim() == 0:
        t = t.unsqueeze(0)
    if t.dim() == 1:
        t = t.unsqueeze(0)  # [1, T]

    tau_rise = tau_rise.unsqueeze(-1)  # [N, 1]
    tau_decay = tau_decay.unsqueeze(-1)  # [N, 1]

    h = torch.exp(-t / tau_decay) - torch.exp(-t / tau_rise)  # [N, T]

    # Normalize each kernel
    peak = h.max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    h = h / peak

    return h


def hill_nl_torch(c: Tensor, kd: Tensor, n: Tensor) -> Tensor:
    """
    Vectorized Hill saturation.
    c: [N, T]
    kd, n: [N] or [N, 1]
    Returns: [N, T]
    """
    c = c.clamp(min=0.0)
    n = n.clamp(min=1e-3)
    kd = kd.clamp(min=1e-9)

    if kd.dim() == 1:
        kd = kd.unsqueeze(-1)
    if n.dim() == 1:
        n = n.unsqueeze(-1)

    c_n = torch.pow(c, n)
    kd_n = torch.pow(kd, n)

    return c_n / (c_n + kd_n)


class BatchedSimulator:
    """Optimized batched simulator with kernel caching."""

    def __init__(
        self,
        dt: float,
        device: str = "cpu",
        nonlinearity: str = "hill",
        max_tau_decay: float = 2.0,
    ):
        # self.spike_train = torch.tensor(spike_train, dtype=torch.float32, device=device)
        self.dt = dt
        self.device = device
        self.nonlinearity = nonlinearity

        # Pre-compute kernel time base (long enough for largest tau_decay)
        self.kT = int(np.ceil(5.0 * max_tau_decay / dt))
        self.tker = torch.arange(self.kT, dtype=torch.float32, device=device) * dt

    def __call__(self, theta: Tensor, spike_train: Tensor) -> Tensor:
        """
        Batched simulation.
        theta: [N, 7] where columns are [tau_rise, tau_decay, amp, kd, n, f0, sigma]
        Returns: [N, T] fluorescence traces
        """

        if spike_train.dim() == 1:
            spike_train = spike_train.unsqueeze(0)  # [1, T]
        else:
            assert spike_train.shape[0] == theta.shape[0]

        # Extract parameters
        tau_rise = theta[:, 0]  # [N]
        tau_decay = theta[:, 1]  # [N]
        amp = theta[:, 2]  # [N]
        kd = theta[:, 3]  # [N]
        n = theta[:, 4]  # [N]
        f0 = theta[:, 5]  # [N]
        sigma = theta[:, 6]  # [N]

        # Build kernels for all parameter sets at once [N, kT]
        h = bi_exp_kernel_torch(self.tker, tau_rise, tau_decay)  # [N, kT]

        # Convolve spike train with each kernel using FFT (fastest for long signals)
        # spike_train is [T], h is [N, kT]
        # Pad to avoid circular convolution artifacts
        T = spike_train.shape[1]
        conv_len = T + self.kT - 1
        fft_len = 2 ** int(np.ceil(np.log2(conv_len)))  # Next power of 2 for efficiency

        # FFT of spike train [fft_len]
        spike_fft = torch.fft.rfft(spike_train, n=fft_len)

        # FFT of kernels [N, fft_len//2 + 1]
        h_padded = torch.nn.functional.pad(h, (0, fft_len - self.kT))
        h_fft = torch.fft.rfft(h_padded, dim=1)

        # Multiply in frequency domain and transform back [N, fft_len]
        c_fft = h_fft * spike_fft
        c_full = torch.fft.irfft(c_fft, n=fft_len)

        # Extract valid portion [N, T]
        c = c_full[:, :T]

        # Apply nonlinearity
        if self.nonlinearity == "hill":
            g = hill_nl_torch(c, kd, n)  # [N, T]
        else:
            g = c

        # Fluorescence
        amp = amp.unsqueeze(-1)  # [N, 1]
        f0 = f0.unsqueeze(-1)  # [N, 1]
        F = f0 + amp * g  # [N, T]

        # Add noise
        if (sigma > 0).any():
            sigma = sigma.unsqueeze(-1)  # [N, 1]
            noise = torch.randn_like(F) * sigma
            F = F + noise

        return F

    def get_default_prior(self):
        """Build tighter priors if you have good starting points."""
        low = torch.tensor(
            [
                0.005,  # tau_rise
                0.2,  # tau_decay
                5.0,  # amp
                5.0,  # kd
                1.0,  # n
                -0.1,  # f0
                0.03,  # sigma
            ],
            device=self.device,
        )

        high = torch.tensor(
            [
                0.008,  # tau_rise
                1.0,  # tau_decay
                30.0,  # amp
                20.0,  # kd
                2.0,  # n
                0.1,  # f0
                0.3,  # sigma
            ],
            device=self.device,
        )

        return BoxUniform(low=low, high=high)
