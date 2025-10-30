# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sbi.utils import BoxUniform
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sbi_calcium_imaging.hill_simulator import BatchedSimulator
from sbi_calcium_imaging.nn.networks import AutoEncoder
from sbi_calcium_imaging.spike_train_utils import (
    load_ground_truth_mat,
    make_spike_train,
)

# %%
DT = 0.008200000000215368
test_simu = BatchedSimulator(DT, device="cpu")
prior = test_simu.get_default_prior()


# %%
theta_samples = []
spike_samples = []
fluo_samples = []
for _ in range(20):
    theta_batch = prior.sample((500,))
    spikes_batch = torch.poisson(torch.ones(500, 4096) * 0.005)
    fluo_batch = test_simu(
        theta_batch,
        spike_train=spikes_batch,
    )
    theta_samples.append(theta_batch)
    spike_samples.append(spikes_batch)
    fluo_samples.append(fluo_batch)
theta_samples = torch.cat(theta_samples, dim=0)
spike_samples = torch.cat(spike_samples, dim=0)
fluo_samples = torch.cat(fluo_samples, dim=0)
print(fluo_samples.shape)
print(spike_samples.shape)
print(theta_samples.shape)


# %%
rdx = np.random.randint(0, fluo_samples.shape[0])
plt.plot(fluo_samples[rdx])
plt.plot(spike_samples[rdx])
plt.show()


# %%
dataset = TensorDataset(fluo_samples.unsqueeze(1), spike_samples.unsqueeze(1))
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# %%
# Set up model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

sequence_length = fluo_samples.shape[-1]

model = AutoEncoder(
    C_in=1,
    C=128,
    L=sequence_length,
    num_blocks=4,
    num_lin_per_mlp=2,
    latent_size=128,
).to(device)

criterion = nn.PoissonNLLLoss(log_input=False)
optimizer = AdamW(model.parameters(), lr=5e-4)

print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")


# %%
def train_model(
    model,
    dataloader,
    criterion,
    optimizer,
    num_epochs=100,
):

    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        # Progress bar for batches
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        # Calculate average loss for epoch
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}")

    return train_losses


# %%
# Run training
print("Starting training...")
train_losses = train_model(
    model=model,
    dataloader=dataloader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=50,
)

# %%
# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses)
plt.title("Training Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()


# %%
# sample some test data points
test_theta_samples = prior.sample((50,))

test_poison_spikes = torch.poisson(torch.ones(50, 4096) * 0.005)
with torch.no_grad():
    test_F_samples = test_simu(
        test_theta_samples,
        # spike_train=torch.tensor(spike_train, dtype=torch.float32),
        spike_train=test_poison_spikes,
    )


test_F_samples = test_F_samples.unsqueeze(1)  # [N, T, 1]
test_poison_spikes = test_poison_spikes.unsqueeze(1)  # [N, T, 1]

with torch.no_grad():
    test_outputs = model(test_F_samples.to(device).float()).cpu()

# %%
idx = np.random.randint(0, test_F_samples.shape[0])
plt.plot(test_F_samples[idx, 0].numpy(), alpha=0.5)
plt.plot(test_poison_spikes[idx, 0].numpy(), alpha=0.8)
plt.plot(test_outputs[idx, 0].numpy(), alpha=0.8)
plt.show()

# %%
fluo_time, fluo_mean, ap_times_s = load_ground_truth_mat("peters_trace.mat")
spike_train, dt = make_spike_train(fluo_time, ap_times_s)

plt.plot(fluo_mean)
plt.show()

# %%
start = 5000
dur = 4096
fluo_mean_tensor = (
    torch.tensor(fluo_mean[start : start + dur]).unsqueeze(0).unsqueeze(0)
)
with torch.no_grad():
    fluo_out = model(fluo_mean_tensor.to(device).float()).cpu()
plt.plot(spike_train[start : start + dur])
plt.plot(fluo_out[0, 0].numpy())
# plt.plot(fluo_mean[start : start + dur])
plt.show()
