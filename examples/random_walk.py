import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
from weighted_sampling import run_smc, sample, observe


def random_walk_model(data):
    # Initial state
    x = sample("x_0", dist.Normal(0.0, 1.0))

    # Loop over time steps
    for t, y in enumerate(data):
        x = sample(f"x_{t+1}", dist.Normal(x, 1.0))
        observe(y, dist.Normal(x, 1.0))


def generate_synthetic_data(num_steps=10):
    torch.manual_seed(42)
    true_x = [0.0]
    data = []

    for t in range(num_steps):
        next_val = true_x[-1] + torch.randn(1).item()
        true_x.append(next_val)
        obs = next_val + torch.randn(1).item() * 1.0
        data.append(torch.tensor(obs))

    return data, true_x


data, true_x = generate_synthetic_data(num_steps=50)

results = run_smc(random_walk_model, data, num_particles=1000, ess_threshold=0.5)

# Plotting
trace_keys = [f"x_{t}" for t in range(len(true_x))]
traces = torch.stack([results[k] for k in trace_keys]).detach().cpu().numpy()  # (T, N)
log_weights = results["log_weights"].detach().cpu()
weights = torch.exp(log_weights - log_weights.max()).numpy()

# Select 50 random traces
N = traces.shape[1]
indices = torch.randperm(N)[:50].numpy()

plt.figure(figsize=(10, 6))

# Plot selected traces
for idx in indices:
    # Alpha scaled by weight (max weight = 1.0)
    alpha = max(0.01, min(1.0, weights[idx]))
    plt.plot(traces[:, idx], color="blue", alpha=alpha * 0.5)

# Plot true values
plt.plot(true_x, color="red", linewidth=2, label="True Value")

plt.title("True Values and Filtered Traces")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.savefig("examples/plots/random_walk_plot.png")
