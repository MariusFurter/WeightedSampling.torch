"""
State-Space Filtering (Bootstrap Particle Filter)
===================================================
Track a latent AR(1) state through noisy observations.
Resampling triggers automatically when ESS drops.

Model:
    x_0     ~ Normal(0, 1)
    x_{t+1} ~ Normal(0.8 * x_t, 0.5)
    y_t     ~ Normal(x_t, 0.5)

Generates: examples/plots/state_space_model.png
"""

import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
from weighted_sampling import model, sample, observe, expectation


# ---- Model ----


@model
def state_space_model(observations):
    x = sample("x_0", dist.Normal(0.0, 1.0))
    for t, y in enumerate(observations):
        x = sample(f"x_{t+1}", dist.Normal(0.8 * x, 0.5))
        observe(y, dist.Normal(x, 0.5))


# ---- Synthetic data ----


def make_data(T=50):
    """Simulate from the AR(1) state-space model."""
    torch.manual_seed(42)
    true_states = [torch.randn(1).item()]
    observations = []
    for _ in range(T):
        x_new = 0.8 * true_states[-1] + 0.5 * torch.randn(1).item()
        true_states.append(x_new)
        observations.append(torch.tensor(x_new + 0.5 * torch.randn(1).item()))
    return observations, true_states


# ---- Plotting ----


def plot_results(result, observations, true_states):
    T = len(observations)
    timesteps = range(T + 1)

    # Compute filtered posterior means
    filtered_means = []
    for t in timesteps:
        mean = expectation(result, lambda **kw: kw[f"x_{t}"])
        filtered_means.append(mean.item())

    # Extract particle traces
    log_w = result["log_weights"].detach().cpu()
    weights = torch.exp(log_w - log_w.max()).numpy()
    traces = torch.stack([result[f"x_{t}"] for t in timesteps]).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(9, 3.5))

    # Subsample particle traces for clarity
    indices = torch.randperm(traces.shape[1])[:60].numpy()
    for idx in indices:
        alpha = max(0.02, min(1.0, float(weights[idx]))) * 0.4
        ax.plot(
            timesteps, traces[:, idx], color="steelblue", alpha=alpha, linewidth=0.5
        )

    ax.plot(timesteps, true_states, color="red", linewidth=2, label="True state")
    ax.plot(
        timesteps,
        filtered_means,
        color="black",
        linewidth=1.5,
        linestyle="--",
        label="Filtered mean",
    )
    ax.scatter(
        range(1, T + 1),
        [y.item() for y in observations],
        color="orange",
        s=15,
        zorder=5,
        label="Observations",
        alpha=0.7,
    )

    ax.set_xlabel("Time step")
    ax.set_ylabel("State")
    ax.set_title("State-Space Filtering: Particle Traces and Filtered Mean")
    ax.legend()

    plt.tight_layout()
    plt.savefig("examples/plots/state_space_model.png", dpi=150)
    print("Plot saved to examples/plots/state_space_model.png")


# ---- Main ----

if __name__ == "__main__":
    observations, true_states = make_data(T=50)

    result = state_space_model(observations, num_particles=1000)
    print(f"Log evidence: {result.log_evidence:.2f}")

    plot_results(result, observations, true_states)
