"""
Bayesian Linear Regression with MH Moves
=========================================
Infer slope and intercept from noisy data using SMC with
random-walk Metropolis-Hastings rejuvenation moves.

Generates: examples/plots/linear_regression.png
"""

import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from weighted_sampling import (
    run_smc,
    sample,
    observe,
    summary,
    move,
    RandomWalkProposal,
)


# ---- Model ----


def linear_regression(data):
    a = sample("a", dist.Normal(0, 5))
    b = sample("b", dist.Normal(0, 5))
    for x, y in data:
        observe(y, dist.Normal(a * x + b, 0.1))
        move(["a", "b"], RandomWalkProposal(scale=0.1), threshold=0.5)


# ---- Synthetic data ----


def make_data(num_points=10, true_a=2.0, true_b=-1.0, noise=0.1):
    torch.manual_seed(0)
    xs = torch.linspace(0, 10, num_points)
    ys = true_a * xs + true_b + noise * torch.randn(num_points)
    return list(zip(xs, ys)), xs


# ---- Plotting ----


def plot_results(result, data, xs, true_a, true_b):
    a_samples = result["a"].detach().cpu().numpy()
    b_samples = result["b"].detach().cpu().numpy()
    log_w = result["log_weights"].detach().cpu()
    weights = torch.exp(log_w - log_w.max()).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Left: posterior regression lines
    ax = axes[0]
    x_np = xs.numpy()
    indices = torch.randperm(len(a_samples))[:80].numpy()
    for idx in indices:
        alpha = max(0.02, min(1.0, float(weights[idx]))) * 0.5
        ax.plot(
            x_np,
            a_samples[idx] * x_np + b_samples[idx],
            color="steelblue",
            alpha=alpha,
        )
    ax.plot(x_np, true_a * x_np + true_b, color="red", linewidth=2, label="True line")
    ax.scatter(
        [d[0].item() for d in data],
        [d[1].item() for d in data],
        color="black",
        s=20,
        zorder=5,
        label="Data",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Posterior Regression Lines")
    ax.legend()

    # Right: joint posterior scatter
    ax = axes[1]
    ax.scatter(a_samples, b_samples, c=weights, cmap="Blues", s=8, alpha=0.6)
    ax.axvline(
        true_a, color="red", linestyle="--", linewidth=1, label=f"true a = {true_a}"
    )
    ax.axhline(
        true_b, color="red", linestyle="--", linewidth=1, label=f"true b = {true_b}"
    )
    ax.set_xlabel("a (slope)")
    ax.set_ylabel("b (intercept)")
    ax.set_title("Joint Posterior")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("examples/plots/linear_regression.png", dpi=150)
    print("Plot saved to examples/plots/linear_regression.png")


# ---- Main ----

if __name__ == "__main__":
    true_a, true_b = 2.0, -1.0
    data, xs = make_data(true_a=true_a, true_b=true_b)

    result = run_smc(linear_regression, data, num_particles=1000, ess_threshold=0.5)

    stats = summary(result)
    print(f"a: {stats['a']['mean']:.3f} ± {stats['a']['std']:.3f}  (true: {true_a})")
    print(f"b: {stats['b']['mean']:.3f} ± {stats['b']['std']:.3f}  (true: {true_b})")
    print(f"Log evidence: {result.log_evidence:.2f}")

    plot_results(result, data, xs, true_a, true_b)
