"""
Effect of Metropolis-Hastings Moves
====================================
Compare SMC inference with and without MH rejuvenation moves
on a model where the likelihood concentrates far from the prior.

Without moves, resampling causes particle collapse (many duplicates).
With moves, particles are diversified after resampling.
"""

import torch
import torch.distributions as dist
from weighted_sampling import (
    run_smc,
    sample,
    observe,
    move,
    summary,
    RandomWalkProposal,
)


def model_without_move():
    x = sample("x", dist.Normal(0.0, 1.0))
    observe(torch.tensor(5.0), dist.Normal(x, 0.1))


def model_with_move():
    x = sample("x", dist.Normal(0.0, 1.0))
    observe(torch.tensor(5.0), dist.Normal(x, 0.1))
    move("x", RandomWalkProposal(scale=0.5))


if __name__ == "__main__":
    N = 500
    torch.manual_seed(42)

    # Without MH move
    result_no_move = run_smc(model_without_move, num_particles=N)
    stats_no = summary(result_no_move)

    # With MH move
    torch.manual_seed(42)
    result_with_move = run_smc(model_with_move, num_particles=N)
    stats_with = summary(result_with_move)

    print("Without MH move:")
    print(
        f"  Mean: {stats_no['x']['mean']:.4f}  Std: {stats_no['x']['std']:.4f}  Unique particles: {stats_no['x']['n_unique']}"
    )
    print("\nWith MH move:")
    print(
        f"  Mean: {stats_with['x']['mean']:.4f}  Std: {stats_with['x']['std']:.4f}  Unique particles: {stats_with['x']['n_unique']}"
    )
