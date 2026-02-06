import torch
import torch.distributions as dist
from weighted_sampling import (
    run_smc,
    sample,
    observe,
    move,
    summary,
    RandomWalkProposal,
    AdaptiveProposal,
)


def simple_model_no_move():
    x = sample("x", dist.Normal(0.0, 1.0))
    # Make observing strong so we get collapse/weight variance
    observe(torch.tensor(5.0), dist.Normal(x, 0.1))
    return x


def simple_model_with_move():
    x = sample("x", dist.Normal(0.0, 1.0))
    observe(torch.tensor(5.0), dist.Normal(x, 0.1))

    # At this point, weights are very unequal.
    # If we forced resampling here, we would have duplicates.
    # But run_smc handles resampling.
    # Let's apply move.
    move("x", RandomWalkProposal(scale=0.5))
    return x


def test_mh():
    torch.manual_seed(42)
    N = 100

    print("Running without move...")
    trace1 = run_smc(simple_model_no_move, num_particles=N)  # Resampling likely happens
    stats1 = summary(trace1)
    print(
        f"Mean: {stats1['x']['mean']:.4f}, Std: {stats1['x']['std']:.4f}, Unique: {stats1['x']['n_unique']}"
    )

    torch.manual_seed(42)
    print("\nRunning with move...")
    trace2 = run_smc(simple_model_with_move, num_particles=N)
    stats2 = summary(trace2)
    print(
        f"Mean: {stats2['x']['mean']:.4f}, Std: {stats2['x']['std']:.4f}, Unique: {stats2['x']['n_unique']}"
    )

    # We expect 'with move' to have more unique particles if resampling caused collapse,
    # OR broadly similar stats but different values.
    # Note: 'move' is applied AFTER observe.
    # Observe updates weights.
    # If we don't resample, updates weights.
    # Move doesn't change weights.

    # If no resampling happened:
    # unique1 = 100
    # unique2 = 100 (but moved positions).

    # To force collapse, we can inject a manual resample or use very low ESS threshold?
    # Or just loop.


if __name__ == "__main__":
    test_mh()
