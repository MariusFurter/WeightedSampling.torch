import torch
import torch.distributions as dist
import time
from weighted_sampling import run_smc, sample, observe, move, RandomWalkProposal


def generate_synthetic_data(num_points=20):
    torch.manual_seed(0)
    x = torch.linspace(0, 10, num_points)
    true_a = 2.0
    true_b = -1.0
    true_sigma = 0.5
    y = true_a * x + true_b + torch.randn(num_points) * true_sigma
    return list(zip(x, y))


def linear_model(data):
    a = sample("a", dist.Normal(0, 5))
    b = sample("b", dist.Normal(0, 5))

    for i, (x, y) in enumerate(data):
        y_pred = a * x + b
        observe(y, dist.Normal(y_pred, 0.5))

        # Force move on every step to benchmark replay performance
        # threshold=1.1 ensures the move is always executed (ratio <= 1.0 < 1.1)
        move("a", "b", RandomWalkProposal(scale=0.1), threshold=1.1)


def benchmark_mh():
    # Parameters designed to stress MH move / replay
    num_particles = 1000
    num_points = 50
    data = generate_synthetic_data(num_points)

    print(f"Running MH Linear Regression Benchmark...")
    print(f"Particles: {num_particles}")
    print(f"Data Points: {num_points}")
    print(f"Total Moves: {num_points}")  # One move per data point

    start_time = time.time()
    # Note: track_joint=True is no longer strictly required for performance as move() now auto-enables it.
    # We remove it here to verify auto-detection.
    results = run_smc(
        linear_model,
        data,
        num_particles=num_particles,
        ess_threshold=0.5,
    )
    end_time = time.time()

    duration = end_time - start_time
    print(f"Done.")
    print(f"Total Time: {duration:.4f}s")
    print(f"Moves/sec: {num_points / duration:.2f}")
    print(f"Log Evidence: {results['log_evidence'].item():.4f}")


if __name__ == "__main__":
    benchmark_mh()
