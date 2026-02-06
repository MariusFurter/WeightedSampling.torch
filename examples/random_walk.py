import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
from weighted_sampling import run_smc, sample, observe


def random_walk_model(data):
    # Initial state
    x = sample("x_0", dist.Normal(0.0, 1.0))

    # Loop over time steps
    for t, y in enumerate(data):
        # Transition: x_t ~ Normal(x_{t-1}, 1.0)
        # We name variables "x_1", "x_2", etc.
        # Alternatively, we could update a mutable container, but unique names are cleaner for trace.
        x = sample(f"x_{t+1}", dist.Normal(x, 1.0))

        # Observe
        observe(y, dist.Normal(x, 0.5))


def run_sequential_experiment():
    print("Running Sequential Random Walk...")

    # 1. Generate Synthetic Data
    true_x = [0.0]
    data = []
    torch.manual_seed(42)

    for t in range(10):
        # x_t = x_{t-1} + noise
        next_val = true_x[-1] + torch.randn(1).item()
        true_x.append(next_val)

        # y_t = x_t + noise
        obs = next_val + torch.randn(1).item() * 0.5
        data.append(torch.tensor(obs))

    print(f"True trajectory (last 3): {true_x[-3:]}")

    # 2. Run Inference
    # We pass the data directly to run_smc, which forwards it to the model
    trace = run_smc(random_walk_model, data, num_particles=1000)

    # 3. Analyze last step
    final_x = trace["x_10"]
    print(f"Estimated x_10: {final_x.mean().item():.4f}")
    print(f"True x_10:      {true_x[-1]:.4f}")

    # Check if we tracked well
    error = abs(final_x.mean().item() - true_x[-1])
    if error < 1.0:
        print("SUCCESS: Tracking within reasonable bounds.")
    else:
        print("WARNING: Tracking seems off.")


if __name__ == "__main__":
    run_sequential_experiment()
