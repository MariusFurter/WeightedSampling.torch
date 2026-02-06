import unittest
import torch
import torch.distributions as dist
from weighted_sampling import run_smc, sample, observe


def random_walk_model(data):
    # Initial state
    x = sample("x_0", dist.Normal(0.0, 1.0))

    # Loop over time steps
    for t, y in enumerate(data):
        x = sample(f"x_{t+1}", dist.Normal(x, 1.0))
        observe(y, dist.Normal(x, 0.5))


class TestRandomWalk(unittest.TestCase):
    def test_sequential_random_walk(self):
        torch.manual_seed(42)

        # 1. Generate Synthetic Data
        true_x = [0.0]
        data = []

        for t in range(10):
            next_val = true_x[-1] + torch.randn(1).item()
            true_x.append(next_val)
            obs = next_val + torch.randn(1).item() * 0.5
            data.append(torch.tensor(obs))

        # 2. Run Inference
        trace = run_smc(random_walk_model, data, num_particles=1000)

        # 3. Analyze last step
        final_x = trace["x_10"]
        estimated_mean = final_x.mean().item()
        true_last = true_x[-1]

        print(f"\nRandom Walk: Estimated={estimated_mean:.4f}, True={true_last:.4f}")

        # Check if we tracked well
        error = abs(estimated_mean - true_last)
        self.assertLess(error, 1.0, "Tracking error too high")


if __name__ == "__main__":
    unittest.main()
