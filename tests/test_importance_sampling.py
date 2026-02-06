import unittest
import torch
import torch.distributions as dist
from weighted_sampling import run_smc, sample, ImportanceSampler, summary


def importance_model():
    # Target: Normal(3, 1)
    # Proposal: Normal(0, 5) (Wider, shifted)
    target = dist.Normal(3.0, 1.0)
    proposal = dist.Normal(0.0, 5.0)

    is_dist = ImportanceSampler(target, proposal)
    x = sample("x", is_dist)
    return x


class TestImportanceSampling(unittest.TestCase):
    def test_importance_sampling(self):
        torch.manual_seed(42)
        # N needs to be high for decent IS
        trace = run_smc(importance_model, num_particles=100000)

        stats = summary(trace)
        mean = stats["x"]["mean"].item()

        # Target mean is 3.0
        print(f"\nIS Model: Mean={mean:.4f}", " Target=3.0")

        self.assertAlmostEqual(mean, 3.0, delta=0.1, msg="IS mean did not match target")


if __name__ == "__main__":
    unittest.main()
