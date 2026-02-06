import unittest
import torch
import torch.distributions as dist
from weighted_sampling import run_smc, sample, observe


def simple_model():
    # 1. Prior: mu ~ Normal(0, 10)
    mu = sample("mu", dist.Normal(0.0, 10.0))

    # 2. Likelihood: observe data=5.0 from Normal(mu, 1.0)
    # Posterior for Normal-Normal conjugate with known variance 1
    # prior mean=0, var=100
    # likelihood mean=5, var=1
    # post_var = 1 / (1/100 + 1/1) = 100/101 approx 0.99
    # post_mean = post_var * (0/100 + 5/1) = 0.99 * 5 = 4.95
    observe(5.0, dist.Normal(mu, 1.0))

    return mu


class TestSimpleModel(unittest.TestCase):
    def test_simple_gaussian(self):
        torch.manual_seed(42)
        trace = run_smc(simple_model, num_particles=10000)

        mu_samples = trace["mu"]
        post_mean = mu_samples.mean().item()
        post_std = mu_samples.std().item()

        # Expected values
        expected_mean = 4.95
        expected_std = 0.99

        print(f"\nSimple Model: Mean={post_mean:.4f}, Std={post_std:.4f}")

        self.assertAlmostEqual(
            post_mean, expected_mean, delta=0.1, msg="Posterior mean diverged"
        )
        self.assertAlmostEqual(
            post_std, expected_std, delta=0.1, msg="Posterior std diverged"
        )


if __name__ == "__main__":
    unittest.main()
