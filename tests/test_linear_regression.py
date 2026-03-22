import unittest
import torch
import torch.distributions as dist
from weighted_sampling import model, run_smc, sample, observe, summary


@model
def bayesian_linreg(data):
    a = sample("a", dist.Normal(0.0, 2.0))  # prior on slope
    b = sample("b", dist.Normal(0.0, 2.0))  # prior on intercept
    for x, y in data:
        observe(y, dist.Normal(a * x + b, 0.3))


class TestLinearRegression(unittest.TestCase):
    def test_posterior_recovers_true_params(self):
        """
        Regression test for the resample-before-step bug (commit 0378845).

        When observe_site resamples *before* evaluating log_prob, the
        distribution object (constructed by Python before entering the method)
        holds pre-resampling values of a and b, while the trace has been
        shuffled.  This produces incorrect weights and biased posteriors.

        The test checks that the posterior mean of a and b are within
        reasonable tolerance of the true values used to generate the data.
        """
        torch.manual_seed(0)

        true_a, true_b = 2.0, -1.0
        xs = torch.linspace(0, 5, 5)
        ys = true_a * xs + true_b  # noise-free for a tight test
        data = list(zip(xs, ys))

        result = bayesian_linreg(data, num_particles=5000)
        stats = summary(result)

        a_mean = stats["a"]["mean"].item()
        b_mean = stats["b"]["mean"].item()

        self.assertAlmostEqual(
            a_mean,
            true_a,
            delta=0.3,
            msg=f"Posterior mean of a ({a_mean:.3f}) too far from true value ({true_a})",
        )
        self.assertAlmostEqual(
            b_mean,
            true_b,
            delta=0.3,
            msg=f"Posterior mean of b ({b_mean:.3f}) too far from true value ({true_b})",
        )


if __name__ == "__main__":
    unittest.main()
