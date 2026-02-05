import torch
import torch.distributions as dist
from weighted_sampling import run_smc, sample, observe


def simple_model():
    # 1. Prior: mu ~ Normal(0, 10)
    # The 'sample' primitive handles expanding this to (N,) particles.
    mu = sample("mu", dist.Normal(0.0, 10.0))

    # 2. Likelihood: observe data=5.0 from Normal(mu, 1.0)
    # Note: 'mu' is a vector (N,). 'observe' will compute log_prob(5.0) for each particle.
    observe(dist.Normal(mu, 1.0), torch.tensor(5.0))

    return mu


def run_experiment():
    print("Running Simple Gaussian Inference...")
    trace = run_smc(simple_model, num_particles=10000)

    mu_samples = trace["mu"]
    post_mean = mu_samples.mean().item()
    post_std = mu_samples.std().item()

    print(f"Posterior Mean: {post_mean:.4f} (Expected ~4.95)")
    print(f"Posterior Std:  {post_std:.4f} (Expected ~0.99)")


if __name__ == "__main__":
    run_experiment()
