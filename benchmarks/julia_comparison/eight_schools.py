# Eight Schools Benchmark — WeightedSampling.torch (SMC)
#
# Hierarchical model from Gelman et al. (2003) "Bayesian Data Analysis", Sec 5.5.
# Runs two variants: resampling-only and with MH moves.
#
# Usage: python benchmarks/julia_comparison/eight_schools.py

import torch
import torch.distributions as dist
import math
import time
import statistics
from weighted_sampling import (
    run_smc,
    sample,
    observe,
    move,
    deterministic,
    RandomWalkProposal,
)


# Data
J = 8
y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
sigma = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]


def eight_schools_no_move(J, y, sigma):
    mu = sample("mu", dist.Normal(0.0, 5.0))
    log_tau = sample("log_tau", dist.Normal(math.log(5.0), 1.0))
    tau = deterministic("tau", torch.exp(log_tau))

    for j in range(J):
        theta_j = sample(f"theta_{j}", dist.Normal(mu, tau))
        observe(y[j], dist.Normal(theta_j, sigma[j]))


def eight_schools_move(J, y, sigma):
    mu = sample("mu", dist.Normal(0.0, 5.0))
    log_tau = sample("log_tau", dist.Normal(math.log(5.0), 1.0))
    tau = deterministic("tau", torch.exp(log_tau))

    for j in range(J):
        theta_j = sample(f"theta_{j}", dist.Normal(mu, tau))
        observe(y[j], dist.Normal(theta_j, sigma[j]))
        move(["mu", "log_tau"], RandomWalkProposal(scale=0.5), threshold=0.5)


def run_variant(label, model_fn, n_particles_list, n_runs):
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")

    # Warmup
    torch.manual_seed(42)
    run_smc(model_fn, J, y, sigma, num_particles=100, ess_threshold=0.5)

    for n_particles in n_particles_list:
        print(f"\n--- n_particles = {n_particles} ---")

        times = []
        for run in range(n_runs):
            torch.manual_seed(42 + run)
            t0 = time.perf_counter()
            results = run_smc(
                model_fn,
                J,
                y,
                sigma,
                num_particles=n_particles,
                ess_threshold=0.5,
            )
            t1 = time.perf_counter()
            times.append(t1 - t0)

        med_time = statistics.median(times)

        log_w = results["log_weights"]
        weights = torch.softmax(log_w, dim=0)
        mu_vals = results["mu"]
        mu_mean = (weights * mu_vals).sum().item()
        n_unique_mu = len(torch.unique(torch.round(mu_vals, decimals=6)))

        print(
            f"Median time:  {med_time:.4f} s  (range: {min(times):.4f} – {max(times):.4f})"
        )
        print(f"Log evidence: {results['log_evidence'].item():.4f}")
        print(f"μ mean:       {mu_mean:.2f}")
        print(f"Unique μ:     {n_unique_mu} / {n_particles}")


def benchmark_eight_schools():
    n_particles_list = [1_000, 5_000, 10_000]
    n_runs = 5

    print("=" * 60)
    print("Eight Schools Benchmark — WeightedSampling.torch (SMC)")
    print("=" * 60)
    print(f"Runs per config: {n_runs} (median reported)")

    run_variant("Resampling only", eight_schools_no_move, n_particles_list, n_runs)
    run_variant("With MH moves", eight_schools_move, n_particles_list, n_runs)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    benchmark_eight_schools()
