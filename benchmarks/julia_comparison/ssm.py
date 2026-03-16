# State-Space Model Benchmark — WeightedSampling.torch (SMC)
#
# Linear-Gaussian SSM:
#   x_0 ~ N(0, 1)
#   x_t = 0.9 * x_{t-1} + ε_t,  ε_t ~ N(0, 1)
#   y_t ~ N(x_t, 1)
#
# Usage: python benchmarks/julia_comparison/ssm.py

import torch
import torch.distributions as dist
import math
import time
import statistics
from weighted_sampling import run_smc, sample, observe

A = 0.9  # AR coefficient
Q = 1.0  # process noise std
R = 1.0  # observation noise std


def generate_data(T):
    torch.manual_seed(42)
    xs = [0.0]
    ys = []
    for t in range(T):
        x_new = A * xs[-1] + Q * torch.randn(1).item()
        xs.append(x_new)
        ys.append(x_new + R * torch.randn(1).item())
    return xs, ys


def kalman_filter(ys):
    T = len(ys)
    mu = 0.0
    sigma2 = 1.0
    log_evidence = 0.0
    filtered_means = []

    for t in range(T):
        mu_pred = A * mu
        sigma2_pred = A**2 * sigma2 + Q**2

        S = sigma2_pred + R**2
        K = sigma2_pred / S
        innov = ys[t] - mu_pred
        mu = mu_pred + K * innov
        sigma2 = (1 - K) * sigma2_pred

        log_evidence += -0.5 * (math.log(2 * math.pi * S) + innov**2 / S)
        filtered_means.append(mu)

    return filtered_means, log_evidence


def ssm_model(data, a, q, r):
    x = sample("x_0", dist.Normal(0.0, 1.0))
    for t, y_val in enumerate(data):
        x = sample(f"x_{t+1}", dist.Normal(a * x, q))
        observe(y_val, dist.Normal(x, r))


def benchmark_ssm():
    T = 200
    n_particles_list = [1_000, 5_000, 10_000]
    n_runs = 5

    _, ys = generate_data(T)
    kf_means, kf_evidence = kalman_filter(ys)

    print("=" * 60)
    print("SSM Benchmark — WeightedSampling.torch (Bootstrap PF)")
    print("=" * 60)
    print(f"Timesteps: {T}")
    print(f"Kalman log evidence: {kf_evidence:.4f}")
    print(f"Runs per config: {n_runs} (median reported)")

    # Warmup
    torch.manual_seed(0)
    run_smc(ssm_model, ys, A, Q, R, num_particles=100, ess_threshold=0.5)

    print(f"\n{'─' * 60}")
    print(f"  Resampling only")
    print(f"{'─' * 60}")

    for n_particles in n_particles_list:
        print(f"\n--- n_particles = {n_particles} ---")

        times = []
        for run in range(n_runs):
            torch.manual_seed(42 + run)
            t0 = time.perf_counter()
            results = run_smc(
                ssm_model,
                ys,
                A,
                Q,
                R,
                num_particles=n_particles,
                ess_threshold=0.5,
            )
            t1 = time.perf_counter()
            times.append(t1 - t0)

        med_time = statistics.median(times)

        # Posterior summary for x_T
        log_w = results["log_weights"]
        weights = torch.softmax(log_w, dim=0)
        x_T = results[f"x_{T}"]
        x_T_est = (weights * x_T).sum().item()

        print(
            f"Median time:      {med_time:.4f} s  (range: {min(times):.4f} – {max(times):.4f})"
        )
        print(f"Steps/sec:        {T / med_time:.0f}")
        print(
            f"Log evidence:     {results['log_evidence'].item():.4f}  (Kalman: {kf_evidence:.4f}, diff: {abs(results['log_evidence'].item() - kf_evidence):.4f})"
        )
        print(f"E[x_T]:           {x_T_est:.4f}  (Kalman: {kf_means[-1]:.4f})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    benchmark_ssm()
