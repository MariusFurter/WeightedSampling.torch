"""
Verification of Log Marginal Likelihood
========================================
Compare the SMC estimate of log p(y_{1:T}) against the
exact Kalman filter solution on a linear-Gaussian SSM.

Model:
    x_0 ~ Normal(0, 1)
    x_t = 0.8 * x_{t-1} + Normal(0, 0.5)
    y_t = x_t + Normal(0, 0.5)
"""

import math
import torch
import torch.distributions as dist
from weighted_sampling import run_smc, sample, observe


# ---- SMC model ----


def smc_model(data):
    x = sample("x_init", dist.Normal(0, 1))
    for t, y_val in enumerate(data):
        x = sample(f"x_{t}", dist.Normal(0.8 * x, 0.5))
        observe(y_val, dist.Normal(x, 0.5))


# ---- Exact Kalman filter ----


def kalman_filter_evidence(data):
    """Compute exact log p(y_{1:T}) via the Kalman filter."""
    F, Q = 0.8, 0.5**2  # transition coefficient, process noise variance
    H, R = 1.0, 0.5**2  # observation coefficient, observation noise variance

    # Initial belief: x_0 ~ N(0, 1)
    mu, P = 0.0, 1.0
    log_evidence = 0.0

    for y in data:
        y = y.item()

        # Predict
        mu_pred = F * mu
        P_pred = F**2 * P + Q

        # Innovation
        S = H**2 * P_pred + R
        residual = y - H * mu_pred
        log_evidence += -0.5 * (math.log(2 * math.pi) + math.log(S) + residual**2 / S)

        # Update
        K = P_pred * H / S
        mu = mu_pred + K * residual
        P = (1 - K * H) * P_pred

    return log_evidence


# ---- Synthetic data ----


def make_data(T=50):
    torch.manual_seed(42)
    x_prev = torch.randn(1).item()
    observations = []
    for _ in range(T):
        x_curr = 0.8 * x_prev + 0.5 * torch.randn(1).item()
        observations.append(x_curr + 0.5 * torch.randn(1).item())
        x_prev = x_curr
    return torch.tensor(observations)


# ---- Main ----

if __name__ == "__main__":
    data = make_data(T=50)

    # Exact
    exact = kalman_filter_evidence(data)

    # SMC estimate
    torch.manual_seed(100)
    result = run_smc(smc_model, data, num_particles=1000)
    smc = result["log_evidence"].item()

    # Compare
    diff = abs(exact - smc)
    print(f"Kalman filter (exact): {exact:.4f}")
    print(f"SMC estimate:          {smc:.4f}")
    print(f"Absolute difference:   {diff:.4f}")
    print(f"Relative error:        {diff / abs(exact) * 100:.2f}%")

    if diff < 2.0:
        print("\nSUCCESS: SMC matches Kalman filter within expected variance.")
    else:
        print("\nWARNING: Discrepancy detected.")
