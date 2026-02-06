import torch
import torch.distributions as dist
import math
from weighted_sampling import run_smc, sample, observe


def make_data(T=50):
    torch.manual_seed(42)
    # Hidden State: x_t = 0.8 * x_{t-1} + N(0, 0.5^2)
    # Observation:  y_t = x_t + N(0, 0.5^2)

    true_x = []
    y = []

    # Initial state x_{-1}
    x_prev = torch.randn(1).item()

    for _ in range(T):
        # Transition
        x_curr = 0.8 * x_prev + 0.5 * torch.randn(1).item()

        # Observation
        y_val = x_curr + 0.5 * torch.randn(1).item()

        x_prev = x_curr
        y.append(y_val)

    return torch.tensor(y)


# -------------------------------------------------------------------------
# 1. SMC Model Definition
# -------------------------------------------------------------------------
def smc_model(data):
    # Latent state initialization (x_{-1})
    x = sample("x_init", dist.Normal(0, 1))

    for t, y_val in enumerate(data):
        # Transition: x_t | x_{t-1}
        x = sample(f"x_{t}", dist.Normal(0.8 * x, 0.5))

        # Observation: y_t | x_t
        observe(y_val, dist.Normal(x, 0.5))


# -------------------------------------------------------------------------
# 2. Exact Kalman Filter Calculation
# -------------------------------------------------------------------------
def kalman_filter_evidence(data):
    """
    Computes exact log p(data) for the LGSSM defined above.
    """
    # Model Parameters
    F = 0.8  # State Transition
    Q = 0.5**2  # Process Noise Variance
    H = 1.0  # Observation Matrix
    R = 0.5**2  # Observation Noise Variance

    # Initial Belief: x_init ~ N(0, 1)
    mu = 0.0
    P = 1.0

    log_evidence = 0.0

    for y in data:
        y = y.item()

        # --- Predict Step ---
        # x_t|t-1
        mu_pred = F * mu
        P_pred = F**2 * P + Q

        # --- Update Step ---
        # Innovation (residual)
        residual = y - H * mu_pred

        # Innovation Covariance S = H P H' + R
        S = H**2 * P_pred + R

        # Accumulate Log Likelihood of the innovation: log N(residual | 0, S)
        ll = -0.5 * math.log(2 * math.pi) - 0.5 * math.log(S) - 0.5 * (residual**2) / S
        log_evidence += ll

        # Kalman Gain
        K = P_pred * H / S

        # New Belief x_t|t
        mu = mu_pred + K * residual
        P = (1 - K * H) * P_pred

    return log_evidence


if __name__ == "__main__":
    data = make_data(T=50)

    print("-" * 60)
    print("Verification of Log Marginal Likelihood (Model Evidence)")
    print("-" * 60)

    # 1. Exact Answer via Kalman Filter
    exact_lml = kalman_filter_evidence(data)
    print(f"Exact Evidence (Kalman Filter):  {exact_lml:.4f}")

    # 2. Approximate Answer via SMC
    # Run with resample enabled (default)
    torch.manual_seed(100)
    results = run_smc(smc_model, data, num_particles=1000)
    smc_lml = results["log_evidence"].item()
    # 3. Calculate Error
    diff = abs(exact_lml - smc_lml)
    print(f"Absolute Difference:             {diff:.4f}")
    print(f"Relative Error:                  {diff / abs(exact_lml) * 100:.2f}%")
    print("-" * 60)

    if diff < 2.0:
        print("SUCCESS: SMC matches Kalman Filter within expected variance.")
    else:
        print("WARNING: Discrepancy detected.")
