import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
from weighted_sampling import run_smc, sample, observe, summary


def model(data):
    a = sample("a", dist.Normal(0, 5))
    b = sample("b", dist.Normal(0, 5))
    for x, y in data:
        y_pred = a * x + b
        observe(y, dist.Normal(y_pred, 0.1))


def generate_synthetic_data(num_points=20):
    torch.manual_seed(0)
    x = torch.linspace(0, 10, num_points)
    true_a = 2.0
    true_b = -1.0
    true_sigma = 0.1
    y = true_a * x + true_b + torch.randn(num_points) * true_sigma
    return list(zip(x, y)), x, true_a, true_b


data, x_range, true_a, true_b = generate_synthetic_data()

results = run_smc(model, data, num_particles=1000, ess_threshold=0.5)

print(summary(results))
print(results["log_evidence"])

# Plotting
a_samples = results["a"].detach().cpu().numpy()
b_samples = results["b"].detach().cpu().numpy()
log_weights = results["log_weights"].detach().cpu()
weights = torch.exp(log_weights - log_weights.max()).numpy()

# Select 50 random traces
N = len(a_samples)
indices = torch.randperm(N)[:50].numpy()

plt.figure(figsize=(10, 6))

x_np = x_range.numpy()

# Plot selected traces (regression lines)
for idx in indices:
    # Alpha scaled by weight (max weight = 1.0)
    alpha = max(0.01, min(1.0, weights[idx]))
    y_pred = a_samples[idx] * x_np + b_samples[idx]
    plt.plot(x_np, y_pred, color="blue", alpha=alpha * 0.5)

# Plot true line
y_true = true_a * x_np + true_b
plt.plot(x_np, y_true, color="red", linewidth=2, label="True Line")

# Plot data points
x_data = [d[0].item() for d in data]
y_data = [d[1].item() for d in data]
plt.scatter(x_data, y_data, color="black", label="Data", zorder=5)

plt.title("Linear Regression: True Line and Posterior Samples")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig("linear_regression_plot.png")
print("Plot saved to linear_regression_plot.png")
