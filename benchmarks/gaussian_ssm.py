import torch
import torch.distributions as dist
import time
from weighted_sampling import run_smc, sample, observe


def generate_data(num_timesteps):
    torch.manual_seed(42)
    x = torch.zeros(num_timesteps)
    y = torch.zeros(num_timesteps)
    x_curr = torch.tensor(0.0)
    for t in range(num_timesteps):
        x_curr = x_curr + torch.randn(1)
        x[t] = x_curr
        y[t] = x_curr + torch.randn(1)
    return y


def ssm_model(data):
    # Standard Bootstrap Particle Filter for Gaussian SSM
    x_curr = sample("x_0", dist.Normal(0.0, 1.0))
    for t, y in enumerate(data):
        # State transition: x_t ~ N(x_{t-1}, 1)
        x_curr = sample(f"x_{t+1}", dist.Normal(x_curr, 1.0))
        # Observation: y_t ~ N(x_t, 1)
        observe(y, dist.Normal(x_curr, 1.0))


def benchmark_ssm():
    # Parameters designed to stress resampling and step overhead
    num_particles = 10000
    num_timesteps = 100
    data = generate_data(num_timesteps)

    print(f"Running Gaussian SSM Benchmark...")
    print(f"Particles: {num_particles}")
    print(f"Timesteps: {num_timesteps}")

    start_time = time.time()
    results = run_smc(ssm_model, data, num_particles=num_particles, ess_threshold=0.5)
    end_time = time.time()

    duration = end_time - start_time
    print(f"Done.")
    print(f"Total Time: {duration:.4f}s")
    print(f"Steps/sec: {num_timesteps / duration:.2f}")
    print(f"Log Evidence: {results['log_evidence'].item():.4f}")


if __name__ == "__main__":
    benchmark_ssm()
