import torch
import torch.distributions as dist
from weighted_sampling import run_smc, sample, ImportanceSampler


def importance_model():
    # Target: Normal(3, 1)
    # Proposal: Normal(0, 5) (Wider, shifted)

    target = dist.Normal(3.0, 1.0)
    proposal = dist.Normal(0.0, 5.0)

    # Use ImportanceSampler wrapper
    # logic: sample x ~ proposal
    # weight += log_target(x) - log_proposal(x)
    is_dist = ImportanceSampler(target, proposal)

    x = sample("x", is_dist)
    return x


def run_is_experiment():
    print("Running Importance Sampling Test...")
    # N needs to be high for decent IS
    trace = run_smc(importance_model, num_particles=100000)

    x_samples = trace["x"]
    mean = x_samples.mean().item()
    std = x_samples.std().item()

    print(f"Sampled Mean: {mean:.4f} (Target: 3.0)")
    print(f"Sampled Std:  {std:.4f}  (Target: 1.0)")

    if abs(mean - 3.0) < 0.1:
        print("SUCCESS: IS correctly targeted the posterior.")
    else:
        print("FAILURE: Did not match target.")


if __name__ == "__main__":
    run_is_experiment()
