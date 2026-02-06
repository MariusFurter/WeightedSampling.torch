import torch
import torch.distributions as dist
import itertools
from weighted_sampling import run_smc, sample, observe, summary, DiscreteConditional

# -------------------------------------------------------------------------
# Example: "Wet Grass" Bayesian Network
# -------------------------------------------------------------------------
# Variables: Cloudy (C), Sprinkler (S), Rain (R), WetGrass (W)

# Domains: False=0, True=1
#
# Graph:
#   C -> S
#   C -> R
#   S, R -> W

# 1. Define Probability Logic


def cloudy_probs():
    # P(C=T) = 0.5
    return [0.5, 0.5]


def sprinkler_probs(cloudy):
    # P(S=T|C=F) = 0.5
    # P(S=T|C=T) = 0.1
    if cloudy == 1:  # True
        return [0.9, 0.1]
    else:  # False
        return [0.5, 0.5]


def rain_probs(cloudy):
    # P(R=T|C=F) = 0.2
    # P(R=T|C=T) = 0.8
    if cloudy == 1:
        return [0.2, 0.8]
    else:
        return [0.8, 0.2]


def wet_grass_probs(sprinkler, rain):
    # P(W=T | S, R)
    if sprinkler == 1 and rain == 1:
        return [0.01, 0.99]
    elif sprinkler == 1 and rain == 0:
        return [0.1, 0.9]
    elif sprinkler == 0 and rain == 1:
        return [0.1, 0.9]
    else:  # S=0, R=0
        return [1.0, 0.0]


# 2. Build Conditional Distributions
# (Empty list for root nodes)
# Note: Root nodes can just be regular Categoricals, but our helper works too.
# Domain sizes: Cloudy (N/A) -> [0.5, 0.5]
# Wait, DiscreteConditional expects parents. For root, we can just use Categorical directly.
# But let's be consistent.

cpts = {
    "cloudy": DiscreteConditional(cloudy_probs, []),
    "sprinkler": DiscreteConditional(sprinkler_probs, [2]),
    "rain": DiscreteConditional(rain_probs, [2]),
    "wet_grass": DiscreteConditional(wet_grass_probs, [2, 2]),
}

# 3. Define the Probabilistic Model


def bayes_net_model(observations):
    # Sample Cloudy
    c = sample("cloudy", cpts["cloudy"])

    # Sample Sprinkler | Cloudy
    s = sample("sprinkler", cpts["sprinkler"](c))

    # Sample Rain | Cloudy
    r = sample("rain", cpts["rain"](c))

    # Observe WetGrass
    w_dist = cpts["wet_grass"](s, r)
    observe(observations["wet_grass"], w_dist)


# 4. Run Inference


def run_inference():
    print("Inference: P(Rain=True | WetGrass=True)")

    # Observations: WetGrass = True (1)
    obs = {"wet_grass": torch.tensor(1)}

    # Since run_smc calls model(data), we can just pass obs directly
    results = run_smc(bayes_net_model, obs, num_particles=5000)

    # Analyze results
    print(summary(results))
    print(results["log_weights"])


if __name__ == "__main__":
    run_inference()
