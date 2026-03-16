"""
Discrete Bayesian Network: "Wet Grass"
=======================================
Infer P(Rain | WetGrass=True) in the classic Wet Grass network.

Graph:
    Cloudy -> Sprinkler
    Cloudy -> Rain
    Sprinkler, Rain -> WetGrass

Domains: 0 = False, 1 = True
"""

import torch
from weighted_sampling import run_smc, sample, observe, summary, DiscreteConditional


# ---- Conditional probability tables ----


def cloudy_probs():
    # P(Cloudy=True) = 0.5
    return [0.5, 0.5]


def sprinkler_probs(cloudy):
    # P(Sprinkler=True | Cloudy=True)  = 0.1
    # P(Sprinkler=True | Cloudy=False) = 0.5
    if cloudy == 1:
        return [0.9, 0.1]
    else:
        return [0.5, 0.5]


def rain_probs(cloudy):
    # P(Rain=True | Cloudy=True)  = 0.8
    # P(Rain=True | Cloudy=False) = 0.2
    if cloudy == 1:
        return [0.2, 0.8]
    else:
        return [0.8, 0.2]


def wet_grass_probs(sprinkler, rain):
    # P(WetGrass=True | Sprinkler, Rain)
    if sprinkler == 1 and rain == 1:
        return [0.01, 0.99]
    elif sprinkler == 1 or rain == 1:
        return [0.1, 0.9]
    else:
        return [1.0, 0.0]


cpts = {
    "cloudy": DiscreteConditional(cloudy_probs, domain_sizes=[]),
    "sprinkler": DiscreteConditional(sprinkler_probs, domain_sizes=[2]),
    "rain": DiscreteConditional(rain_probs, domain_sizes=[2]),
    "wet_grass": DiscreteConditional(wet_grass_probs, domain_sizes=[2, 2]),
}


# ---- Model ----


def wet_grass_model():
    c = sample("cloudy", cpts["cloudy"]())
    s = sample("sprinkler", cpts["sprinkler"](c))
    r = sample("rain", cpts["rain"](c))

    # Condition on WetGrass = True
    observe(torch.tensor(1), cpts["wet_grass"](s, r))


# ---- Main ----

if __name__ == "__main__":
    result = run_smc(wet_grass_model, num_particles=10_000)

    stats = summary(result)
    print("Posterior given WetGrass = True:")
    print(f"  P(Cloudy)    ≈ {stats['cloudy']['mean']:.3f}")
    print(f"  P(Sprinkler) ≈ {stats['sprinkler']['mean']:.3f}")
    print(f"  P(Rain)      ≈ {stats['rain']['mean']:.3f}")
