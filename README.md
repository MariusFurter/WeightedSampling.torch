# WeightedSampling.torch

A lightweight, vectorized Probabilistic Programming Language (PPL) for Sequential Monte Carlo (SMC) in PyTorch.

## Features

- **Vectorized Execution**: All $N$ particles run the same model function simultaneously via PyTorch broadcasting — no Python loops over particles.
- **Sequential Monte Carlo**: ESS-triggered multinomial resampling keeps the particle population healthy across sequential observations.
- **Metropolis-Hastings Moves**: `RandomWalkProposal` and `AdaptiveProposal` (AM with particle-estimated covariance) rejuvenate particles after resampling.
- **Importance Sampling**: Plug in a separate proposal via `ImportanceSampler`; log-importance weights are accumulated automatically.
- **Discrete Models**: `DiscreteConditional` provides lazy, memoized conditional probability tables for discrete Bayesian networks.
- **PyTorch Integration**: Works with any `torch.distributions.Distribution` out of the box.
- **Analysis Utilities**: `expectation` and `summary` compute weighted posterior statistics from results.

## Installation

```bash
pip install git+https://github.com/mariusfurter/WeightedSampling.torch.git
```

## Usage

### 1. Simple Inference

Define a model as a standard Python function using `sample` and `observe`.

```python
import torch
import torch.distributions as dist
from weighted_sampling import run_smc, sample, observe, expectation

def my_model():
    # Define Prior
    # Scalar params are automatically broadcast to N particles
    mu = sample("mu", dist.Normal(0.0, 10.0))

    # Observe Data
    # 'mu' is a tensor of shape (N,)
    observe(torch.tensor(5.0), dist.Normal(mu, 1.0))

# Run Inference
result = run_smc(my_model, num_particles=1000)

# Compute weighted posterior mean
print("Posterior Mean:", expectation(result, lambda mu: mu).item())
```

### 2. Sequential Models (Random Walk)

Models can run loops. Resampling is triggered automatically when Effective Sample Size (ESS) drops.

```python
def random_walk(data):
    x = sample("x_0", dist.Normal(0.0, 1.0))
    for t, y in enumerate(data):
        x = sample(f"x_{t+1}", dist.Normal(x, 1.0))
        observe(y, dist.Normal(x, 0.5))
```

### 3. Importance Sampling

You can use a proposal distribution different from the target.

```python
from weighted_sampling import run_smc, sample, expectation, ImportanceSampler
import torch.distributions as dist

def my_model():
    # Target: Normal(3, 1), Proposal: Normal(0, 5)
    is_dist = ImportanceSampler(dist.Normal(3.0, 1.0), dist.Normal(0.0, 5.0))
    sample("x", is_dist)

result = run_smc(my_model, num_particles=10000)
print("Mean:", expectation(result, lambda x: x).item())  # ≈ 3.0
```

### 4. Utility Functions

The library provides tools to analyze the results of the inference.

**Summary Statistics:**

```python
from weighted_sampling import summary

stats = summary(trace)
print("Mean of x:", stats["x"]["mean"])
print("Std Dev of x:", stats["x"]["std"])
print("Effective unique particles:", stats["x"]["n_unique"])
```

**Expectations:**

Calculate the weighted expectation of an arbitrary function of the latent variables.

```python
from weighted_sampling import expectation

# Compute E[x^2]
expected_sq = expectation(trace, lambda x: x ** 2)
```

### 5. Discrete Models

For discrete Bayesian networks, you can use `DiscreteConditional` to define conditional probability tables efficiently.

```python
from weighted_sampling import DiscreteConditional

# P(Cloudy)
cloudy_dist = DiscreteConditional(lambda: [0.5, 0.5], domain_sizes=[])

# P(Rain | Cloudy)
def rain_probs(cloudy):
    return [0.8, 0.2] if cloudy == 0 else [0.2, 0.8]

rain_dist = DiscreteConditional(rain_probs, domain_sizes=[2])

# In model:
c = sample("cloudy", cloudy_dist())
r = sample("rain", rain_dist(c))
```

### 6. Metropolis-Hastings Moves

To improve sample diversity and mitigate degeneracy (particle collapse), you can apply MCMC moves to variables.

```python
from weighted_sampling import sample, observe, move, run_smc
from weighted_sampling import RandomWalkProposal, AdaptiveProposal
import torch
import torch.distributions as dist

def model_with_move():
    x = sample("x", dist.Normal(0.0, 1.0))
    observe(torch.tensor(5.0), dist.Normal(x, 0.1))

    # 1. Simple Random Walk Proposal
    move("x", RandomWalkProposal(scale=0.1))

    # 2. Or Adaptive Proposal based on particle covariance
    # move("x", AdaptiveProposal())
```

> **Note on Variable Names During Moves:** When using `move`, the model is replayed to compute acceptance ratios. Variable names must be unique within a single model execution once a move is initiated. Overwriting a variable name (e.g. `x = sample("val", ...); x = sample("val", ...)`) is forbidden and will raise a `ValueError`.

### 7. Model Decorator

The `@model` decorator allows a model function to be called directly with SMC arguments.

```python
from weighted_sampling import model
import torch.distributions as dist

@model
def my_model(data):
    mu = sample("mu", dist.Normal(0.0, 10.0))
    for y in data:
        observe(y, dist.Normal(mu, 1.0))

result = my_model(data, num_particles=1000, ess_threshold=0.5)
```

## Run tests

```bash
pytest tests/
```
