# WeightedSampling.torch

A lightweight, vectorized Probabilistic Programming Language (PPL) for Sequential Monte Carlo (SMC) in PyTorch.

## Features

- **Vectorized Execution**: All particles are processed in parallel using PyTorch broadcasting ("Single Instruction, Multiple Data").
- **Sequential Monte Carlo**: Supports standard SMC with configurable resampling.
- **PyTorch Integration**: Built on top of `torch.distributions`.
- **Custom Inference**: Supports arbitrary importance sampling proposals via `WeightedDistribution`.

## Usage

### 1. Simple Inference

Define a model as a standard Python function using `sample` and `observe`.

```python
import torch.distributions as dist
from weighted_sampling import run_smc, sample, observe

def model():
    # Define Prior
    # Scalar params are automatically broadcast to N particles
    mu = sample("mu", dist.Normal(0.0, 10.0))

    # Observe Data
    # 'mu' is a tensor of shape (N,)
    observe(dist.Normal(mu, 1.0), torch.tensor(5.0))

    return mu

# Run Inference
trace = run_smc(model, num_particles=1000)
print("Posterior Mean:", trace["mu"].mean().item())
```

### 2. Sequential Models (Random Walk)

Models can run loops. Resampling is triggered automatically when Effective Sample Size (ESS) drops.

```python
def random_walk(data):
    x = sample("x_0", dist.Normal(0.0, 1.0))
    for t, y in enumerate(data):
        x = sample(f"x_{t+1}", dist.Normal(x, 1.0))
        observe(dist.Normal(x, 0.5), y)
```

### 3. Importance Sampling

You can use a proposal distribution different from the target.

```python
from weighted_sampling import ImportanceSampler

# Target: Normal(3, 1), Proposal: Normal(0, 5)
is_dist = ImportanceSampler(dist.Normal(3.0, 1.0), dist.Normal(0.0, 5.0))
x = sample("x", is_dist)
```

## Architecture

- **Context Stack**: Uses `threading.local` global state to manage particle traces without passing state arguments.
- **Resampling**: Checks ESS after every `sample` or `observe`. If `ESS < threshold * N`, it resamples ancestors and permutes the entire trace history.

## Installation

```bash
pip install git+https://github.com/mariusfurter/WeightedSampling.torch.git
```

## Run tests

```bash
python -m unittest discover tests
```
