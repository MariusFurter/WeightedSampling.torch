import torch
import torch.distributions as dists
from typing import Callable, Any, Dict, Union

from .context import SMCContext, SMCScope, get_active_context
from .distributions import as_weighted, WeightedDistribution

# -------------------------------------------------------------------------
# Primitives
# -------------------------------------------------------------------------


def sample(
    name: str, distribution: Union[dists.Distribution, WeightedDistribution]
) -> torch.Tensor:
    """
    Samples a value for a random variable.

    Args:
        name: Unique name for this sample site.
        distribution: A torch.distributions.Distribution object OR a WeightedDistribution.

    Returns:
        The sampled tensor of shape (N, ...).
    """
    ctx = get_active_context()

    # 1. Adapt and Sample (Vectorized)
    # The adapter handles broadcasting and computing incremental weights
    w_dist = as_weighted(distribution)
    x, log_w_inc = w_dist.sample_with_weight(ctx.N)

    # 2. Update State
    ctx.trace[name] = x
    ctx.log_weights = ctx.log_weights + log_w_inc

    # 3. Resample Check
    ctx.resample_if_needed()

    return x


def observe(distribution: dists.Distribution, value: Any):
    """
    Conditions the inference on an observed value.
    Updates the particle weights based on the log-probability of the observation.

    Args:
        distribution: The prior/likelihood distribution.
        value: The observed data. Can be a scalar or tensor.
               If scalar, it broadcasts to the batch shape of the distribution.
    """
    ctx = get_active_context()

    # 1. Score (Compute Log Weights)
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)

    log_w = distribution.log_prob(value)

    # constant weight update (scalar) broadcasts fine, but explicit expansion helps clarity if needed

    ctx.log_weights = ctx.log_weights + log_w

    # 2. Resample Check
    ctx.resample_if_needed()


def deterministic(name: str, value: Any) -> torch.Tensor:
    """
    Deterministically computes a value and adds it to the trace.
    Broadcasts the value to the number of particles if necessary.

    Args:
        name: Unique name for this operation.
        value: The value to record (tensor or scalar).

    Returns:
        The value as a tensor with shape (N, ...).
    """
    ctx = get_active_context()
    x = torch.as_tensor(value)

    # Expand if necessary to ensure first dimension is N (num_particles)
    if x.ndim == 0:
        x = x.expand(ctx.N)
    elif x.shape[0] != ctx.N:
        x = x.unsqueeze(0).expand(ctx.N, *x.shape)

    ctx.trace[name] = x
    return x


# -------------------------------------------------------------------------
# Inference Loop
# -------------------------------------------------------------------------


def run_smc(
    model: Callable, num_particles: int = 1000, ess_threshold: float = 0.5
) -> Dict[str, torch.Tensor]:
    """
    Runs Sequential Monte Carlo inference on the given model.

    Args:
        model: A python function taking no arguments (or handled via closure/args separately).
        num_particles: Number of particles (N) to use.
        ess_threshold: Threshold for resampling (fraction of N).

    Returns:
        trace: A dictionary of sampled tensors {name: tensor}.
    """
    ctx = SMCContext(num_particles, ess_threshold)

    with SMCScope(ctx):
        model()

    return ctx.trace
