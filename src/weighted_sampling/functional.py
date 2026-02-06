import torch
import torch.distributions as dists
from typing import Callable, Any, Dict, Union

from .context import SMCContext, SMCScope, get_active_context
from .distributions import as_weighted, WeightedDistribution

# -------------------------------------------------------------------------
# Primitives
# -------------------------------------------------------------------------


def sample(
    name: str, distribution: Union[WeightedDistribution, dists.Distribution]
) -> torch.Tensor:
    """
    Samples a value for a random variable.

    Args:
        name: Unique name for this sample site.
        distribution: A WeightedDistribution or torch.distributions.Distribution object.

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


def observe(value: Any, distribution: Union[WeightedDistribution, dists.Distribution]):
    """
    Conditions the inference on an observed value.
    Updates the particle weights based on the log-probability of the observation.

    Args:
        distribution: The prior/likelihood distribution (must implement log_prob).
        value: The observed data. Can be a scalar or tensor.
    """
    ctx = get_active_context()

    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)

    # Calculate log_prob
    # If distribution is a DistributionAdapter, it creates an expanded dist inside sample(),
    # but currently its log_prob just delegates.
    # We should ensure we evaluate log_prob against the particle context if possible.

    # Adapt validation
    w_dist = as_weighted(distribution)

    # We try to trust the broadcast behavior of log_prob(value)
    # If value is (N, ...) and dist is (N, ...), we get (N,).
    # If value is scalar and dist is (N, ...), we get (N,).
    # If value is scalar and dist is scalar, we get scalar.

    log_w = w_dist.log_prob(value)

    # Ensure log_w is compatible with (N,)
    if log_w.ndim == 0:
        # Scalar weight -> apply to all particles
        ctx.log_weights = ctx.log_weights + log_w
    elif log_w.shape[0] == ctx.N:
        # Vector weight -> elementwise
        ctx.log_weights = ctx.log_weights + log_w
    else:
        # Shape mismatch (e.g. log_w is (M,) but N=100)
        # Try to expand? Only if shape[0]==1
        if log_w.shape[0] == 1:
            ctx.log_weights = ctx.log_weights + log_w.expand(ctx.N)
        else:
            raise RuntimeError(
                f"Observed log_prob shape {log_w.shape} incompatible with particles N={ctx.N}"
            )

    # 2. Resample Check
    ctx.resample_if_needed()


def deterministic(name: str, value: Any) -> torch.Tensor:
    """
    Deterministically computes a value and adds it to the trace.
    Broadcasts the value to the number of particles if necessary.
    """
    ctx = get_active_context()
    x = torch.as_tensor(value)

    # Expand if necessary to ensure first dimension is N (num_particles)
    if x.ndim == 0:
        x = x.expand(ctx.N)
    elif x.shape[0] != ctx.N:
        # Assume global parameter that needs broadcasting to particles
        # e.g. (3, 4) -> (N, 3, 4)
        x = x.unsqueeze(0).expand(ctx.N, *x.shape)

    # If x.shape[0] == ctx.N, we assume it's already a particle trace.

    ctx.trace[name] = x
    return x


# -------------------------------------------------------------------------
# Inference Loop
# -------------------------------------------------------------------------


def run_smc(
    model: Callable,
    *args,
    num_particles: int = 1000,
    ess_threshold: float = 0.5,
    **kwargs,
) -> Dict[str, Any]:
    """
    Runs Sequential Monte Carlo inference on the given model.

    Returns:
        trace: A dictionary of sampled tensors {name: tensor}.
               Also includes 'log_marginal_likelihood' if not present in trace.
    """
    ctx = SMCContext(num_particles, ess_threshold)

    with SMCScope(ctx):
        model(*args, **kwargs)

    results = ctx.trace

    # Add metadata to results if keys are available
    if "log_marginal_likelihood" not in results:
        results["log_marginal_likelihood"] = ctx.log_marginal_likelihood
    if "log_weights" not in results:
        results["log_weights"] = ctx.log_weights

    return results
