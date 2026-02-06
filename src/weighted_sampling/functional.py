import torch
import inspect
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
               Also includes 'log_evidence' if not present in trace.
    """
    ctx = SMCContext(num_particles, ess_threshold)

    with SMCScope(ctx):
        model(*args, **kwargs)

    results = ctx.trace

    # Add metadata to results if keys are available
    if "log_evidence" not in results:
        results["log_evidence"] = ctx.log_evidence
    if "log_weights" not in results:
        results["log_weights"] = ctx.log_weights

    return results


def expectation(
    results: Dict[str, Any],
    test_fn: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """
    Computes the weighted expectation of a test function based on the results object.

    Args:
        results: The results dictionary returned by run_smc.
                 Must contain 'log_weights' and sampled variables.
        test_fn: A callable that accepts arguments matching the keys in results
                 (e.g. variable names) and returns a tensor.
                 The function is applied to the trace tensors (which have shape (N, ...)).

    Returns:
        The weighted expected value of the test function.
        The result will have the same shape as the output of test_fn, minus the first dimension.
    """
    if "log_weights" not in results:
        raise ValueError("Results dictionary must contain 'log_weights'.")

    log_weights = results["log_weights"]
    # Normalize weights
    weights = torch.softmax(log_weights, dim=0)

    # Inspect test_fn signature to bind arguments from results
    sig = inspect.signature(test_fn)
    kwargs = {}

    # Check if test_fn accepts **kwargs
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    if accepts_kwargs:
        # Pass everything we have
        kwargs = results
    else:
        # Only pass arguments that are requested
        for name in sig.parameters:
            if name in results:
                kwargs[name] = results[name]

    # Evaluate test_fn
    # Logic: The variables in results are (N, ...).
    # We rely on test_fn to handle vectorized inputs (standard PyTorch ops do this).
    values = test_fn(**kwargs)

    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values)

    # Compute weighted sum
    # values shape: (N, D1, D2, ...)
    # weights shape: (N,)
    # We need to broadcast weights to (N, 1, 1, ...)

    # Ensure values has at least 1 dim
    if values.ndim == 0:
        # If test_fn returns a scalar (weird if inputs are vectors), we can't really average across particles
        # unless it returned a single scalar for the whole batch?
        # But inputs are particles.
        # If inputs are (N,), output should be (N,).
        # If output is scalar conformally (e.g. user reduced it manually?), it's effectively constant?
        # Let's assume output is (N, ...) where N is num_particles.
        raise ValueError(
            f"test_fn returned a 0-d tensor. It should return a tensor with first dimension N={log_weights.shape[0]}"
        )

    if values.shape[0] != weights.shape[0]:
        raise ValueError(
            f"test_fn output shape {values.shape} does not match number of particles {weights.shape[0]}"
        )

    # Broadcast weights
    view_shape = [weights.shape[0]] + [1] * (values.ndim - 1)
    w_expanded = weights.view(*view_shape)

    return (values * w_expanded).sum(dim=0)


def summary(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Calculates weighted summary statistics for the results object.
    Includes weighted mean, weighted standard deviation, and number of unique particles
    (as a measure of diversity) for each random variable.

    Args:
        results: The results dictionary returned by run_smc.

    Returns:
        A dictionary mapping variable names to a dictionary of statistics:
        {
            'var_name': {
                'mean': tensor,
                'std': tensor,
                'n_unique': int
            },
            ...
        }
    """
    if "log_weights" not in results:
        raise ValueError("Results dictionary must contain 'log_weights'.")

    log_weights = results["log_weights"]
    weights = torch.softmax(log_weights, dim=0)

    stats = {}

    for name, value in results.items():
        # Skip metadata
        if name in ("log_weights", "log_evidence"):
            continue

        if not isinstance(value, torch.Tensor):
            continue

        # Check if first dimension matches particle count
        if value.shape[0] != weights.shape[0]:
            continue

        # 1. Weighted Mean
        # Reshape weights for broadcasting: (N, 1, 1...)
        view_shape = [weights.shape[0]] + [1] * (value.ndim - 1)
        w_expanded = weights.view(*view_shape)

        mean = (value * w_expanded).sum(dim=0)

        # 2. Weighted Standard Deviation
        # variance = sum(w * (x - mean)^2)
        # Note: using biased weighted variance for simplicity
        diff_sq = (value - mean) ** 2
        variance = (diff_sq * w_expanded).sum(dim=0)
        std = variance.sqrt()

        # 3. Diversity (Unique Values)
        # We count unique values along the particle dimension (dim=0)
        # For a scalar variable (N,), this is unique scalar values.
        # For a vector variable (N, D), this is unique vectors.
        unique_vals = torch.unique(value, dim=0)
        n_unique = unique_vals.shape[0]

        stats[name] = {
            "mean": mean,
            "std": std,
            "n_unique": n_unique,
        }

    return stats
