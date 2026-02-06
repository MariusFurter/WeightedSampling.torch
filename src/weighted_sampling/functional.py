import torch
import inspect
import torch.distributions as dists
from typing import Callable, Any, Dict, Union, Tuple

from .context import (
    SMCContext,
    SMCScope,
    get_active_context,
    ConditionedContext,
    StopReplay,
)
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
    return ctx.sample_site(name, distribution)


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

    ctx.observe_site(value, distribution)


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


def _replay_trace(
    trace: Dict[str, Any], ctx: SMCContext, stop_index: int
) -> torch.Tensor:
    replay_ctx = ConditionedContext(trace, ctx.N, stop_at_move_index=stop_index)
    replay_ctx.model = ctx.model
    replay_ctx.args = ctx.args
    replay_ctx.kwargs = ctx.kwargs

    if ctx.model is None:
        raise ValueError("Model is not set in the context.")

    try:
        with SMCScope(replay_ctx):
            ctx.model(*ctx.args, **ctx.kwargs)
    except StopReplay:
        pass

    # Manually restore the outer context because SMCScope cleans up aggressively
    SMCScope(ctx).__enter__()

    return replay_ctx.log_weights


def move(*args: Any, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Apply a Metropolis-Hastings move to variables.

    Usage:
        move("name", proposal_dist)
        move("name1", "name2", proposal_factory)
        move("name", proposal, threshold=0.5)

    Args:
        *args: Variable names (str) followed by a proposal distribution or factory.
        **kwargs: Optional arguments. Supported:
                  threshold (float): Override global unique_regeneration_threshold.
    """
    ctx = get_active_context()

    # Increment counter to identify this specific move call
    ctx.move_counter += 1
    current_move_index = ctx.move_counter

    # Parse arguments
    names = [arg for arg in args if isinstance(arg, str)]
    proposal = args[-1]  # Should be the last arg

    # Determine threshold
    threshold = kwargs.get("threshold", None)

    if not names:
        raise ValueError("Must provide at least one variable name to move.")

    # 0. Check context type: If we are Replaying
    if isinstance(ctx, ConditionedContext):
        if ctx.stop_at_move_index == current_move_index:
            raise StopReplay()
        # Return values for names. If multiple, return tuple.
        if len(names) == 1:
            return ctx.trace[names[0]]
        else:
            return tuple(ctx.trace[name] for name in names)

    # Check Gating Condition
    if threshold is not None:
        ratios = []
        for name in names:
            if name in ctx.trace:
                val = ctx.trace[name]
                # Calculate unique count along particle dimension
                if val.ndim > 1:
                    unique_count = len(torch.unique(val, dim=0))
                else:
                    unique_count = len(torch.unique(val))
                ratios.append(unique_count / ctx.N)

        if ratios:
            min_ratio = min(ratios)
            if min_ratio >= threshold:
                # Skip Move -> Return current values
                if len(names) == 1:
                    return ctx.trace[names[0]]
                else:
                    return tuple(ctx.trace[name] for name in names)

    # 1. Get current state
    x_old_dict = {}
    for name in names:
        if name not in ctx.trace:
            raise ValueError(
                f"Variable '{name}' not found in trace. Cannot apply move."
            )
        x_old_dict[name] = ctx.trace[name]

    # 2. Propose x_new
    if hasattr(proposal, "propose"):
        # We need weights for adaptive proposals
        weights = torch.softmax(ctx.log_weights, dim=0)
        x_new_dict = proposal.propose(x_old_dict, weights)
    else:
        raise ValueError(
            "Proposal must have 'propose' method. "
            "Use RandomWalkProposal() or AdaptiveProposal() factories."
        )

    # Validate shapes
    for name, x_new in x_new_dict.items():
        if x_new.shape != x_old_dict[name].shape:
            raise RuntimeError(
                f"Proposal sample shape {x_new.shape} mismatch with variable '{name}' shape {x_old_dict[name].shape}"
            )

    # 3. Replay Old Trace
    trace_old = ctx.trace.copy()
    log_prob_old = _replay_trace(trace_old, ctx, current_move_index)

    # 4. Replay New Trace
    trace_new = ctx.trace.copy()
    for name, x_new in x_new_dict.items():
        trace_new[name] = x_new
    log_prob_new = _replay_trace(trace_new, ctx, current_move_index)

    # 5. MH Step
    # Assume symmetric proposal for now (Metropolis)
    log_alpha = log_prob_new - log_prob_old

    # Accept/Reject
    log_u = torch.log(torch.rand(ctx.N, device=log_alpha.device))
    accepted = log_u < log_alpha

    # 6. Update Trace
    # Use x_old shape of first variable to determine broadcasting of accepted mask?
    # No, accepted is (N,). Need to broadcast per variable.

    updated_values = []
    for name in names:
        x_old = x_old_dict[name]
        x_new = x_new_dict[name]

        view_shape = [ctx.N] + [1] * (x_old.ndim - 1)
        acc_mask = accepted.view(*view_shape)

        x_final = torch.where(acc_mask, x_new, x_old)

        # Update in-place
        ctx.trace[name][:] = x_final
        updated_values.append(x_final)

    if len(updated_values) == 1:
        return updated_values[0]
    else:
        return tuple(updated_values)


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
    ctx.model = model
    ctx.args = args
    ctx.kwargs = kwargs

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
