import torch
import inspect
import torch.distributions as dists
from typing import Callable, Any, Dict, Union, Tuple, Optional, List
from .context import (
    SMCContext,
    SMCScope,
    get_active_context,
)
from .distributions import WeightedDistribution

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
    x = ctx.sample_site(name, distribution)

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

    ctx.observe_site(value, distribution)


def deterministic(name: str, value: Any) -> torch.Tensor:
    """
    Deterministically computes a value and adds it to the trace.
    Broadcasts the value to the number of particles if necessary.
    """
    ctx = get_active_context()
    x = ctx.deterministic_site(name, value)
    return x


def move(
    names: Union[str, List[str]],
    proposal: Any,
    *,
    threshold: Optional[float] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Apply a Metropolis-Hastings move to variables.

    Usage:
        move("name", proposal_dist)
        move(["name1", "name2"], proposal_factory)
        move("name", proposal, threshold=0.5)

    Args:
        names: Single variable name (str) or list of names (List[str]).
        proposal: The proposal distribution or factory (must have .propose()).
        threshold: Optional override for global unique_regeneration_threshold.
    """
    ctx = get_active_context()

    # Normalize to list
    if isinstance(names, str):
        names_list = [names]
    else:
        names_list = names

    return ctx.move_site(names_list, proposal, threshold=threshold)


# -------------------------------------------------------------------------
# Inference Loop
# -------------------------------------------------------------------------


class SMCResult(dict):
    """
    Encapsulates the results of an SMC execution.
    Behaves like a dictionary but adds convenience methods for analysis.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'SMCResult' object has no attribute '{name}'")

    @property
    def log_evidence(self) -> torch.Tensor:
        return self["log_evidence"]

    @property
    def log_weights(self) -> torch.Tensor:
        return self["log_weights"]

    def sample(self, num_samples: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Resample particles using torch.multinomial according to the normalized weights.

        Args:
            num_samples: Number of particles to draw. Defaults to the current number of particles.

        Returns:
            A dictionary mapping variable names to resampled tensors.
        """
        log_weights = self["log_weights"]
        n: int = num_samples if num_samples is not None else log_weights.shape[0]
        weights = torch.softmax(log_weights, dim=0)
        indices = torch.multinomial(weights, n, replacement=True)

        resampled = {}
        for name, value in self.items():
            if name in ("log_weights", "log_evidence"):
                continue
            if (
                isinstance(value, torch.Tensor)
                and value.shape[0] == log_weights.shape[0]
            ):
                resampled[name] = value[indices]

        return resampled

    def expectation(self, test_fn: Callable[..., torch.Tensor]) -> torch.Tensor:
        return expectation(self, test_fn)

    def summary(self) -> Dict[str, Dict[str, Any]]:
        return summary(self)


def probe_model_structure(model: Callable, *args, **kwargs) -> Tuple[int, int]:
    """
    Executes the model with a limited context to count the expected number of steps
    and determine the trace buffer capacity needed.

    Args:
        model: The probabilistic model function.
        args: Arguments to pass to the model.
        kwargs: Keyword arguments to pass to the model.

    Returns:
        A tuple of (num_steps, trace_capacity).
    """
    # Use 1 particle for minimal overhead.
    # ess_threshold=-1.0 disables resampling.
    # track_joint=False eliminates joint log-prob overhead.
    # Large trace_capacity since N=1 makes it cheap.
    ctx = SMCContext(
        num_particles=1, ess_threshold=-1.0, track_joint=False, trace_capacity=65536
    )
    ctx.model = model
    ctx.args = args
    ctx.kwargs = kwargs

    steps = 0

    def counter():
        nonlocal steps
        steps += 1

    ctx.step_callback = counter

    # We assume the structure is predominantly static independent of N.
    with SMCScope(ctx):
        model(*args, **kwargs)

    return steps, ctx.trace.columns_used


def run_smc(
    model: Callable,
    *args,
    num_particles: int = 1000,
    ess_threshold: float = 0.5,
    track_joint: bool = False,
    progress_bar: bool = False,
    validate: bool = True,
    debug: bool = False,
    **kwargs,
) -> SMCResult:
    """
    Runs Sequential Monte Carlo inference on the given model.

    Args:
        model: The probabilistic model function.
        num_particles: Number of particles to use (N).
        ess_threshold: Threshold for Effective Sample Size to trigger resampling.
        track_joint: Explicitly enable tracking of joint log-probability.
                     Note: 'move' operations will automatically enable this if encountered.
        progress_bar: Whether to show a progress bar (using tqdm).
        validate: Whether to run a quick validation pass (N=1) to check model structure
                  and trace length before full execution. This enables percentage
                  reporting in the progress bar.
        debug: Whether to print verbose debug information during execution.

    Returns:
        SMCResult: An object containing sampled tensors and metadata.
                   Sampled tensors have shape (N, ...).
                   Key properties: 'log_evidence', 'log_weights'.
    """
    ctx = SMCContext(num_particles, ess_threshold, track_joint=track_joint, debug=debug)

    total_steps = None
    trace_capacity = None
    if validate:
        total_steps, trace_capacity = probe_model_structure(model, *args, **kwargs)
        ctx = SMCContext(
            num_particles,
            ess_threshold,
            track_joint=track_joint,
            debug=debug,
            trace_capacity=trace_capacity,
        )

    ctx.model = model
    ctx.args = args
    ctx.kwargs = kwargs
    ctx._total_steps = total_steps

    pbar = None
    if progress_bar:
        from tqdm import tqdm

        pbar = tqdm(total=total_steps, desc="SMC Steps")

        def update_pbar():
            pbar.update(1)

        ctx.step_callback = update_pbar

    try:
        with SMCScope(ctx):
            model(*args, **kwargs)
    finally:
        if pbar is not None:
            pbar.close()

    # Convert PackedTrace to a plain dict for the result object
    results = dict(ctx.trace.items())

    # Add metadata to results if keys are available
    if "log_evidence" not in results:
        results["log_evidence"] = ctx.log_evidence
    if "log_weights" not in results:
        results["log_weights"] = ctx.log_weights

    return SMCResult(results)


def model(func: Callable) -> Callable:
    """
    Decorator to enable running a function as a probabilistic model via SMC.

    Usage:
        @model
        def model(data):
            ...

        result = model(data, num_particles=500)
    """

    def wrapper(*args, **kwargs):
        # Extract SMC specific arguments from kwargs
        smc_params = [
            "num_particles",
            "ess_threshold",
            "track_joint",
            "progress_bar",
            "validate",
            "debug",
        ]
        smc_kwargs = {}
        model_kwargs = kwargs.copy()

        for k in smc_params:
            if k in model_kwargs:
                smc_kwargs[k] = model_kwargs.pop(k)

        return run_smc(func, *args, **smc_kwargs, **model_kwargs)

    return wrapper


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

    # Check if test_fn accepts **kwargs
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    if accepts_kwargs:
        values = test_fn(**results)
    else:
        # Only pass arguments that are requested
        relevant_kwargs = {k: v for k, v in results.items() if k in sig.parameters}
        values = test_fn(**relevant_kwargs)

    values = torch.as_tensor(values)

    # Validation
    N = weights.shape[0]
    if values.ndim == 0 or values.shape[0] != N:
        raise ValueError(
            f"test_fn returned shape {values.shape}. Expected first dimension to be N={N} (num_particles)."
        )

    # Broadcast weights to (N, 1, 1, ...) matching values dimensions
    w_expanded = weights.view(N, *([1] * (values.ndim - 1)))

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
