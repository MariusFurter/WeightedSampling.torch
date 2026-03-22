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

    @property
    def norm_weights(self) -> torch.Tensor:
        """Normalized (non-log) importance weights."""
        return torch.softmax(self["log_weights"], dim=0)

    def __repr__(self) -> str:
        lw = self["log_weights"]
        N = lw.shape[0]
        le = self["log_evidence"]
        w = torch.softmax(lw, dim=0)
        ess = (1.0 / (w * w).sum()).item()

        var_names = [
            k
            for k in self
            if k not in ("log_weights", "log_evidence")
            and isinstance(self[k], torch.Tensor)
            and self[k].shape[0] == N
        ]

        lines = [
            f"SMCResult(num_particles={N}, log_evidence={le.item():.4f}, ESS={ess:.1f})",
            f"  Variables: {', '.join(var_names)}",
        ]
        for v in var_names:
            shape_str = (
                "x".join(str(s) for s in self[v].shape[1:])
                if self[v].ndim > 1
                else "scalar"
            )
            lines.append(
                f"    {v}: shape=({N}, {shape_str})"
                if shape_str != "scalar"
                else f"    {v}: shape=({N},)"
            )
        return "\n".join(lines)

    def print_summary(self, num_bins: int = 20) -> None:
        """Print a rich tabular summary with inline histograms."""
        print(_format_summary_table(self, num_bins=num_bins, stats=self.summary()))

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
        """
        Computes the weighted expectation of a test function.

        Args:
            test_fn: A callable that accepts arguments matching variable names
                     in this result and returns a tensor of shape (N, ...).

        Returns:
            The weighted expected value, with the particle dimension summed out.
        """
        log_weights = self["log_weights"]
        weights = torch.softmax(log_weights, dim=0)

        sig = inspect.signature(test_fn)
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

        if accepts_kwargs:
            values = test_fn(**self)
        else:
            relevant_kwargs = {k: v for k, v in self.items() if k in sig.parameters}
            values = test_fn(**relevant_kwargs)

        values = torch.as_tensor(values)

        N = weights.shape[0]
        if values.ndim == 0 or values.shape[0] != N:
            raise ValueError(
                f"test_fn returned shape {values.shape}. "
                f"Expected first dimension to be N={N} (num_particles)."
            )

        w_expanded = weights.view(N, *([1] * (values.ndim - 1)))
        return (values * w_expanded).sum(dim=0)

    def summary(self) -> "SMCSummary":
        """
        Calculates weighted summary statistics (mean, std, n_unique)
        for each random variable.

        Returns:
            SMCSummary: A dict-like object mapping variable names to stat dicts.
                        Prints as a formatted table with marginal histograms.
        """
        log_weights = self["log_weights"]
        weights = torch.softmax(log_weights, dim=0)

        stats = {}
        for name, value in self.items():
            if name in ("log_weights", "log_evidence"):
                continue
            if not isinstance(value, torch.Tensor):
                continue
            if value.shape[0] != weights.shape[0]:
                continue

            view_shape = [weights.shape[0]] + [1] * (value.ndim - 1)
            w_expanded = weights.view(*view_shape)

            mean = (value * w_expanded).sum(dim=0)

            diff_sq = (value - mean) ** 2
            variance = (diff_sq * w_expanded).sum(dim=0)
            std = variance.sqrt()

            unique_vals = torch.unique(value, dim=0)
            n_unique = unique_vals.shape[0]

            stats[name] = {
                "mean": mean,
                "std": std,
                "n_unique": n_unique,
            }

        return SMCSummary(stats, self)


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
    results: SMCResult,
    test_fn: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """
    Computes the weighted expectation of a test function based on the results object.
    Convenience wrapper for ``results.expectation(test_fn)``.

    Args:
        results: The SMCResult returned by run_smc.
        test_fn: A callable that accepts arguments matching the keys in results
                 (e.g. variable names) and returns a tensor.
                 The function is applied to the trace tensors (which have shape (N, ...)).

    Returns:
        The weighted expected value of the test function.
        The result will have the same shape as the output of test_fn, minus the first dimension.
    """
    return results.expectation(test_fn)


def summary(results: SMCResult) -> "SMCSummary":
    """
    Calculates weighted summary statistics for the results object.
    Convenience wrapper for ``results.summary()``.

    Args:
        results: The SMCResult returned by run_smc.

    Returns:
        SMCSummary: A dict-like object mapping variable names to stat dicts
                    that prints as a formatted table.
    """
    return results.summary()


class SMCSummary(dict):
    """
    Weighted summary statistics for SMC results.
    Behaves like a dict mapping variable names to stat dicts,
    but prints as a formatted table with marginal histograms.
    """

    def __init__(self, stats: Dict[str, Dict[str, Any]], results: SMCResult):
        super().__init__(stats)
        self._results = results

    def __str__(self) -> str:
        return _format_summary_table(self._results, stats=self)

    def __repr__(self) -> str:
        return self.__str__()


def _spark_histogram(
    values: torch.Tensor, weights: torch.Tensor, num_bins: int, width: int
) -> List[str]:
    """
    Create a small vertical histogram using block characters.

    Returns a list of strings (one per row, from top to bottom) that form
    a histogram of `width` columns and a fixed height of 3 rows.
    """
    HEIGHT = 3
    blocks = " ▁▂▃▄▅▆▇█"

    vals_flat = values.flatten().float().contiguous()
    wts = (
        weights.repeat(vals_flat.shape[0] // weights.shape[0])
        if vals_flat.shape[0] != weights.shape[0]
        else weights
    )

    lo, hi = vals_flat.min().item(), vals_flat.max().item()
    if lo == hi:
        # Degenerate: all same value → single spike in the middle
        bar = [" " * width] * (HEIGHT - 1) + [
            " " * (width // 2) + "█" + " " * (width - width // 2 - 1)
        ]
        return bar

    edges = torch.linspace(lo, hi, num_bins + 1)
    # Bin indices for each value
    bin_idx = torch.bucketize(vals_flat, edges[1:-1])  # 0..num_bins-1
    counts = torch.zeros(num_bins)
    counts.scatter_add_(0, bin_idx, wts)

    # Resample bins to `width` columns via nearest-neighbor
    col_idx = torch.linspace(0, num_bins - 1, width).long()
    col_counts = counts[col_idx]

    mx = col_counts.max().item()
    if mx == 0:
        return [" " * width] * HEIGHT

    # Normalize to 0..HEIGHT*8 (sub-block resolution)
    scaled = (col_counts / mx * HEIGHT * 8).round().int().tolist()

    rows = []
    for row in range(HEIGHT, 0, -1):
        line = ""
        for s in scaled:
            level = s - (row - 1) * 8
            if level <= 0:
                line += " "
            elif level >= 8:
                line += blocks[8]
            else:
                line += blocks[level]
        rows.append(line)
    return rows


def _format_summary_table(
    results: SMCResult,
    num_bins: int = 20,
    stats: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """
    Build a formatted summary table string with mini-histograms above each column.
    """
    if stats is None:
        stats = summary(results)
    if not stats:
        return "No variables to summarize."

    lines: list = []

    log_weights = results["log_weights"]
    weights = torch.softmax(log_weights, dim=0)
    N = log_weights.shape[0]

    # Collect scalar entries: for multi-dim variables, flatten into separate columns
    columns: list = []  # list of (label, mean_str, std_str, n_unique_str, values_1d)
    for name, st in stats.items():
        val = results[name]
        mean_t = st["mean"]
        std_t = st["std"]
        n_unique = st["n_unique"]

        if mean_t.ndim == 0:
            # scalar variable
            columns.append(
                (
                    name,
                    f"{mean_t.item():.4f}",
                    f"{std_t.item():.4f}",
                    str(n_unique),
                    val.flatten().float(),
                )
            )
        else:
            # multi-dim: one column per element (up to 8)
            numel = mean_t.numel()
            mean_flat = mean_t.flatten()
            std_flat = std_t.flatten()
            limit = min(numel, 8)
            for i in range(limit):
                label = f"{name}[{i}]" if numel > 1 else name
                columns.append(
                    (
                        label,
                        f"{mean_flat[i].item():.4f}",
                        f"{std_flat[i].item():.4f}",
                        str(n_unique),
                        val[:, i].float() if val.ndim > 1 else val.float(),
                    )
                )
            if numel > limit:
                columns.append((f"{name}[...]", "...", "...", str(n_unique), None))

    # Determine column widths
    row_labels = ["mean", "std", "n_unique"]
    label_col_w = max(len(r) for r in row_labels) + 1  # for the row-label column

    col_widths = []
    for label, m, s, nu, _ in columns:
        w = max(len(label), len(m), len(s), len(nu), 14)
        col_widths.append(w)

    # Build histograms (3 rows each)
    HIST_ROWS = 3
    hist_blocks: list = []
    for (label, m, s, nu, vals), cw in zip(columns, col_widths):
        if vals is not None:
            hist_blocks.append(_spark_histogram(vals, weights, num_bins, cw))
        else:
            hist_blocks.append([" " * cw] * HIST_ROWS)

    # Header info
    le = results["log_evidence"]
    ess = (1.0 / (weights * weights).sum()).item()
    lines.append(f"SMC Summary (N={N}, log_evidence={le.item():.4f}, ESS={ess:.1f})")
    lines.append("")

    pad = " " * label_col_w

    # Histogram rows
    for row_i in range(HIST_ROWS):
        parts = [pad]
        for col_i, cw in enumerate(col_widths):
            parts.append(hist_blocks[col_i][row_i].ljust(cw))
        lines.append("  ".join(parts))

    # Variable name row
    parts = [pad]
    for (label, *_), cw in zip(columns, col_widths):
        parts.append(label.center(cw))
    sep_line = "──".join(["─" * label_col_w] + ["─" * cw for cw in col_widths])
    lines.append("  ".join(parts))
    lines.append(sep_line)

    # Stat rows
    for row_label in row_labels:
        parts = [row_label.rjust(label_col_w)]
        for i, (label, m, s, nu, _) in enumerate(columns):
            val_str = {"mean": m, "std": s, "n_unique": nu}[row_label]
            parts.append(val_str.center(col_widths[i]))
        lines.append("  ".join(parts))

    return "\n".join(lines)
