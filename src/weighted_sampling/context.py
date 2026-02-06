import threading
import torch
from typing import Optional, Callable, Tuple, Dict, Any
from torch.distributions import Categorical
from .distributions import as_weighted, WeightedDistribution

# -------------------------------------------------------------------------
# Global State Management
# -------------------------------------------------------------------------

_SMC_STACK = threading.local()


def get_active_context():
    """Returns the currently active SMCContext, or raises an error if none exists."""
    ctx = getattr(_SMC_STACK, "active_context", None)
    if ctx is None:
        raise RuntimeError(
            "No active SMC context found. "
            "Are you calling sample()/observe() inside a model function passed to run_smc()?"
        )
    return ctx


class SMCContext:
    """
    Manages the state of the Sequential Monte Carlo execution.
    storage:
        trace: dict[str, Tensor] - Stores the history of sampled variables.
                                   Shape: (N, ...)
        log_weights: Tensor      - Current log-importance weights for each particle.
                                   Shape: (N,)
    """

    def __init__(
        self,
        num_particles: int,
        ess_threshold: float = 0.5,
        track_joint: bool = False,
    ):
        self.N = num_particles
        self.ess_threshold = ess_threshold
        self.track_joint = track_joint
        self.trace = {}
        # Initialize weights to 0
        self.log_weights = torch.zeros(self.N)
        # Track joint log-probability P(x,y) separately to optimize MH moves
        # Only allocated if requested
        if self.track_joint:
            self.log_joint = torch.zeros(self.N)

        # State to allow replay
        self.model: Optional[Callable] = None
        self.args: Tuple = ()
        self.kwargs: Dict[str, Any] = {}

        # Internal counter to track 'move' calls for robust replay alignment
        self.move_counter = 0

    def sample_site(
        self, name: str, distribution: WeightedDistribution
    ) -> torch.Tensor:
        """
        Standard SMC sampling logic.
        """
        # 1. Adapt and Sample (Vectorized)
        w_dist = as_weighted(distribution)
        x, log_w_inc = w_dist.sample_with_weight(self.N)

        # 2. Update State
        self.trace[name] = x
        self.log_weights = self.log_weights + log_w_inc
        # Track model density p(x) for MH
        if self.track_joint:
            self.log_joint = self.log_joint + w_dist.log_prob(x)

        # 3. Resample Check
        self.resample_if_needed()

        return x

    def observe_site(self, value: torch.Tensor, distribution: WeightedDistribution):
        """
        Standard SMC observe logic.
        """
        w_dist = as_weighted(distribution)
        log_w = w_dist.log_prob(value)

        # Ensure log_w is compatible with (N,)
        if log_w.ndim == 0:
            if self.track_joint:
                self.log_joint = self.log_joint + log_w
        elif log_w.shape[0] == self.N:
            self.log_weights = self.log_weights + log_w
            if self.track_joint:
                self.log_joint = self.log_joint + log_w
        else:
            # Handle shape mismatch manually
            if log_w.shape[0] == 1:
                expanded = log_w.expand(self.N)
                self.log_weights = self.log_weights + expanded
                if self.track_joint:
                    self.log_weights = self.log_weights + expanded
                self.log_joint = self.log_joint + expanded
            else:
                raise RuntimeError(
                    f"Observed log_prob shape {log_w.shape} incompatible with particles N={self.N}"
                )

        # Resample Check
        self.resample_if_needed()

    def deterministic_site(self, name: str, value: Any) -> torch.Tensor:
        """
        Handles deterministic values, broadcasting them to the number of particles.
        """
        x = torch.as_tensor(value)

        # Expand if necessary to ensure first dimension is N (num_particles)
        if x.ndim == 0:
            x = x.expand(self.N)
        elif x.shape[0] != self.N:
            # Assume global parameter that needs broadcasting to particles
            # e.g. (3, 4) -> (N, 3, 4)
            x = x.unsqueeze(0).expand(self.N, *x.shape)

        self.trace[name] = x
        return x

    @property
    def log_evidence(self) -> torch.Tensor:
        """
        Estimates the log model evidence (log Z) of the model.
        calc: log( 1/N * sum(exp(log_w)) ) = logsumexp(log_w) - log(N)
        """
        return torch.logsumexp(self.log_weights, dim=0) - torch.log(
            torch.tensor(
                self.N, dtype=self.log_weights.dtype, device=self.log_weights.device
            )
        )

    def effective_sample_size(self) -> torch.Tensor:
        """
        Computes ESS = 1 / sum(w_normalized^2)
        """
        # log_w_norm = log_w - logsumexp(log_w)
        log_w_norm = torch.nn.functional.log_softmax(self.log_weights, dim=0)

        # ESS = 1 / sum(exp(log_w_norm)^2)
        #     = 1 / sum(exp(2 * log_w_norm))
        return 1.0 / torch.exp(2 * log_w_norm).sum(dim=0)

    def resample_if_needed(self):
        """
        Checks ESS and performs resampling if ESS < threshold * N.
        """
        ess = self.effective_sample_size()
        if ess < self.N * self.ess_threshold:
            self._resample()

    def _resample(self):
        """
        Performs multinomial resampling:
        1. Draw ancestor indices from categorical dist based on current weights.
        2. Permute all tensors in self.trace.
        3. Reset weights to uniform (average of current weights).
        """
        # 1. Draw Ancestors
        # Using torch.multinomial is faster than overhead of creating Categorical distribution
        # weights must be probabilities (non-negative, sum > 0)
        weights = torch.softmax(self.log_weights, dim=0)
        ancestors = torch.multinomial(weights, self.N, replacement=True)

        # 2. Permute History
        for tensor in self.trace.values():
            # In-place update to keep local variables in model synced with trace
            tensor[:] = tensor[ancestors]

        # Permute log_joint
        if self.track_joint:
            self.log_joint[:] = self.log_joint[ancestors]

        # 3. Reset Weights
        # We reset to the average log-weight to preserve the total weight mass (and thus Log Z estimate).
        log_avg_w = self.log_evidence
        self.log_weights = log_avg_w.expand(self.N).clone()


class StopReplay(Exception):
    """Internal exception to stop model replay at a specific point."""

    pass


class ConditionedContext(SMCContext):
    """
    Context for replaying a trace to compute joint log-probability P(x, y).
    Trace is fixed. Sampling sites turn into observation sites.
    No resampling performed.
    """

    def __init__(
        self, trace, num_particles: int, stop_at_move_index: Optional[int] = None
    ):
        super().__init__(num_particles, ess_threshold=-1.0, track_joint=False)
        # Deep copy traces to ensure no accidental mutations,
        # though we shouldn't be mutating anyway in replay.
        # Shallow copy of dict is enough if tensors are not mutated in place.
        self.trace = trace.copy()
        self.stop_at_move_index = stop_at_move_index
        self.visited = set()

        # Recalculate weights from scratch
        # Represent log-probs of joint density
        self.log_weights = torch.zeros(self.N)

    def sample_site(
        self, name: str, distribution: WeightedDistribution
    ) -> torch.Tensor:
        # Check for duplicates
        if name in self.visited:
            raise ValueError(
                f"Variable '{name}' already visited during replay. "
                "Variable names must be unique within a single execution."
            )
        self.visited.add(name)

        # 1. Retrieve fixed value
        if name not in self.trace:
            raise RuntimeError(f"Variable '{name}' not found in trace during replay.")
        x = self.trace[name]

        # 2. Compute Log Probability (Model Density)
        w_dist = as_weighted(distribution)
        log_prob = w_dist.log_prob(x)

        # 3. Accumulate Weight
        if log_prob.ndim == 0:
            self.log_weights = self.log_weights + log_prob
        elif log_prob.shape[0] == self.N:
            self.log_weights = self.log_weights + log_prob
        else:
            # Broadcasting if needed
            if log_prob.shape[0] == 1:
                self.log_weights = self.log_weights + log_prob.expand(self.N)
            else:
                raise RuntimeError(
                    f"Log prob shape {log_prob.shape} mismatch in replay."
                )

        return x

    def deterministic_site(self, name: str, value: Any) -> torch.Tensor:
        # Check for duplicates or overwriting sample
        if name in self.visited:
            raise ValueError(
                f"Variable '{name}' already visited during replay. "
                "Variable names must be unique within a single execution."
            )
        self.visited.add(name)

        return super().deterministic_site(name, value)

    def resample_if_needed(self):
        pass  # Disable resampling


# -------------------------------------------------------------------------
# Context Manager Utilities
# -------------------------------------------------------------------------


class SMCScope:
    """Context manager to properly set/unset the active global context."""

    def __init__(self, context: SMCContext):
        self.context = context

    def __enter__(self):
        _SMC_STACK.active_context = self.context
        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(_SMC_STACK, "active_context"):
            del _SMC_STACK.active_context
