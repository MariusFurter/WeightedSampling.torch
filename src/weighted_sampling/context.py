import threading
import torch
from typing import Optional, Callable, Tuple, Dict, Any, Union, List
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
        debug: bool = False,
    ):
        self.N = num_particles
        self.ess_threshold = ess_threshold
        self.track_joint = track_joint
        self.debug = debug
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

        # Optional callback for progress tracking (e.g. updating a progress bar)
        self.step_callback: Optional[Callable[[], None]] = None

    def sample_site(
        self, name: str, distribution: WeightedDistribution
    ) -> torch.Tensor:
        """
        Standard SMC sampling logic.
        """
        if self.step_callback:
            self.step_callback()

        if self.debug:
            print(f"[DEBUG] Sample '{name}'")

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
        if self.step_callback:
            self.step_callback()

        if self.debug:
            print(f"[DEBUG] Observe {value:.2f}")

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

    def _replay_trace(self, trace: Dict[str, Any], stop_index: int) -> torch.Tensor:
        """Helper to replay a trace and compute log-density."""
        replay_ctx = ConditionedContext(trace, self.N, stop_at_move_index=stop_index)
        replay_ctx.model = self.model
        replay_ctx.args = self.args
        replay_ctx.kwargs = self.kwargs

        if self.model is None:
            raise ValueError("Model is not set in the context.")

        try:
            with SMCScope(replay_ctx):
                self.model(*self.args, **self.kwargs)
        except StopReplay:
            pass

        # Use explicitly the current context in SMCScope to restore it
        # Note: SMCScope __exit__ deletes the active_context, so we must restore self.
        _SMC_STACK.active_context = self

        return replay_ctx.log_weights

    def _should_skip_move(self, names: List[str], threshold: Optional[float]) -> bool:
        if threshold is None:
            return False

        ratios = []
        for name in names:
            if name in self.trace:
                val = self.trace[name]
                # Calculate unique count along particle dimension
                if val.ndim > 1:
                    unique_count = len(torch.unique(val, dim=0))
                else:
                    unique_count = len(torch.unique(val))
                ratios.append(unique_count / self.N)

        if ratios:
            min_ratio = min(ratios)
            return min_ratio >= threshold
        return False

    def _propose_new_values(
        self, proposal: Any, x_old_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if hasattr(proposal, "propose"):
            # We need weights for adaptive proposals
            weights = torch.softmax(self.log_weights, dim=0)
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
        return x_new_dict

    def _update_trace_with_move(
        self,
        names: List[str],
        x_old_dict: Dict[str, torch.Tensor],
        x_new_dict: Dict[str, torch.Tensor],
        accepted: torch.Tensor,
    ) -> List[torch.Tensor]:
        updated_values = []
        for name in names:
            x_old = x_old_dict[name]
            x_new = x_new_dict[name]

            view_shape = [self.N] + [1] * (x_old.ndim - 1)
            acc_mask = accepted.view(*view_shape)

            x_final = torch.where(acc_mask, x_new, x_old)

            # Update in-place
            self.trace[name][:] = x_final
            updated_values.append(x_final)
        return updated_values

    def move_site(
        self,
        names: List[str],
        proposal: Any,
        threshold: Optional[float] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply a Metropolis-Hastings move to variables.
        """
        if self.step_callback:
            self.step_callback()

        if self.debug:
            print(f"[DEBUG] Move attempt on {names}")

        # Increment counter to identify this specific move call
        self.move_counter += 1
        current_move_index = self.move_counter

        if not names:
            raise ValueError("Must provide at least one variable name to move.")

        # Check Gating Condition
        if self._should_skip_move(names, threshold):
            if len(names) == 1:
                return self.trace[names[0]]
            else:
                return tuple(self.trace[name] for name in names)

        # 1. Get current state
        x_old_dict = {}
        for name in names:
            if name not in self.trace:
                raise ValueError(
                    f"Variable '{name}' not found in trace. Cannot apply move."
                )
            x_old_dict[name] = self.trace[name]

        # 2. Propose x_new
        x_new_dict = self._propose_new_values(proposal, x_old_dict)

        # 3. Replay Old Trace
        # Optimization: Use cached log_joint if available to avoid redundant replay
        # If not tracking, we perform the replay once, but enable tracking for future moves
        if hasattr(self, "log_joint"):
            log_prob_old = self.log_joint.clone()
        else:
            trace_old = self.trace.copy()
            log_prob_old = self._replay_trace(trace_old, current_move_index)

            # Enable tracking for future moves (Auto-detect optimization)
            self.track_joint = True
            self.log_joint = log_prob_old.clone()

        # 4. Replay New Trace
        trace_new = self.trace.copy()
        for name, x_new in x_new_dict.items():
            trace_new[name] = x_new
        log_prob_new = self._replay_trace(trace_new, current_move_index)

        # 5. MH Step
        # Assume symmetric proposal for now (Metropolis)
        log_alpha = log_prob_new - log_prob_old

        # Accept/Reject
        log_u = torch.log(torch.rand(self.N, device=log_alpha.device))
        accepted = log_u < log_alpha

        # 6. Update Trace
        updated_values = self._update_trace_with_move(
            names, x_old_dict, x_new_dict, accepted
        )

        # Update log_joint if tracked
        if hasattr(self, "log_joint"):
            self.log_joint[:] = torch.where(accepted, log_prob_new, self.log_joint)

        if len(updated_values) == 1:
            return updated_values[0]
        else:
            return tuple(updated_values)

    def _resample(self):
        """
        Performs multinomial resampling:
        1. Draw ancestor indices from categorical dist based on current weights.
        2. Permute all tensors in self.trace.
        3. Reset weights to uniform (average of current weights).
        """
        if self.debug:
            ess = self.effective_sample_size().item()
            print(f"[DEBUG] Resampling triggered (ESS={ess:.2f})")

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

    def move_site(
        self,
        names: List[str],
        proposal: Any,
        threshold: Optional[float] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        In replay mode, 'move' acts as a stopping point or a no-op retrieval.
        """
        self.move_counter += 1
        current_move_index = self.move_counter

        if self.stop_at_move_index == current_move_index:
            raise StopReplay()

        # Return values for names. If multiple, return tuple.
        if len(names) == 1:
            return self.trace[names[0]]
        else:
            return tuple(self.trace[name] for name in names)

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
