import threading
import torch
from torch.distributions import Categorical

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

    def __init__(self, num_particles: int, ess_threshold: float = 0.5):
        self.N = num_particles
        self.ess_threshold = ess_threshold
        self.trace = {}
        # Initialize weights to 0
        self.log_weights = torch.zeros(self.N)

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
        # Categorical expects logits (unnormalized log-probs)
        ancestors = Categorical(logits=self.log_weights).sample(torch.Size((self.N,)))

        # 2. Permute History
        for name, tensor in self.trace.items():
            # In-place update to keep local variables in model synced with trace
            tensor[:] = tensor[ancestors]

        # 3. Reset Weights
        # We reset to the average log-weight to preserve the total weight mass (and thus Log Z estimate).
        log_avg_w = self.log_evidence
        self.log_weights = log_avg_w.expand(self.N).clone()


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
