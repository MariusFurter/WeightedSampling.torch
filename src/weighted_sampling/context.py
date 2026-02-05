import threading
import torch
from torch.distributions import Categorical

# -------------------------------------------------------------------------
# Global State Management
# -------------------------------------------------------------------------

_SMC_STACK = threading.local()


def get_active_context():
    """Returns the currently active SMCContext, or raises an error if none exists."""
    if not hasattr(_SMC_STACK, "active_context") or _SMC_STACK.active_context is None:
        raise RuntimeError(
            "No active SMC context found. "
            "Are you calling sample()/observe() inside a model function passed to run_smc()?"
        )
    return _SMC_STACK.active_context


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
        # Initialize log_weights to 0 (weights = 1)
        self.log_weights = torch.zeros(self.N)

    def effective_sample_size(self) -> torch.Tensor:
        """
        Computes ESS = (sum w)^2 / sum(w^2)
        In log domain:
            2 * log_sum_exp(log_w) - log_sum_exp(2 * log_w)
            Then exp()
        """
        # More numerically stable implementation usually normalizes weights first
        # w = exp(log_w - max(log_w))
        # ess = (sum w)^2 / sum w^2
        # But we can work in log domain:

        log_w = self.log_weights

        # log(sum(w))
        log_sum_w = torch.logsumexp(log_w, dim=0)

        # log(sum(w^2)) = log(sum(exp(log_w * 2)))
        log_sum_w_sq = torch.logsumexp(log_w * 2, dim=0)

        log_ess = 2 * log_sum_w - log_sum_w_sq
        return torch.exp(log_ess)

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
        3. Reset weights to uniform (log_weight = 0).
        """
        # 1. Draw Ancestors
        # We need normalized probabilities for Categorical
        # logits=self.log_weights is sufficient for Categorical
        ancestors = Categorical(logits=self.log_weights).sample((self.N,))

        # 2. Permute History (The "SIMD" Magic)
        for name, tensor in self.trace.items():
            # tensor is shape (N, ...). We index the first dim.
            self.trace[name] = tensor[ancestors]

        # 3. Reset Weights
        self.log_weights = torch.zeros(self.N, device=self.log_weights.device)


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
