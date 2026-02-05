import torch
import torch.distributions as dists
from typing import Tuple, Optional, Any


class WeightedDistribution:
    """
    Abstract base class / Protocol for distributions that support weighted sampling.
    This allows implementing Importance Sampling where proposal != target.

    If you use a standard torch.distributions.Distribution, it is assumed
    that proposal == target, and the incremental weight is 0.
    """

    @property
    def batch_shape(self) -> torch.Size:
        raise NotImplementedError

    def sample_and_log_weight(
        self, sample_shape: torch.Size = torch.Size()
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates samples and computes the incremental log-importance weight.

        Returns:
            samples: Tensor of shape sample_shape + batch_shape + event_shape
            log_weights: Tensor of shape sample_shape + batch_shape
                         (log target - log proposal)
        """
        raise NotImplementedError

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Standard sampling (discards weights)."""
        val, _ = self.sample_and_log_weight(sample_shape)
        return val


class ImportanceSampler(WeightedDistribution):
    """
    Helper to wrap a target and proposal distribution.
    target: The true distribution p(x) (must implement log_prob)
    proposal: The proposal distribution q(x) (must implement sample and log_prob)
    """

    def __init__(self, target, proposal):
        self.target = target
        self.proposal = proposal

    @property
    def batch_shape(self):
        return self.proposal.batch_shape

    def sample_and_log_weight(self, sample_shape=torch.Size()):
        # 1. Sample from proposal q(x)
        x = self.proposal.sample(sample_shape)

        # 2. Compute importance weight w = p(x) / q(x)
        # log_w = log p(x) - log q(x)
        log_q = self.proposal.log_prob(x)
        log_p = self.target.log_prob(x)

        return x, log_p - log_q
