import torch
import torch.distributions as dists
from typing import Tuple, Union, Protocol, runtime_checkable


@runtime_checkable
class WeightedDistribution(Protocol):
    """
    Protocol for distributions that support weighted sampling.
    """

    def sample_with_weight(
        self, num_particles: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates samples and computes the incremental log-importance weight.

        Args:
            num_particles: The number of particles (N) required by the SMC context.

        Returns:
            samples: Tensor of shape (N, ...)
            log_weights: Tensor of shape (N,)
        """
        ...


class StandardAdapter:
    """
    Adapts a standard torch.distributions.Distribution to the WeightedDistribution protocol.
    Assumes proposal == target, so incremental weight is 0.
    Handles broadcasting scalar parameters to N particles.
    """

    def __init__(self, dist: dists.Distribution):
        self.dist = dist

    def sample_with_weight(
        self, num_particles: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        try:
            # Attempt to expand the distribution to the desired batch size
            expanded_dist = self.dist.expand(torch.Size((num_particles,)))
            x = expanded_dist.sample()
        except RuntimeError:
            # Fallback for when expansion fails
            # In this case we blindly sample and hope the user provided correct shape.
            x = self.dist.sample()

        # Check shape safety
        if x.shape[0] != num_particles:
            raise RuntimeError(
                f"Sampled shape {x.shape} does not match num_particles={num_particles}. Check your distribution parameters."
            )

        # Incremental weight is 0
        log_w = torch.zeros(num_particles, device=x.device)
        return x, log_w


class ImportanceSampler:
    """
    Implements Importance Sampling:
    Sample from 'proposal', Weight by 'target / proposal'.
    """

    def __init__(self, target: dists.Distribution, proposal: dists.Distribution):
        self.target = target
        self.proposal = proposal

    def sample_with_weight(
        self, num_particles: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Sample from Proposal (re-use adapter logic for broadcasting)
        # We delegate to the standard adapter logic locally to be safe
        try:
            expanded_proposal = self.proposal.expand(torch.Size((num_particles,)))
            x = expanded_proposal.sample()
        except RuntimeError:
            x = self.proposal.sample()

        if x.shape[0] != num_particles:
            raise RuntimeError(
                f"Proposal sampled shape {x.shape} does not match num_particles={num_particles}."
            )

        # 2. Compute Weights
        # log_w = log p(x) - log q(x)
        # We need to score against the possibly expanded distributions to ensure (N,) output
        log_q = self.proposal.log_prob(x)
        log_p = self.target.log_prob(x)

        # Ensure weights are (N,)
        log_w_inc = log_p - log_q
        if log_w_inc.ndim == 0:
            log_w_inc = log_w_inc.expand(num_particles)
        elif log_w_inc.shape[0] != num_particles:
            # Try to broadcast if it's (1,)
            if log_w_inc.shape[0] == 1:
                log_w_inc = log_w_inc.expand(num_particles)
            else:
                raise RuntimeError(
                    f"Weight shape {log_w_inc.shape} mismatch with particles {num_particles}"
                )

        return x, log_w_inc


class CustomWeighted:
    """
    Allows creating a weighted distribution from a proposal and an arbitrary log-weight function.
    Useful when the target isn't a clean distribution object.
    """

    def __init__(self, proposal: dists.Distribution, log_weight_fn):
        self.proposal = proposal
        self.log_weight_fn = log_weight_fn

    def sample_with_weight(
        self, num_particles: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            expanded_proposal = self.proposal.expand(torch.Size((num_particles,)))
            x = expanded_proposal.sample()
        except RuntimeError:
            x = self.proposal.sample()

        if x.shape[0] != num_particles:
            raise RuntimeError(
                f"Proposal sampled shape {x.shape} does not match num_particles={num_particles}."
            )

        log_w_inc = self.log_weight_fn(x)
        return x, log_w_inc


def as_weighted(
    d: Union[dists.Distribution, WeightedDistribution],
) -> WeightedDistribution:
    """
    Factory function to normalize inputs into a WeightedDistribution.
    """
    # Check if it's already a WeightedDistribution (Structural Duck Typing)
    if isinstance(d, WeightedDistribution):
        return d

    if isinstance(d, dists.Distribution):
        return StandardAdapter(d)

    raise ValueError(
        f"Object {d} is neither a torch.Distribution nor a WeightedDistribution."
    )
