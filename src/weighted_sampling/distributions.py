import torch
import torch.distributions as dists
from typing import Tuple, Union, Protocol, runtime_checkable, Callable


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

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the log-probability density of the target distribution.
        """
        ...


class DistributionAdapter:
    """
    Adapts a torch.distributions.Distribution to the WeightedDistribution protocol.
    Assumes proposal == target, so incremental weight is 0.
    Handles broadcasting scalar parameters to N particles.
    """

    def __init__(self, dist: dists.Distribution):
        self.dist = dist

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(value)

    def sample_with_weight(
        self, num_particles: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Try to expand to batch_shape=(num_particles,)
        # This handles:
        #   - Scalar parameters: Normal(0,1) -> expanded to particle batch
        #   - Particle parameters: Normal(randn(N),1) -> expanded/kept as particle batch
        try:
            x = self.dist.expand(torch.Size((num_particles,))).sample()
        except (RuntimeError, ValueError):
            # Fallback for incompatible shapes (e.g. Normal(randn(K), 1) where K!=N).
            # We assume the user wants N independent samples of the distribution.
            x = self.dist.sample(torch.Size((num_particles,)))

        if x.shape[0] != num_particles:
            raise RuntimeError(
                f"Sampled shape {x.shape} does not match num_particles={num_particles}."
            )

        # Incremental weight is 0 (since proposal == target)
        log_w = torch.zeros(num_particles, device=x.device)
        return x, log_w


class ImportanceSampler:
    """
    Implements Importance Sampling:
    Sample from 'proposal', Weight by 'target / proposal'.
    """

    def __init__(self, target, proposal: dists.Distribution):
        self.target = target
        self.proposal = proposal

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self.target.log_prob(value)

    def sample_with_weight(
        self, num_particles: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Sample from Proposal
        try:
            # Prefer expanding the proposal to capture particle batch semantics
            x = self.proposal.expand(torch.Size((num_particles,))).sample()
        except (RuntimeError, ValueError):
            x = self.proposal.sample(torch.Size((num_particles,)))

        if x.shape[0] != num_particles:
            raise RuntimeError(
                f"Proposal sampled shape {x.shape} does not match num_particles={num_particles}."
            )

        # 2. Compute Weights: log_w = log p(x) - log q(x)
        log_q = self.proposal.log_prob(x)
        log_p = self.target.log_prob(x)

        log_w_inc = log_p - log_q

        # Ensure weights are (N,)
        if log_w_inc.ndim == 0:
            log_w_inc = log_w_inc.expand(num_particles)
        elif log_w_inc.shape[0] != num_particles:
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

    def __init__(
        self,
        sample_fn: Callable[[int], torch.Tensor],
        log_weight_fn: Callable[[torch.Tensor], torch.Tensor],
        log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
    ):
        self.sample_fn = sample_fn
        self.log_weight_fn = log_weight_fn
        self.log_prob_fn = log_prob_fn

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self.log_prob_fn(value)

    def sample_with_weight(
        self, num_particles: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.sample_fn(num_particles)
        log_w_inc = self.log_weight_fn(x)
        return x, log_w_inc


def as_weighted(
    d: Union[WeightedDistribution, dists.Distribution],
) -> WeightedDistribution:
    """
    Factory function to normalize inputs into a WeightedDistribution.
    """
    # Check if it's already a WeightedDistribution (Structural Duck Typing)
    if isinstance(d, WeightedDistribution):
        return d

    # torch.distributions.Distribution usually satisfies requirements
    if isinstance(d, dists.Distribution):
        return DistributionAdapter(d)

    raise ValueError(
        f"Object {d} is neither a WeightedDistribution nor a torch.distributions.Distribution."
    )
