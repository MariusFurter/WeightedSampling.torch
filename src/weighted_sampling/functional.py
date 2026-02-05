import torch
import torch.distributions as dists
from typing import Callable, Any, Dict

from .context import SMCContext, SMCScope, get_active_context

# -------------------------------------------------------------------------
# Primitives
# -------------------------------------------------------------------------


def sample(name: str, distribution: dists.Distribution) -> torch.Tensor:
    """
    Samples a value for a random variable.

    Args:
        name: Unique name for this sample site.
        distribution: A torch.distributions.Distribution object.
                      If its parameters are scalars (batch_shape=()), it will be expanded to (N,).
                      If its parameters are already batched (batch_shape=(N,...)), it samples directly.

    Returns:
        The sampled tensor of shape (N, ...).
        (Note: The idea.md specifies returning *only* the sample, weights are updated internally)
    """
    ctx = get_active_context()

    # 1. Vectorized Sampling
    # Check if custom WeightedDistribution (Importance Sampling)
    if hasattr(distribution, "sample_and_log_weight"):
        # We assume the user handles batching correctly in their custom class,
        # or we might need to pass sample_shape specific to N if batch_shape is empty.
        # Minimal broadcasting logic:
        if len(distribution.batch_shape) == 0:
            sample_shape = (ctx.N,)
        else:
            sample_shape = ()

        x, log_w_inc = distribution.sample_and_log_weight(sample_shape)

        # Add incremental weights
        ctx.log_weights = ctx.log_weights + log_w_inc

    else:
        # Standard PyTorch Distribution (Proposal = Target)
        # Check if we need to broadcast to N particles
        if len(distribution.batch_shape) == 0:
            x = distribution.sample((ctx.N,))
        else:
            x = distribution.sample()

        # log_w_inc is 0 implicitly

    # 2. Update State
    ctx.trace[name] = x

    # 3. Resample Check
    ctx.resample_if_needed()

    return x


def observe(distribution: dists.Distribution, value: torch.Tensor):
    """
    Conditions the inference on an observed value.
    Updates the particle weights based on the log-probability of the observation.

    Args:
        distribution: The prior/likelihood distribution.
        value: The observed data. Can be a scalar or tensor.
               Broadcasting rules of dist.log_prob(value) apply.
    """
    ctx = get_active_context()

    # 1. Score (Compute Log Weights)
    # dist.log_prob(value) should return shape (N,) or broadcastable to it.
    # If dist params are scalar, log_prob might return scalar -> need to expand?
    # Context: if dist is scalar params, log_prob is scalar.
    # BUT, in SMC, particles have weights. Adding scalar c to all log_weights
    # doesn't change the relative weights!
    # However, usually 'value' is fixed data, but 'distribution' parameters might vary per particle.
    # OR distribution is fixed, and we just multiply weight? (pointless if constant).
    # Typically in 'observe(z | x)', 'z' is data, 'x' is latent particle.
    # So 'distribution' usually has batch_shape=(N,) because it depends on 'x'.

    log_w = distribution.log_prob(value)

    # Ensure log_w matches (N,) for safety, though tensor broadcasting usually handles add
    if log_w.ndim == 0:
        # Constant weight update for all particles (does not affect ESS, but tracks marginal likelihood)
        # We can just add it.
        pass

    ctx.log_weights = ctx.log_weights + log_w

    # 2. Resample Check
    ctx.resample_if_needed()


# -------------------------------------------------------------------------
# Inference Loop
# -------------------------------------------------------------------------


def run_smc(
    model: Callable, num_particles: int = 1000, ess_threshold: float = 0.5
) -> Dict[str, torch.Tensor]:
    """
    Runs Sequential Monte Carlo inference on the given model.

    Args:
        model: A python function taking no arguments (or handled via closure/args separately).
        num_particles: Number of particles (N) to use.
        ess_threshold: Threshold for resampling (fraction of N).

    Returns:
        trace: A dictionary of sampled tensors {name: tensor}.
    """
    ctx = SMCContext(num_particles, ess_threshold)

    with SMCScope(ctx):
        model()

    return ctx.trace
