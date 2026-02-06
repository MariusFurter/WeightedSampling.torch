import torch
import torch.distributions as dist
from typing import Callable, List, Tuple

from .distributions import DistributionAdapter


class DiscreteConditional:
    """
    A utility to create a conditional categorical distribution from a
    function defining the probabilities.

    This class uses memoization to cache probabilities for observed parent configurations,
    avoiding the need to precompute the entire Conditional Probability Table (CPT).
    """

    def __init__(self, prob_fn: Callable, domain_sizes: List[int]):
        """
        Args:
            prob_fn: A function that takes parent values (integers) as arguments
                     and returns a list or tensor of probabilities for the child variable.
                     The function should accept arguments matching the order in domain_sizes.
            domain_sizes: A list of integers representing the size of the domain
                          for each parent variable. Used for validation.
        """
        self.prob_fn = prob_fn
        self.domain_sizes = domain_sizes
        self.cache = {}  # Cache (parent_values_tuple) -> probs_tensor

    def __call__(self, *args: torch.Tensor) -> dist.Categorical:
        """
        Returns a torch.distributions.Categorical distribution conditioned on the input parents.

        Args:
            *args: Tensor arguments for the parents.
        """
        # 1. Cast inputs to tensors (Long)
        tensors = [torch.as_tensor(a, dtype=torch.long) for a in args]

        # 2. Broadcast to find common batch shape
        # This handles scalars (0-d), vectors (N-d), and mixes elegantly.
        try:
            broadcasted = torch.broadcast_tensors(*tensors)
        except RuntimeError:
            raise ValueError(
                f"Incompatible shapes in arguments: {[t.shape for t in tensors]}"
            )

        # Handle 0 parents case (root node)
        if not broadcasted:
            return dist.Categorical(probs=self._get_probs(()))

        # 3. Stack and Flatten
        # (*batch_shape, num_parents)
        stacked = torch.stack(broadcasted, dim=-1)

        # (Total_Elements, num_parents)
        # Flatten all batch dims to iterate linearly
        flat_input = stacked.reshape(-1, stacked.shape[-1])

        # 4. Iterate and Retrieve (Optimized with Unique)
        # Identify unique parent configurations to minimize calls to prob_fn
        unique_rows, inverse_indices = torch.unique(
            flat_input, dim=0, return_inverse=True
        )

        unique_probs_list = []
        for row in unique_rows:
            key = tuple(row.tolist())
            unique_probs_list.append(self._get_probs(key))

        # Stack unique probabilities: (Num_Unique, K)
        # Ensure result is on same device as inputs
        unique_probs_stack = torch.stack(unique_probs_list).to(stacked.device)

        # Map back to full batch: (Total_Elements, K)
        flat_probs = unique_probs_stack[inverse_indices]

        # 5. Reshape Result
        # Stack -> (Total_Elements, K)
        # Reshape -> (*batch_shape, K)
        output_shape = broadcasted[0].shape + (-1,)
        probs = flat_probs.view(output_shape)

        return dist.Categorical(probs=probs)

    def _get_probs(self, key: tuple) -> torch.Tensor:
        if key not in self.cache:
            probs = self.prob_fn(*key)
            self.cache[key] = torch.as_tensor(probs, dtype=torch.float)
        return self.cache[key]

    # -------------------------------------------------------------------------
    # WeightedDistribution Protocol Implementation (for root/0-parent case)
    # -------------------------------------------------------------------------

    def sample_with_weight(
        self, num_particles: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.domain_sizes:
            raise RuntimeError(
                "Cannot sample directly from a conditional distribution with parents. "
                "Call the object with parent values first to obtain a distribution."
            )
        # Delegate to DistributionAdapter wrapping the root distribution
        return DistributionAdapter(self()).sample_with_weight(num_particles)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self.domain_sizes:
            raise RuntimeError(
                "Cannot evaluate log_prob directly on a conditional distribution with parents. "
                "Call the object with parent values first to obtain a distribution."
            )
        return self().log_prob(value)
