import torch
import torch.distributions as dist
from typing import Dict


class Proposal:
    def propose(
        self, current_values: Dict[str, torch.Tensor], weights: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class RandomWalkProposal(Proposal):
    def __init__(self, scale: float = 0.1):
        self.scale = scale

    def propose(
        self, current_values: Dict[str, torch.Tensor], weights: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {
            k: dist.Normal(v, self.scale).sample() for k, v in current_values.items()
        }


class AdaptiveProposal(Proposal):
    def __init__(self, scale_factor: float = 2.38, epsilon: float = 1e-5):
        self.scale_factor = scale_factor
        self.epsilon = epsilon

    def propose(
        self, current_values: Dict[str, torch.Tensor], weights: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # 1. Flatten and Concatenate all variables
        # We need specific ordering. current_values dict order is preserved in Python 3.7+
        # But let's be safe and iterate keys.
        keys = list(current_values.keys())
        tensors = [current_values[k] for k in keys]
        N = tensors[0].shape[0]

        flat_tensors = []
        shapes = []
        dims = []

        for x in tensors:
            if x.shape[0] != N:
                raise ValueError("Mismatch in number of particles")
            shapes.append(x.shape)
            if x.ndim == 1:
                # (N,) -> (N, 1)
                flat_tensors.append(x.unsqueeze(1))
                dims.append(1)
            elif x.ndim == 2:
                # (N, D)
                flat_tensors.append(x)
                dims.append(x.shape[1])
            else:
                raise ValueError(
                    f"adaptive_proposal supports only 1D or 2D tensors. Got {x.shape}"
                )

        x_flat = torch.cat(flat_tensors, dim=1)  # (N, D_total)
        D_total = x_flat.shape[1]

        # 2. Compute Weighted Mean and Covariance
        w_col = weights.unsqueeze(1)  # (N, 1)
        mean = (x_flat * w_col).sum(dim=0)  # (D_total,)
        center = x_flat - mean
        cov = (center.T * weights) @ center  # (D_total, D_total)

        # 3. Adaptive Scaling
        optimal_scale = self.scale_factor / (D_total**0.5)
        mitigated_cov = (
            cov * optimal_scale
            + torch.eye(D_total, device=x_flat.device) * self.epsilon
        )

        # 4. Sample
        proposal_dist = dist.MultivariateNormal(
            loc=x_flat, covariance_matrix=mitigated_cov
        )
        x_new_flat = proposal_dist.sample()

        # 5. Split and Unflatten
        new_values = {}
        curr_idx = 0
        for i, k in enumerate(keys):
            d = dims[i]
            val = x_new_flat[:, curr_idx : curr_idx + d]
            if len(shapes[i]) == 1:
                val = val.squeeze(1)
            new_values[k] = val
            curr_idx += d

        return new_values
