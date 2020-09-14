# pylint: disable=no-member, not-callable
from typing import Callable, Optional

import torch

from .types import Tensor, Array


class Uniform(torch.distributions.uniform.Uniform):
    def log_prob(self, value):
        f"""{torch.distributions.uniform.Uniform.__doc__}

        Our version is modified to return torch.distributions.uniform.Uniform(value).min(dim=-1)
        """
        assert value.size(-1) == self.batch_shape[-1]
        return super().log_prob(value).min(dim=-1)[0]


def create_prior(
    zdim: int,
    lower_bound: Tensor = torch.tensor(0.0),
    upper_bound: Tensor = torch.tensor(1.0),
    size_eval_grid: int = 10000,  # TODO you could make this non-hypercubic
    masking_fn: Callable[[Tensor,], Tensor] = None,
):
    """Yields pytorch uniform prior with zdim.
    
    Args:
        ...
        masking_fn: Takes in z and returns shape [N, zdim, 1].
    """
    if masking_fn is None:
        return create_unconstraied_prior(zdim, lower_bound, upper_bound, size_eval_grid)
    else:
        return create_constrained_prior(zdim, lower_bound, upper_bound, size_eval_grid, masking_fn)


def _expand_bounds_to_zdim(
    zdim: int,
    lower_bound: Tensor,
    upper_bound: Tensor,
):
    """Check the sizes of the bounds and zdim. Expand them to match or throw an error."""
    assert lower_bound.ndim == 1 or lower_bound.ndim == 0, f"lower_bound.ndim == {lower_bound.ndim}"
    assert upper_bound.ndim == 1 or upper_bound.ndim == 0, f"upper_bound.ndim == {upper_bound.ndim}"
    assert lower_bound.shape == upper_bound.shape

    if lower_bound.shape == () or lower_bound.shape[0] == 1:
        lower_bound = lower_bound.expand(zdim)
        upper_bound = upper_bound.expand(zdim)
    assert lower_bound.shape[0] == zdim, f"lower_bound.shape == {lower_bound.shape}"
    assert upper_bound.shape[0] == zdim, f"upper_bound.shape == {upper_bound.shape}"
    
    return lower_bound, upper_bound


def create_unconstraied_prior(
    zdim: int,
    lower_bound: Tensor,
    upper_bound: Tensor,
    size_eval_grid: int, 
):
    lower_bound, upper_bound = _expand_bounds_to_zdim(
        zdim,
        lower_bound,
        upper_bound,
    )
    return Uniform(lower_bound, upper_bound)


def create_constrained_prior(
    zdim: int,
    lower_bound: Tensor,
    upper_bound: Tensor,
    size_eval_grid: int,
    masking_fn: Callable[[Tensor,], Tensor],
):
    lower_bound, upper_bound = _expand_bounds_to_zdim(
        zdim,
        lower_bound,
        upper_bound,
    )

    zgrid = torch.stack(
        [torch.linspace(low, high, size_eval_grid) for low, high in zip(lower_bound, upper_bound)],
        dim=-1
    )
    dz = torch.abs(zgrid[1, ...] - zgrid[0, ...])

    keep = masking_fn(zgrid)
    lower_bounds = []
    upper_bounds = []
    for i in range(zdim):
        bounds = zgrid[:, i][keep[:, i]]
        lower_bounds.append(bounds[0] - dz[i])
        upper_bounds.append(bounds[-1] + dz[i])
    lower_bounds = torch.stack(lower_bounds)
    upper_bounds = torch.stack(upper_bounds)

    return Uniform(lower_bounds, upper_bounds)


if __name__ == "__main__":
    pass
