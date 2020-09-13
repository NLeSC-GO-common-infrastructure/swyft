# pylint: disable=no-member, not-callable
import abc
from typing import Optional, Union, Callable, Iterable, Sequence, Tuple
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions.distribution import Distribution

from .core import Mask, sample_hypercube, sample_constrained_hypercube, simulate
from .types import Array, Shape, ScalarFloat, Tensor


class Round(object):
    # The idea is to store everything in lists which are identified by rounds and the associated estimators (or model params)
    # Does it take in a simulator? It should be connected with a simulator I believe.
    def __init__(
        self,
        x: Optional[Array] = None, 
        z: Optional[Array] = None, 
        rounds: Optional[Array] = None, 
        log_likelihood_estimator: Optional[nn.Module] = None,
    ):
        super().__init__()
    
    @property
    def x(self) -> Sequence[Array]:
        raise NotImplementedError

    @property
    def z(self) -> Sequence[Array]:
        raise NotImplementedError
    
    @property
    def rounds(self) -> Sequence[Array]:
        # A sequence which shows which round a particular sample was initially drawn in. Has the same samples per round as x or z
        raise NotImplementedError
    
    @property
    def log_likelihood_estimators(self) -> Sequence[nn.Module]:
        raise NotImplementedError

    def get_dataset(
        self,
        subset_percents: Iterable[float] = (1.0,),
    ) -> Union[Dataset, Sequence[Dataset]]:
        raise NotImplementedError

    def get_dataloader(
        self,
        percent_train: Iterable[float] = (1.0,),
    ) -> Union[DataLoader, Sequence[Dataset]]:
        raise NotImplementedError

    def append(
        self,
        x: Optional[Array] = None, 
        z: Optional[Array] = None, 
        rounds: Optional[Array] = None, 
        log_likelihood_estimator: Optional[nn.Module] = None,
    ) -> None:
        # Should allow the addition of new data and estimators to the RoundCache. 
        # Perhaps some saftey checks so that the user doesn't add two sets of zs before an x or something.
        raise NotImplementedError


class PriorCache(object):
    def __init__(
        self,
        log_density
    ):
        pass


# TODO reimpelment this!
class PosteriorCache(object):
    def __init__(
        self,
        x0: Tensor,
        log_likelihood_estimator: Optional[nn.Module] = None,
        threshold: ScalarFloat = float('-inf'),  # TODO should I make this required with lle?
        mask: Callable[[Array,], Tensor] = None,
        sampler: Callable[[int, int], Array] = None,
    ):
        """Stores the iterated prior, implied mask, induced sampler."""
        self.x0 = x0

        if log_likelihood_estimator is None and mask is None and sampler is None:
            # Initial case when there is nothing learned at all
            self.log_likelihood_estimator = None
            self.threshold = None
            self.mask = None
            self.sampler = sample_hypercube
        else:
            # When there is a learned function thus a mask and thus a sampler.
            self.log_likelihood_estimator = log_likelihood_estimator
            self.threshold = threshold
            self.mask = Mask(self.log_likelihood_estimator, self.x0, self.threshold) if mask is None else mask
            self.sampler = partial(sample_constrained_hypercube, mask=self.mask) if sampler is None else sampler

    def sample(self, num_samples: int, num_params: int) -> Tensor:
        return self.sampler(num_samples, num_params)


class DataDictXZ(torch.utils.data.Dataset):
    def __init__(self, x, z):
        super().__init__()
        self.x = x
        self.z = z
        assert len(x) == len(z), f"The length of x, {len(x)}, was not equal to the length of z, {len(z)}."

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {'x': self.x[idx], 'z': self.z[idx]}


class DataCache(object):
    def __init__(
        self,
        current_round: int,
        prior: Distribution,
        x: Optional[Array] = None, 
        z: Optional[Array] = None, 
        rounds: Optional[Array] = None, 
        model: Optional[Callable[[Array,], Array]] = None,
    ):
        """Data storeage for a given round.

        Args:
            current_round: label for which round this data comes from.
            prior: p(z), pytorch prior which implements .pdf, .logpdf, and .sample
            x: samples x~p(x|z).
            z: samples z~p(z).
            rounds: array of ints identifying the round the sample was first drawn in.
            model: function implementing x = model(z).
        """
        super().__init__()
        self.round = current_round
        self.prior = prior
        self.x = x
        self.z = z
        self.rounds = rounds
        self.model = model

    def get_dataset(
        self,
        subset_percents: Iterable[float] = (1.0,),
        seed: int = None,
        init_dataset: Callable[[Array, Array,], Dataset] = DataDictXZ
    ) -> Union[Dataset, Sequence[Dataset]]:
        dataset = init_dataset(self.x, self.z)
        
        assert np.isclose(sum(subset_percents), 1.0), f"{subset_percents} does not sum to 1."
        length = len(dataset)
        subset_lengths = [int(percent * length) for percent in subset_percents]
        difference = length - sum(subset_lengths)
        assert difference >= 0, f"Your {subset_lengths} sumed to something larger than len(dataset)."
        subset_lengths[0] += difference

        generator = torch.Generator() if seed is None else torch.Generator().manual_seed(seed)
        return torch.utils.data.random_split(dataset, subset_lengths, generator=generator)


if __name__ == "__main__":
    pass
