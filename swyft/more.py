# pylint: disable=no-member
from .types import Array, Shape, ScalarFloat
from typing import Optional, Union, Callable, Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Round(object):
    # The idea is to store everything in lists which are identified by rounds and the associated estimators (or model params)
    # Does it take in a simulator? It should be connected with a simulator I believe.
    def __init__(
        self,
        x: Optional[Array] = None, 
        z: Optional[Array] = None, 
        rounds: Optional[Array] = None, 
        likelihood_estimator: Optional[nn.Module] = None,
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
    def likelihood_estimators(self) -> Sequence[nn.Module]:
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
        likelihood_estimator: Optional[nn.Module] = None,
    ) -> None:
        # Should allow the addition of new data and estimators to the RoundCache. 
        # Perhaps some saftey checks so that the user doesn't add two sets of zs before an x or something.
        raise NotImplementedError

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
        x: Optional[Array] = None, 
        z: Optional[Array] = None, 
        rounds: Optional[Array] = None, 
        likelihood_estimator: Optional[nn.Module] = None,
        model: Optional[Callable[[Array,], Array]] = None,
    ):
        """Data storeage for a given round.

        Args:
            x: samples x~p(x|z).
            z: samples z~p(z).
            rounds: array of ints identifying the round the sample was first drawn in.
            model: function implementing x = model(z).
        """
        super().__init__()
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

def get_masking_fn(likelihood_estimator: nn.Module, x0: Array, threshold: float) -> Callable[[Array,], Array]:
    # Return a function which classifies parameters as above or below the threshold, i.e. returns a boolean tensor
    raise NotImplementedError


def apply_mask(
    masking_fn: Callable[[Array,], Array], 
    x: Array,
    z: Array, 
    rounds: Optional[Array] = None
) -> Union[Tuple[Array, Array,], Tuple[Array, Array, Array]]:
    # Actually mask the parameters (and rounds if given)
    raise NotImplementedError


def sample(
    n_samples = int,
    n_dim = int,
    masking_fn: Optional[Callable[[Array,], Array]] = None,
    existing_z: Optional[Array] = None,
    existing_rounds: Optional[Array] = None,
) -> Array:
    # Start by looking at existing samples, when masking function is there use it, after that sample the hypercube with masking function
    raise NotImplementedError


def train(
    network: nn.Module, 
    train_loader: DataLoader,
    validation_loader: DataLoader,
    early_stopping_patience: int,
    max_epochs: Optional[int] = None,
    lr: float = 1e-3,
    combinations: Optional[Sequence[Sequence[int]]] = None,
    device: Union[torch.device, str] = None,
    non_blocking: bool = True,
) -> Tuple[Sequence[float], Sequence[float], dict]:
    # Given loaders and a network it returns the training stats and the best params
    # When looping over legs, consider that multiple dimension posteriors are probably lower weight than single dimension ones.
    raise NotImplementedError





if __name__ == "__main__":
    pass
