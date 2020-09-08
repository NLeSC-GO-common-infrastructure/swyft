# pylint: disable=no-member, not-callable
from copy import deepcopy
from functools import partial
from typing import Union
from warnings import warn

import numpy as np
from scipy.integrate import trapz

import torch
import torch.nn as nn

from .core import *
from .more import DataDictXZ
from .utils import array_to_tensor
from .hook import ReduceLROnPlateau, EarlyStopping
from .types import Device

class Data(torch.utils.data.Dataset):
    """Data container class.

    Note: The noisemodel allows scheduled noise level increase during training.
    """
    def __init__(self, xz):
        super().__init__()
        self.xz = xz
        self.noisemodel = None

    def set_noisemodel(self, noisemodel):
        self.noisemodel = noisemodel
        self.noiselevel = 1.  # 0: no noise, 1: full noise

    def set_noiselevel(self, level):
        self.noiselevel = level

    def __len__(self):
        return len(self.xz)

    def __getitem__(self, idx):
        xz = self.xz[idx]
        if self.noisemodel is not None:
            x = self.noisemodel(xz['x'].numpy(), z = xz['z'].numpy(), noiselevel = self.noiselevel)
            x = torch.tensor(x).float()
            xz = dict(x=x, z=xz['z'])
        return xz

def gen_train_data(model, nsamples, zdim, mask = None):
    # Generate training data
    if mask is None:
        z = sample_hypercube(nsamples, zdim)
    else:
        z = sample_constrained_hypercube(nsamples, zdim, mask)
    
    x = simulate(model, z)
    dataset = DataDictXZ(x, z)
    
    return dataset

def trainloop(
    net, 
    dataset: torch.utils.data.Dataset, 
    batch_size: int = 8, 
    max_epochs: int = 100, 
    lr: float = 1e-3,
    min_lr: Union[float, Collection[float]] = 0, 
    early_stopping_patience: int = 20, 
    reduce_lr_patience: int = 10,
    reduce_lr_factor: int = 0.1,
    device: Device = 'cpu', 
    num_workers: int = 4,
    nl_schedule = [0.1, 0.3, 1.0]
) -> None:
    nvalid = 512
    ntrain = len(dataset) - nvalid
    dataset_train, dataset_valid = torch.utils.data.random_split(dataset, [ntrain, nvalid])
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True)
    
    optimizer = partial(torch.optim.Adam, lr=1e-3)
    hooks = [
        ReduceLROnPlateau(
            patience=reduce_lr_patience,
            factor=reduce_lr_factor, 
        ),
        EarlyStopping(
            patience=early_stopping_patience
        ),
        # StepNoise()
    ]

    train_loss, valid_loss, best_state_dict = train(
            net, 
            train_loader, 
            valid_loader,
            optimizer,
            max_epochs=max_epochs, 
            hooks=hooks,
            device=device, 
        )
    net.load_state_dict(best_state_dict)

def get_posterior_per_leg(x0, net, dataset, device = 'cpu'):
    """The function returns parameters as seen by the network and lnL as estimated by the network."""
    x0 = x0.to(device)
    z = dataset.z.to(device)
    net = net.to(device)
    lnL = get_lnL(net, x0, z)

    # Save the zs are they are fed into the legs network
    z = combine_z(z, net.combinations)
    return z.cpu(), lnL.cpu()

class SWYFT:
    def __init__(self, x0, model, zdim, head = None, noisemodel = None, device = 'cpu'):
        self.x0 = torch.tensor(x0).float()
        self.model = model
        self.noisemodel = noisemodel
        self.zdim = zdim
        self.head_cls = head  # head network class
        self.device = device

        # Each data_store entry has a corresponding mask entry
        # TODO: Replace with datastore eventually
        self.mask_store = []
        self.data_store = []

        # NOTE: Each trained network goes together with evaluated posteriors (evaluated on x0)
        self.post1d_store = []
        self.net1d_store = []

        # NOTE: We separate N-dim posteriors since they are not used (yet) for refining training data
        self.postNd_store = []
        self.netNd_store = []

    def _get_net(self, combinations, head = None, datanorms = None):
        # Initialize neural network
        if self.head_cls is None and head is None:
            head = None
            ydim = len(self.x0)
        elif head is not None:
            ydim = head(self.x0.unsqueeze(0).to(self.device)).shape[1]
            print("Number of output features:", ydim)
        else:
            head = self.head_cls()
            ydim = head(self.x0.unsqueeze(0)).shape[1]
            print("Number of output features:", ydim)
        net = Network(ydim=ydim, combinations=combinations, head=head, datanorms = datanorms).to(self.device)
        return net

    def append_dataset(self, dataset):
        """Append dataset to data_store, assuming unconstrained prior."""
        self.data_store.append(dataset)
        self.mask_store.append(None)

    def train1d(self, recycle_net = True, max_epochs = 100, nbatch = 8): 
        """Train 1-dim posteriors."""
        # Use most recent dataset by default
        dataset = self.data_store[-1]

        datanorms = get_norms(dataset.x, dataset.z)

        # Start by retraining previous network
        if len(self.net1d_store) > 0 and recycle_net:
            net = deepcopy(self.net1d_store[-1])
        else:
            net = self._get_net(range_of_lists(self.zdim), datanorms = datanorms)

        # Train
        trainloop(
            net, 
            dataset, 
            device = self.device, 
            max_epochs = max_epochs, 
            batch_size = nbatch
        )

        # Get 1-dim posteriors
        zgrid, lnLgrid = get_posterior_per_leg(self.x0, net, dataset, device = self.device)

        # Store results
        self.net1d_store.append(net)
        self.post1d_store.append((zgrid, lnLgrid))

    def data(self, nsamples = 3000, threshold = 1e-6):
        """Generate training data on constrained prior."""
        if len(self.mask_store) == 0:
            mask = None
        else:
            last_net = self.net1d_store[-1]
            mask = Mask(last_net, self.x0.to(self.device), threshold)

        dataset = gen_train_data(self.model, nsamples, self.zdim, mask = mask)
        # dataset.set_noisemodel(self.noisemodel)

        # Store dataset and mask
        self.mask_store.append(mask)
        self.data_store.append(dataset)

    def run(self, nrounds = 1, nsamples = 3000, threshold = 1e-6, max_epochs = 100, recycle_net = True, nbatch = 8):
        """Iteratively generating training data and train 1-dim posteriors."""
        for _ in range(nrounds):
            if self.model is None:
                warn("No model provided. Skipping data generation.")
            else:
                self.data(nsamples = nsamples, threshold = threshold)
            self.train1d(recycle_net = recycle_net, max_epochs = max_epochs, nbatch = nbatch)

    def comb(self, combinations, max_epochs = 100, recycle_net = True, nbatch = 8):
        """Generate N-dim posteriors."""
        # Use by default data from last 1-dim round
        dataset = self.data_store[-1]

        # Generate network
        if recycle_net:
            head = deepcopy(self.net1d_store[-1].head)
            net = self._get_net(combinations, head = head)
        else:
            net = self._get_net(combinations)

        # Train!
        trainloop(
            net, 
            dataset, 
            device = self.device, 
            max_epochs = max_epochs, 
            batch_size = nbatch,
        )

        # Get posteriors and store them internally
        zgrid, lnLgrid = get_posterior_per_leg(
            self.x0, 
            net, 
            dataset, 
            device=self.device
        )

        self.postNd_store.append((combinations, zgrid, lnLgrid))
        self.netNd_store.append(net)

    def posterior(self, indices, version = -1):
        """Return generated posteriors."""
        # NOTE: 1-dim posteriors are automatically normalized
        # TODO: Normalization should be done based on prior range, not enforced by hand
        # TODO: we need normalization... And I believe that the multi target training is going to cause
        # problems in regard to loss functions. (two peaks shaped correctly of different magnitudes)
        if isinstance(indices, int):
            i = indices
            # Sort for convenience
            z = self.post1d_store[version][0][:, i, 0]
            x = self.post1d_store[version][1][:, i]
            isorted = np.argsort(z)
            z, x = z[isorted], x[isorted]
            x = np.exp(x)
            I = trapz(x, z)
            return z, x/I
        else:
            for i in reversed(range(len(self.postNd_store))):
                combinations = self.postNd_store[i][0]
                if indices in combinations:
                    j = combinations.index(indices)
                    return self.postNd_store[i][1][:, j], self.postNd_store[i][2][:, j]
            warn("Did not find requested parameter combination.")
            return None
