# pylint: disable=no-member, not-callable
from typing import Callable, Collection, Optional, Tuple, List
from copy import deepcopy
from contextlib import nullcontext
from collections import defaultdict
from warnings import warn

import numpy as np
import torch
import torch.nn as nn

from .types import Array, Tensor, ScalarFloat, Device
from .utils import get_device_if_not_none, array_to_tensor
from .hook import Hook

#######################
# Convenience functions
#######################

def combine_z(z: Tensor, combinations: Optional[List]) -> Tensor:
    """Generate parameter combinations in last dimension using fancy indexing.
    
    Args:
        z: Parameters of shape [..., Z]
        combinations: List of parameter combinations.
    
    Returns:
        output = z[..., combinations]. When combinations is None, unsqueeze last dim.
    """
    if combinations is None:
        return z.unsqueeze(-1)
    else:
        return z[..., combinations]

def range_of_lists(stop):
    return [[item] for item in range(stop)]

#########################
# Generate sample batches
#########################

def sample_hypercube(num_samples: int, num_params: int) -> Tensor:
    """Return uniform samples from the hyper cube.

    Args:
        num_samples: number of samples.
        num_params: dimension of hypercube.

    Returns:
        Tensor: random samples.
    """
    return torch.rand(num_samples, num_params)

def simulate(model: Callable[[Array], Array], z: Tensor) -> Tensor:
    """Generates x ~ model(z).
    
    Args:
        model: forward model, returns shape (M).
        z: samples z~p(z).

    Returns:
        x: samples x~p(x|z)
    """
    num_simulations, *_  = z.shape

    if num_simulations == 0:
        warn("Simulating z with shape = (0, ...) means no simulation.")
        return torch.tensor([])
    else:
        x_list = []
        try:
            x0 = model(z[0])
            x0 = array_to_tensor(x0)
            input_device = z.device
        except TypeError:
            x0 = model(z[0].cpu())
            x0 = array_to_tensor(x0)
            input_device = torch.device('cpu')

        x_list.append(x0)
        for zz in z[1:]:
            x = model(zz.to(device=input_device))
            x = array_to_tensor(x)
            x_list.append(x)
        return torch.stack(x_list)

##########
# Training
##########

def loss_fn(network: nn.Module, x: Tensor, z: Tensor):
    """Evaluate binary-cross-entropy loss function. Mean over batch.

    Args:
        network: network taking minibatch of samples and returing ratio estimator.
        x: samples x~p(x|z)
        z: samples z~p(z).

    Returns:
        training loss.
    """ #TODO does the loss function depend on which distribution the z was drawn from? it does in SBI for the SNPE versions
    assert x.size(0) == z.size(0), "Number of x and z must be equal."
    assert x.size(0) % 2 == 0, "There must be an even number of samples in the batch for contrastive learning."
    n_batch = x.size(0)

    # Is it the removal of replacement that made it stop working?!

    # bring x into shape
    # (n_batch*2, data-shape)  - repeat twice each sample of x - there are n_batch samples
    # repetition pattern in first dimension is: [a, a, b, b, c, c, d, d, ...]
    x = torch.repeat_interleave(x, 2, dim = 0)

    # bring z into shape
    # (n_batch*2, param-shape)  - repeat twice each sample of z - there are n_batch samples
    # repetition is alternating in first dimension: [a, b, a, b, c, d, c, d, ...]
    z = z.view(n_batch // 2, -1, *z.shape[-1:])
    z = torch.repeat_interleave(z, 2, dim = 0)
    z = z.view(n_batch*2, *z.shape[-1:])
    
    # call network
    lnL = network(x, z)
    lnL = lnL.view(n_batch // 2, 4, -1)

    # Evaluate cross-entropy loss
    # loss = 
    # -ln( exp(lnL(x_a, z_a))/(1+exp(lnL(x_a, z_a))) )
    # -ln( exp(lnL(x_b, z_b))/(1+exp(lnL(x_b, z_b))) )
    # -ln( 1/(1+exp(lnL(x_a, z_b))) )
    # -ln( 1/(1+exp(lnL(x_b, z_a))) )
    loss  = -torch.nn.functional.logsigmoid( lnL[:,0])
    loss += -torch.nn.functional.logsigmoid(-lnL[:,1])
    loss += -torch.nn.functional.logsigmoid(-lnL[:,2])
    loss += -torch.nn.functional.logsigmoid( lnL[:,3])
    loss = loss.sum() / (n_batch // 2)

    return loss

# We have the posterior exactly because our proir is known and flat. Flip bayes theorem, we have the likelihood ratio.
# Consider that the variance of the loss from different legs causes some losses to have high coefficients in front of them.
def train(
    network: nn.Module, 
    train_loader: torch.utils.data.DataLoader,
    validation_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    max_epochs: Optional[int] = None,
    hooks: Optional[Collection[Hook]] = None,
    device: Optional[Device] = None,
    non_blocking: Optional[bool] = True,
):
    """Network training loop.

    Args:
        network: network for ratio estimation.
        train_loader: DataLoader of samples.
        validation_loader: DataLoader of samples.
        optimizer: Takes params and returns optimizer.
        max_epochs : Number of epochs.
        hooks: Hooks like early stopping and scheduled noise. Executed in order.
        device: Move batches to this device.
        non_blocking: non_blocking in .to(device) expression.

    Returns:
        list: list of training losses.
    """
    hooks = [] if hooks is None else hooks

    # TODO consider that the user might want other training stats, like number of correct samples for example
    def do_epoch(loader: torch.utils.data.dataloader.DataLoader, train: bool):
        accumulated_loss = 0
        training_context = nullcontext() if train else torch.no_grad()
        with training_context:
            for batch in loader:
                optimizer.zero_grad()
                
                for hook in hooks:
                    batch['x'] = hook.on_x(batch['x'], batch['z'])
                    batch['z'] = hook.on_z(batch['x'], batch['z'])

                if device is not None:
                    batch = {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}

                loss = loss_fn(network, batch['x'], batch['z'])
                if train:
                    loss.backward()
                    optimizer.step()
                accumulated_loss += loss.detach().cpu().numpy().item()
        return accumulated_loss

    max_epochs =  2 ** 31 - 1 if max_epochs is None else max_epochs
    optimizer = optimizer(network.parameters())
    
    for hook in hooks:
        hook.on_optimizer_init(optimizer)

    n_train_batches = len(train_loader)
    n_validation_batches = len(validation_loader)
    
    train_losses, validation_losses = [], []
    epoch, min_loss = 0, float("Inf")
    while epoch < max_epochs:
        print("Epoch:", epoch, validation_losses)
        network.train()
        train_loss = do_epoch(train_loader, True)
        avg_train_loss = train_loss / n_train_batches
        train_losses.append(avg_train_loss)
        
        network.eval()
        validation_loss = do_epoch(validation_loader, False)
        avg_validation_loss = validation_loss / n_validation_batches
        validation_losses.append(avg_validation_loss)

        epoch += 1
        if epoch == 0 or min_loss > avg_validation_loss:
            min_loss = avg_validation_loss
            best_state_dict = deepcopy(network.state_dict())
        
        for hook in hooks:
            hook.on_val_loss(avg_validation_loss)
        
        for hook in hooks:
            hook.on_epoch_end()

        if any(hook.stop_training for hook in hooks):
            break
    
    if epoch >= max_epochs:
        warn(f"Training finished by reaching max_epochs == {max_epochs}.")

    return train_losses, validation_losses, best_state_dict


######################
# Posterior estimation
######################

# NOTE: z combinations (with pdim > 1) should not be generated here, but just
# fed it. They can be generated externally.

def get_lnL(
    log_likelihood_estimator: nn.Module, 
    x0: Tensor, 
    z: Tensor, 
    n_batch: int = 64
):
    """Return current estimate of unnormalized marginal 1-dim lnL.
    The function can only be applied to exactly the parameter combinations that defined the likelihood_estimator.

    Args:
        log_likelihood_estimator: Has predefined possible paramter combinations.
        x0: Observation.
        z: Takes the shape (nsamples, zdim), same as how log_likelihood_estimator was defined.
        n_batch: minibatch size.

    Returns:
        lnL: (nsamples, n_posteriors)
    """
    nsamples = len(z)

    lnL = []
    for i in range(nsamples//n_batch+1):
        zbatch = z[i*n_batch:(i+1)*n_batch]
        lnL += log_likelihood_estimator(x0.unsqueeze(0), zbatch).detach().cpu()

    return torch.stack(lnL)


##########
# Networks
##########

# From: https://github.com/pytorch/pytorch/issues/36591
class LinearWithChannel(nn.Module):
    def __init__(self, input_size, output_size, channel_size):
        super(LinearWithChannel, self).__init__()

        #initialize weights
        self.w = torch.nn.Parameter(torch.zeros(channel_size, output_size, input_size))
        self.b = torch.nn.Parameter(torch.zeros(channel_size, output_size))

        #change weights to kaiming
        self.reset_parameters(self.w, self.b)

    def reset_parameters(self, weights, bias):
        torch.nn.init.kaiming_uniform_(weights, a=np.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / np.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)

    def forward(self, x):
        x = x.unsqueeze(-1)
        return torch.matmul(self.w, x).squeeze(-1) + self.b

def concatenate_data_and_parameters(x: Tensor, z: Tensor):
    """Combines data vectors x and parameter vectors z.
    
    z : (..., n_posteriors, pdim)
    x : (..., xdim)
    
    returns: (..., n_posteriors, xdim + pdim)
    
    """
    x = x.unsqueeze(-2) # (..., 1, xdim)
    x = x.expand(*z.shape[:-1], *x.shape[-1:]) # (..., n_posteriors, xdim)
    return torch.cat([x, z], -1)

class DenseLegs(nn.Module):
    def __init__(self, ydim, n_posteriors, pdim = 1, dropout_percent = 0.0, NH = 256):
        super().__init__()
        self.fc1 = LinearWithChannel(ydim+pdim, NH, n_posteriors)
        self.fc2 = LinearWithChannel(NH, NH, n_posteriors)
        self.fc3 = LinearWithChannel(NH, NH, n_posteriors)
        self.fc4 = LinearWithChannel(NH, 1, n_posteriors)
        self.drop = nn.Dropout(p = dropout_percent)

        self.af = torch.relu

        # swish activation function for smooth posteriors
        self.af2 = lambda x: x*torch.sigmoid(x)

    def forward(self, x, z):
        x = concatenate_data_and_parameters(x, z)
        x = self.af(self.fc1(x))
        x = self.drop(x)
        x = self.af(self.fc2(x))
        x = self.drop(x)
        x = self.af(self.fc3(x))
        x = self.fc4(x).squeeze(-1)
        return x

def get_norms(
    x: Array, 
    z: Array,
) -> Tuple[Array, Array, Array, Array]:
    x_mean = sum(x)/len(x)
    z_mean = sum(z)/len(z)
    x_var = sum([(x[i]-x_mean)**2 for i in range(len(x))])/len(x)
    z_var = sum([(z[i]-z_mean)**2 for i in range(len(z))])/len(z)

    print("Normalizations")
    print("x_mean", x_mean)
    print("x_err", x_var**0.5)
    print("z_mean", z_mean)
    print("z_err", z_var**0.5)

    return x_mean, x_var**0.5, z_mean, z_var**0.5

# TODO: make paramters register_buffers so that we can save them.
class Network(nn.Module):
    def __init__(
        self, 
        ydim: int, 
        combinations: List,
        head = None, 
        dropout_percent = 0., 
        datanorms = None
    ):
        """Base network combining z-independent head and parallel tail.

        Args:
            ydim: Number of data dimensions going into DenseLeg network
            combinations: List of lists of indicies to form z input to DenseLeg
                examples:
                [[0,1], [3,4]]: p(z_0,z_1) and p(z_3,z_4) are generated
                    initialize network with zdim = 2, pdim = 2
                [[0,1,5,2]]: p(z_0,z_1,z_5,z_2) is generated
                    initialize network with zdim = 1, pdim = 4
            head: Head network, z-independent
            dropout_percent: Percent to drop out on .train()
            datanorms: 

        The forward method of the `head` network takes data `x` as input, and
        returns intermediate state `y`.
        """
        super().__init__()
        self.head = head
        assert all([len(combinations[0]) == len(combo) for combo in combinations])
        n_posteriors = len(combinations)
        pdim = len(combinations[0])
        self.combinations = combinations
        self.legs = DenseLegs(ydim, n_posteriors, pdim = pdim, dropout_percent = dropout_percent)

        # Set datascaling
        if datanorms is None:
            datanorms = [torch.tensor(0.), torch.tensor(1.), torch.tensor(0.5), torch.tensor(0.5)]
        self._set_datanorms(*datanorms)

    def _set_datanorms(self, x_mean, x_std, z_mean, z_std):
        self.x_loc = torch.nn.Parameter(x_mean)
        self.x_scale = torch.nn.Parameter(x_std)
        self.z_loc = torch.nn.Parameter(z_mean)
        self.z_scale = torch.nn.Parameter(z_std)
    
    def forward(self, x, z):
        x = (x-self.x_loc)/self.x_scale
        z = (z-self.z_loc)/self.z_scale

        z = combine_z(z, self.combinations)

        if self.head is not None:
            y = self.head(x)
        else:
            y = x  # Use 1-dim data vector as features

        out = self.legs(y, z)
        return out

def sample_constrained_hypercube(num_samples: int, zdim: int, mask: Callable[[Array,], Tensor]):
    """Return uniform samples from the hyper cube.

    Args:
        num_samples: number of samples.
        zdim: dimension of hypercube.

    Returns:
        Tensor: random samples.
    """
    done = False
    zout = defaultdict(lambda: [])
    counter1 = np.zeros(zdim)  # Counter of accepted points in each z component
    counter2 = np.zeros(zdim)  # Counter of tested points in each z component
    while not done:
        z = torch.rand(num_samples, zdim)
        m = mask(z)
        for i in range(zdim):
            zout[i].append(z[m[:,i], i])
            counter1[i] += m[:,i].sum()
            counter2[i] += num_samples
        done = min(counter1) >= num_samples
    post_vol = np.true_divide(counter1, counter2)  # constrained posterior volume
    print("Constrained posterior volume:", post_vol.prod())
    
    out = torch.stack([torch.cat(zout[i]).squeeze(-1)[:num_samples] for i in range(zdim)]).T
    return out

class Mask(object):
    def __init__(
        self, 
        log_likelihood_estimator: nn.Module, 
        x0: Array, 
        threshold: ScalarFloat,
    ):
        """Classifies parameters as above or below the likelihood threshold.
        The mask can only be applied to exactly the parameter combinations that defined the likelihood_estimator.
        Mask evaluation takes place on the cpu.

        Args:
            likelihood_estimator: Takes parameters of shape [b, zdim], returns a log likelihood [b, n_posteriors].
            x0: Optimized observation.
            threshold: Mask function returns true when likelihood_estimator(z) > log(treshold) for all posteriors.
        """
        self.log_likelihood_estimator = deepcopy(log_likelihood_estimator).eval().cpu()
        self.x0 = x0
        self.threshold = threshold

    def __call__(self, z: Array) -> Tensor:
        """
        Args:
            z : (b, zdim)

        Returns:
            mask : (b, n_posteriors)
        """
        z = array_to_tensor(z)
        dtype, device = z.dtype, z.device
        z = z.cpu()
        x0 = array_to_tensor(self.x0, dtype, device='cpu')

        with torch.no_grad():
            lnL = get_lnL(self.log_likelihood_estimator, x0, z)
            lnL -= lnL.max(axis=0)[0]
            verdict = lnL > np.log(self.threshold)
            return verdict.to(device)

if __name__ == "__main__":
    pass
