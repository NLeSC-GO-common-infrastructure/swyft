# pylint: disable=no-member
from typing import Optional, Union, Callable, Collection, Tuple

import torch
from torch.distributions.distribution import Distribution

from .types import Array, Shape, ScalarFloat, Tensor
from .core import get_lnL
from .more import DataCache


def cull_masked_xzrounds(
    masking_fn: Callable[[Array,], Tensor], 
    x: Array,
    z: Array, 
    rounds: Optional[Array] = None
) -> Tuple[Array, Array, Optional[Array]]:
    keep_inds = masking_fn(z)
    x = x[keep_inds]
    z = z[keep_inds]
    if rounds is None:
        return x, z, None
    else:
        rounds = rounds[keep_inds]
        return x, z, rounds


class ReuseSampler():
    def __init__(
        self,
        data_cache: DataCache,
        target_prior: Distribution,
        epsilon: ScalarFloat,
    ):
        """Utilize previously drawn data.
        
        Args:
            data_cache: Stores the data from a particular round.
            epsilon: Divides the log probablity density into layers of epsilon height.
        """
        pass


# Christoph's example
class DataStore:
    def __init__(self, z, u, umax):
        self.z = z # samples
        self.N = len(z)  # number of samples
        self.u = u  # density function
        self.umax = umax


class Sampler:
    def __init__(self, datastore, epsilon):
        self.epsilon = epsilon
        self.datastore = datastore
        self.available = np.ones(datastore.N, dtype = 'bool')
        self.ui = self.datastore.u(self.datastore.z)
    
    def __call__(self):
        zp = self._sample_prior()  # get proposal
        up = self.datastore.u(zp)
        umax = self.datastore.umax
        p, u0, u1 = self._get_partition(up, umax, self.epsilon)
        w = random.rand(1)[0]
        mask_p = (self.ui <= u1) & (self.ui > u0)
        avail_p = (mask_p & self.available).sum()
        if w > up/u1 or avail_p == 0:
            return zp
        else:
            i = list(mask_p & self.available).index(True)
            z = self.datastore.z[i]
            self.available[i] = False
            return z
    
    @staticmethod
    def _sample_prior():
        return random.rand(1)[0]
    
    def _pull(self, z):
        pass
    
    @staticmethod
    def _get_partition(up, umax, epsilon):
        """up is between epsilon^p < up/umax < epsilon^(p-1)
        
        returns:
            partition number p
            low bound
            up bound
        """
        p = np.floor(np.log(up/umax)/np.log(epsilon))+1
        p = int(p)
        return p, epsilon**p*umax, epsilon**(p-1)*umax
        


# # TODO this was before christoph clarified the algorithm to me
# # currying and partial evaluation would make this much more elegant, there must be a way...
# def reuse_and_sample_from_uniform_and_simulate(
#     masking_fn: Callable[[Array,], Tensor],
#     x: Tensor, 
#     z: Tensor, 
#     rounds: Tensor,
#     target_dist_sampler: Callable[[int, int], Tensor],
#     num_samples: int, 
#     zdim: int,
#     model: Callable[[Array], Array], 
#     current_round: int,
# ):
#     """Reuse samples which were drawn uniformly before, which lie in the support, then sample more."""
#     x, z, rounds = apply_mask_to_xzrounds(masking_fn, x, z, rounds)
#     assert z.shape[1] == zdim
#     num_drawn = x.shape[0]
#     ordering = torch.randperm(num_drawn)
#     x, z, rounds = x[ordering], z[ordering], rounds[ordering]
    
#     x, z, rounds = x[:num_samples], z[:num_samples], rounds[:num_samples]
#     num_samples -= num_drawn

#     if num_samples <= 0:
#         x, z, rounds
#     else:
#         z_new = target_dist_sampler(num_samples, zdim)
#         x_new = simulate(model, z_new)
#         rounds_new = current_round * torch.ones(num_samples, dtype=rounds.dtype, device=rounds.device)
#     x = torch.cat([x, x_new])
#     z = torch.cat([z, z_new])
#     rounds = torch.cat([rounds, rounds_new])
#     return x, z, rounds


# def sample_reuse(
#     x: Tensor, 
#     z: Tensor, 
#     log_likelihood_function: Callable[[Tensor], Tensor],
#     target_sampler: Callable[[int, int], Tensor],
# ):
#     """Given x, z along with estimated likelihoods log_prob_z, samples from target distribution with reuse.
#     It is assumed that the zs are within the support of the target distribution!!
#     """
#     assert x.shape[0] == z.shape[0]
#     assert x.shape[0] == log_prob_z.shape[0]
#     n_existing_data = x.shape[0]
#     zdim = z.shape[0]

#     log_prob_z = log_likelihood_function(z)
#     if log_prob_z.ndim == 1:
#         pass
#     elif log_prob_z.ndim == 2:
#         log_prob_z = max_log_prob_z.max(dim=-1)
#     max_log_prob_z = max_log_prob_z.max()

#     # Need to figure out what is going on with the reuse algorithm. 

#     max__log_prob_z = log_prob_z.max(dim=0)
    
#     # now that get_lnL is giving the right shape, go on implementing this.

#     keep_these = torch.zeros()
#     pass

if __name__ == "__main__":
    pass
