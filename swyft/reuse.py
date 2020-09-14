# pylint: disable=no-member, not-callable
from typing import Optional, Union, Callable, Collection, Tuple
from itertools import count

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
    keep_inds = masking_fn(z).all(dim=-1)
    x = x[keep_inds]
    z = z[keep_inds]
    if rounds is None:
        return x, z, None
    else:
        rounds = rounds[keep_inds]
        return x, z, rounds


class ReuseSampler(object):
    def __init__(
        self,
        data_cache: DataCache,
        target_prior: Distribution,
        epsilon: ScalarFloat,
        available: Tensor = None,
    ):
        """Utilize previously drawn data while sampling from target_prior. 
        keep is stateful, i.e. which samples were drawn is remembered.
        
        Args:
            data_cache: Stores the data from a particular round.
            target_prior: pytorch distribution.
            epsilon: Divides the probablity density into layers of epsilon height.
            available: Tensor indicating whether a certain z is already contained in the new sample.
        """
        self.x, self.z, self.rounds = cull_masked_xzrounds(
            target_prior.support.check,
            data_cache.x,
            data_cache.z,
            data_cache.rounds
        )
        self.target_prior = target_prior
        self.data_prior = data_cache.prior
        
        # self.epsilon = epsilon  # It's just confusing to have two.
        self.log_epsilon = torch.log(torch.tensor(epsilon))
        self.available = torch.ones(self.z.size(0), dtype=torch.bool, device=self.x.device) if available is None else available 

        self.data_prior_of_z = self.data_prior.log_prob(self.z)
        self.data_prior_max = self.data_prior_of_z.max()
        self.target_prior_of_z = self.target_prior.log_prob(self.z)
        self.target_prior_max = self.target_prior_of_z.max()

        self.data_prior_of_z_partition_bounds = self._get_partition(
            self.data_prior_of_z,
            self.data_prior_max,
            self.log_epsilon
        )
        
        self.previous_round = data_cache.round
        self.next_round = self.previous_round + 1
    
    @staticmethod
    def _get_partition(prior_of_z: Tensor, prior_max: ScalarFloat, log_epsilon: ScalarFloat) -> Tensor:
        """Returns the bounds of the corresponding partition by log_epsilon in the last dimension.
        There exists a p such that epsilon ** p < prior_of_z / prior_max < epsilon ** (p-1).

        Returns:
            lower_bound, upper_bound
        """
        print(prior_of_z.shape, prior_max.shape, log_epsilon.shape)
        p = 1 + ((torch.log(prior_of_z) - torch.log(prior_max)) / log_epsilon)
        p = torch.floor(p).to(torch.long)
        return prior_max * log_epsilon ** p, prior_max * log_epsilon ** (p-1)
    
    @staticmethod
    def _check_within_bounds(prior_of_z: Tensor, lower_bounds: Tensor, upper_bounds: Tensor) -> Tensor:
        """Given M prior_of_z evaluations and N bounds, return an M x N boolean array where True implies that
        prior_of_z_m is in between lower_bound_n and upper_bound_n.
        """
        assert prior_of_z.ndim == 1, f"{prior_of_z.ndim}"
        assert lower_bounds.ndim == 1, f"{lower_bounds.ndim}" 
        assert upper_bounds.ndim == 1, f"{upper_bounds.ndim}"
        assert lower_bounds.shape == upper_bounds.shape, f"lower: {lower_bounds.shape}, upper: {upper_bounds.shape}"
        
        M = prior_of_z.size(0)
        N = lower_bounds.size(0)

        is_above = prior_of_z.unsqueeze(-1).expand(M, N) > lower_bounds
        is_below = prior_of_z.unsqueeze(-1).expand(M, N) <= upper_bounds
        return is_above & is_below

    def sample(self, n: Optional[int] = None):
        # Get the new samples
        n = [1] if n is None else [n]
        target_z = self.target_prior.sample(n)
        data_prior_of_target_z = self.data_prior.log_prob(target_z)

        if not self.available.any():
            return target_z
        
        # Check if it is possible to use any old samples instead
        target_z_partition_lower, target_z_partition_upper = self._get_partition(
            data_prior_of_target_z, 
            self.data_prior_max,  # TODO, is it okay to use the data_prior_max over the data, not the actual max?
            self.log_epsilon
        )
        w = torch.rand(n).log()
        stochastic_result = w > (data_prior_of_target_z / target_z_partition_upper)
        data_prior_within_bounds = self._check_within_bounds(
            self.data_prior_of_z,
            target_z_partition_lower,
            target_z_partition_upper,
        ).t()
        # TODO this is a problem because the uniform is flat and outside support its logpdf is -inf
        # i.e. can't turn in to epsilon levels.
        print(target_z_partition_lower)
        print(target_z_partition_upper)

        samples = []
        for idx, tz, sr, wb in zip(count(), target_z, stochastic_result, data_prior_within_bounds):
            if not self.available.any():
                samples.append(tz)
            elif sr:
                samples.append(tz)
            else:
                print(self.available.any(), wb.any())
                available_and_wb = self.available & wb
                print(available_and_wb.any())
                samples.append(self.z[available_and_wb][0])
                self.available[[available_and_wb][0]] = False
        samples = torch.stack(samples)
        return torch.cat([samples, target_z[idx + 1:]])


# Christoph's example
class DataStore:
    def __init__(self, z, u, umax):
        self.z = z # samples
        self.N = len(z)  # number of samples
        self.u = u  # density function
        self.umax = umax


class Sampler(object):
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
