# pylint: disable=no-member, not-callable
from pathlib import Path
from warnings import warn

import numpy as np
import scipy
from scipy.stats import norm
from scipy.integrate import simps
import torch
from torch import nn

_VERBOSE = 1


def set_verbosity(v):
    global _VERBOSE
    _VERBOSE = v


def verbosity():
    global _VERBOSE
    return _VERBOSE


from .types import (
    Array,
    Combinations,
    Device,
    Dict,
    List,
    Optional,
    PathType,
    Sequence,
    Tensor,
)


def get_obs_shapes(obs):
    return {k: v.shape for k, v in obs.items()}


def dict_to_device(d, device, non_blocking=False):
    return {k: v.to(device, non_blocking=non_blocking) for k, v in d.items()}


def dict_to_tensor(d, device="cpu", non_blocking=False, indices=slice(0, None)):
    return {
        k: torch.tensor(v[indices]).float().to(device, non_blocking=non_blocking)
        for k, v in d.items()
    }


def dict_to_tensor_unsqueeze(
    d, device="cpu", non_blocking=False, indices=slice(0, None)
):
    return {
        k: torch.tensor(v[indices])
        .float()
        .unsqueeze(0)
        .to(device, non_blocking=non_blocking)
        for k, v in d.items()
    }


def get_2d_combinations(indices: List[int]):
    """Given a list of indices, calculate the lower triangular part of the cartesian product.
    Appropriate for retrieving all 2d combinations of indices (up to permutation).

    Args:
        indices: a list of indices
    """
    output = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            output.append([indices[i], indices[j]])
    return output


def combine_z(z: Tensor, combinations: Optional[List]) -> Tensor:
    """Generate parameter combinations in last dimension using fancy indexing.

    Args:
        z: parameters of shape [..., Z]
        combinations: list of parameter combinations.

    Returns:
        output = z[..., combinations]
    """
    return z[..., combinations]


def set_device(gpu: bool = False) -> torch.device:
    """Select device, defaults to cpu."""
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        # torch.set_default_tensor_type("torch.cuda.FloatTensor")
    elif gpu and not torch.cuda.is_available():
        warn("Although the gpu flag was true, the gpu is not avaliable.")
        device = torch.device("cpu")
        # torch.set_default_tensor_type("torch.FloatTensor")
    else:
        device = torch.device("cpu")
        # torch.set_default_tensor_type("torch.FloatTensor")
    return device


def get_x(list_xz):
    """Extract x from batch of samples."""
    return [xz["x"] for xz in list_xz]


def get_z(list_xz):
    """Extract z from batch of samples."""
    return [xz["z"] for xz in list_xz]


def get_device_if_not_none(device: Optional[Device], tensor: Tensor) -> Device:
    """Returns device if not None, else returns tensor.device."""
    return tensor.device if device is None else device


np_bool_types = [np.bool]
np_int_types = [np.int8, np.int16, np.int32, np.int64]
np_float_types = [np.float32, np.float64]
torch_bool_types = [torch.bool]
torch_int_types = [torch.int8, torch.int16, torch.int32, torch.int64]
torch_float_types = [torch.float32, torch.float64]


def array_to_tensor(
    array: Array, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
) -> Tensor:
    """Converts np.ndarray and torch.Tensor to torch.Tensor with dtype and on device.
    When dtype is None, unsafe casts all float-type arrays to torch.float32 and all int-type arrays to torch.int64
    """
    input_dtype = array.dtype
    if isinstance(input_dtype, np.dtype):
        if dtype is None:
            if input_dtype in np_float_types:
                dtype = torch.float32
            elif input_dtype in np_int_types:
                dtype = torch.int64
            elif input_dtype in np_bool_types:
                dtype = torch.bool
            else:
                raise TypeError(
                    f"{input_dtype} was not a supported numpy int, float, or bool."
                )
        return torch.from_numpy(array).to(dtype=dtype, device=device)
    elif isinstance(input_dtype, torch.dtype):
        if dtype is None:
            if input_dtype in torch_float_types:
                dtype = torch.float32
            elif input_dtype in torch_int_types:
                dtype = torch.int64
            elif input_dtype in torch_bool_types:
                dtype = torch.bool
            else:
                raise TypeError(
                    f"{input_dtype} was not a supported torch int, float, or bool."
                )
        return array.to(dtype=dtype, device=device)
    else:
        raise TypeError(
            f"{input_dtype} was not recognized as a supported numpy.dtype or torch.dtype."
        )


def tobytes(x: Array):
    if isinstance(x, np.ndarray):
        return x.tobytes()
    elif isinstance(x, Tensor):
        return x.numpy().tobytes()
    else:
        raise TypeError(f"{type(x)} does not support tobytes.")


def depth(seq: Sequence):
    if seq and isinstance(seq, str):
        return 0
    elif seq and isinstance(seq, Sequence):
        return 1 + max(depth(item) for item in seq)
    else:
        return 0


def process_combinations(comb: Combinations):
    d = depth(comb)
    if d == 0:
        return [[comb]]
    elif d == 1:
        return [[i] for i in comb]
    elif d == 2:
        return comb
    else:
        raise ValueError(f"{comb} is not understood to be of type Combinations.")


def is_empty(directory: PathType):
    directory = Path(directory)
    if next(directory.iterdir(), None) is None:
        return True
    else:
        return False


def get_stats(z, p):
    # Returns central credible intervals
    zmax = z[p.argmax()]
    c = scipy.integrate.cumtrapz(p, z, initial=0)
    res = np.interp([0.025, 0.16, 0.5, 0.84, 0.975], c, z)
    xmedian = res[2]
    xerr68 = [res[1], res[3]]
    xerr95 = [res[0], res[4]]
    return {
        "mode": zmax,
        "median": xmedian,
        "cred68": xerr68,
        "cred95": xerr95,
        "err68": (xerr68[1] - xerr68[0]) / 2,
        "err95": (xerr95[1] - xerr95[0]) / 2,
    }


def cred1d(re, x0: Array):
    """Calculate credible regions.

    Args:
        re (RatioEstimator): ratio estimators
        x0: true observation
    """
    zdim = re.zdim
    for i in range(zdim):
        z, p = re.posterior(x0, i)
        res = get_stats(z, p)
        print("z%i = %.5f +- %.5f" % (i, res["median"], res["err68"]))


class Module(nn.Module):
    """Thin wrapper around pytorch modules that enables automatic reloading of network classes with correct arguments."""

    registry = {}

    def __init__(self, *args, **kwargs):
        """Store arguments of subclass instantiation."""
        self._swyft_args = [args, kwargs]
        super().__init__()

    def __init_subclass__(cls, **kwargs):
        """Register subclasses."""
        super().__init_subclass__(**kwargs)
        cls.registry[cls.__name__] = cls
        cls._swyft_tag = cls.__name__

    @property
    def swyft_args(self):
        return self._swyft_args

    @property
    def swyft_tag(self):
        """Return subclass tag."""
        # TODO: Confirm this is stable. Alternative is to as users to provide tags.
        return self._swyft_tag

    def swyft_state_dict(self):
        torch_state_dict = self.state_dict()
        return dict(
            torch_state_dict=torch_state_dict,
            swyft_args=self.swyft_args,
            swyft_tag=self.swyft_tag,
        )

    @classmethod
    def from_swyft_state_dict(cls, state_dict):
        subcls = cls.registry[state_dict["swyft_tag"]]
        args, kwargs = state_dict["swyft_args"]
        instance = subcls(*args, **kwargs)
        instance.load_state_dict(state_dict["torch_state_dict"])
        return instance


def _corner_params(params):
    out = []
    for i in range(len(params)):
        for j in range(i, len(params)):
            if i == j:
                out.append((params[i],))
            else:
                out.append((params[i], params[j]))
    return out


def sort_param_list(param_list: Sequence):
    result = []
    for v in param_list:
        if not isinstance(v, tuple):
            v = (v,)
        else:
            v = tuple(sorted(v))
        result.append(v)
    return result


def format_param_list(params, all_params=None, mode="custom"):
    # Use all parameters if params == None
    if params is None and all_params is None:
        raise ValueError("Specify parameters!")
    if params is None:
        params = all_params

    if mode == "custom" or mode == "1d":
        param_list = params
    elif mode == "2d":
        param_list = _corner_params(params)
    else:
        raise KeyError("Invalid mode argument.")

    return sort_param_list(param_list)


def all_finite(x):
    if isinstance(x, dict):
        return all(_all_finite(v) for v in x.values())
    elif isinstance(x, (torch.Tensor, np.ndarray)):
        return _all_finite(x)
    elif isinstance(x, list):
        return all(_all_finite(v) for v in x)
    else:
        raise NotImplementedError("That type is not yet implemented.")


def _all_finite(x: Array):
    if isinstance(x, torch.Tensor):
        return torch.all(torch.isfinite(x))
    else:
        return np.all(np.isfinite(x))


def swyftify_params(params: Array, parameter_names: List[str]) -> Dict[str, Array]:
    """Translates a [..., dim] tensor into a dictionary with dim keys.

    Args:
        params (Array):

    Returns:
        swyft_parameters (dict): 
    """
    return {k: params[..., i] for i, k in enumerate(parameter_names)}


def unswyftify_params(
    swyft_params: Dict[str, Array], parameter_names: List[str]
) -> Array:
    """Translates a dictionary with dim keys into a tensor with [..., dim] shape.

    Args:
        swyft_params (Dict[str, Array]): dictionary with parameter_names as keys, tensors as values
        parameter_names (List[str]): 

    Returns:
        Array: stacked params
    """
    if isinstance(swyft_params[parameter_names[0]], torch.Tensor):
        return torch.stack([swyft_params[name] for name in parameter_names], dim=-1)
    # elif isinstance(swyft_params[parameter_names[0]], np.ndarray):
    else:
        return np.stack([swyft_params[name] for name in parameter_names], axis=-1)


def swyftify_observation(observation: torch.Tensor):
    assert observation.ndim == 1, f"ndim was {observation.ndim}, but should be 1."
    return dict(x=observation)


def unswyftify_observation(swyft_observation: dict):
    return swyft_observation["x"]

def grid_interpolate_samples(x, y, bins = 1000, return_norm = False):
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    x_grid = np.linspace(x[0], x[-1], bins)
    y_grid = np.interp(x_grid, x, y)
    norm = simps(y_grid, x_grid)
    y_grid_normed = y_grid/norm
    if return_norm:
        return x_grid, y_grid_normed, norm
    else:
        return x_grid, y_grid_normed

def get_entropy(x, y, y_true = None, bins = 1000):
    """Estimate 1-dim entropy, norm and KL divergence.
    
    Args:
        x (Array): x-values
        y (Array): probability density y = p(x)
        y_true (function): functional form of the true probability density for KL calculation
        bins (int): Number of bins to use for interpolation.
    """
    x_int, y_int, norm = grid_interpolate_samples(x, y, bins = bins, return_norm = True)
    entropy = simps(y_int*np.log(y_int), x_int)
    if y_true is not None:
        y_int_true = y_true(x_int)
        KL = simps(y_int*np.log(y_int/y_int_true), x_int)
        return dict(entropy = entropy, norm = norm, KL = KL)
    return dict(entropy = entropy, norm = norm)

def sample_diagnostics(samples, true_posteriors = {}, true_params = {}):
    result = {}
    for params in samples['weights'].keys():
        if len(params) > 1:
            continue
        else:  # 1-dim case
            x = samples['params'][params[0]]
            y = samples['weights'][params]
            if params in true_posteriors.keys():
                y_true = true_posteriors[params]
            else:
                y_true = None

            result[params] = get_entropy(x, y, y_true = y_true)
    return result


if __name__ == "__main__":
    pass

