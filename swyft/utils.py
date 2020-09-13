# pylint: disable=no-member
from warnings import warn
from typing import Optional

import numpy as np
import torch

from .types import Device, Tensor, Array


def set_device(gpu: bool = False) -> Device:
    """Sets the default tensor device."""
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    elif gpu and not torch.cuda.is_available():
        warn("Although the gpu flag was true, the gpu is not avaliable.")
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")
    return device


def get_device_if_not_none(device: Optional[Device], tensor: Tensor) -> Device:
    """Returns device if not None, else returns tensor.device."""
    return tensor.device if device is None else device


def array_to_tensor(array: Array, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None) -> Tensor:
    """Converts np.ndarray and torch.Tensor to torch.Tensor with dtype and on device. 
    When dtype is None, unsafe casts all float-type arrays to torch.get_default_dtype()
    """
    torch_float_types = [torch.half, torch.bfloat16, torch.float, torch.double]
    
    input_dtype = array.dtype
    if isinstance(input_dtype, np.dtype):
        # When the ndarray type is any float send to the torch default (when torch default is a float).
        if dtype is None and input_dtype == np.float and torch.get_default_dtype() in torch_float_types:
            dtype = torch.get_default_dtype()
        return torch.from_numpy(array).to(dtype=dtype, device=device)
    elif isinstance(input_dtype, torch.dtype):
        if dtype is None and dtype in torch_float_types and torch.get_default_dtype() in torch_float_types:
            dtype = torch.get_default_dtype()
        return array.to(dtype=dtype, device=device)
    else:
        raise TypeError(f"{input_dtype} was neither numpy.dtype or torch.dtype.")


if __name__ == "__main__":
    pass
