# pylint: disable=no-member

# Inspired by https://github.com/mackelab/sbi/blob/main/sbi/types.py
# That file is using Affero General Public License v3, see <https://www.gnu.org/licenses/>

import numpy as np
import torch
from typing import Union, Tuple

Tensor = torch.Tensor
Device = Union[torch.device, str]

Array = Union[np.ndarray, torch.Array]
Shape = Union[torch.Size, Tuple[int, ...]]
ScalarFloat = Union[torch.Array, float]
