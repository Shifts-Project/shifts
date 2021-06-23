# Copyright 2020 The OATomobile Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for nested data structures involving NumPy and PyTorch."""

import torch
import numpy as np


def safe_torch_to_float(val):
    if isinstance(val, float):
        return val
    elif isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy().item(0)
    else:
        raise ValueError(
            f'val is neither or type `float` nor `torch.Tensor`: {type(val)}')


def safe_torch_to_numpy(val):
    if isinstance(val, np.ndarray):
        return val
    elif isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy()
    else:
        raise ValueError(
            f'val is neither or type `np.ndarray` nor `torch.Tensor`: '
            f'{type(val)}')
