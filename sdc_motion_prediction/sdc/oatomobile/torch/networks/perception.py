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
"""Perception (e.g., LIDAR) feature extractors."""

from typing import Callable
from typing import Optional
from typing import Sequence

import torch
import torch.nn as nn


class MobileNetV2(nn.Module):
  """A `PyTorch Hub` MobileNetV2 model wrapper."""

  def __init__(
      self,
      num_classes: int,
      in_channels: int = 3,
  ) -> None:
    """Constructs a MobileNetV2 model."""
    super(MobileNetV2, self).__init__()

    self._model = torch.hub.load(
      'pytorch/vision:v0.9.0', 'mobilenet_v2', num_classes=num_classes)

    # HACK(filangel): enables non-RGB visual features.
    _tmp = self._model.features._modules['0']._modules['0']
    self._model.features._modules['0']._modules['0'] = nn.Conv2d(
        in_channels=in_channels,
        out_channels=_tmp.out_channels,
        kernel_size=_tmp.kernel_size,
        stride=_tmp.stride,
        padding=_tmp.padding,
        bias=_tmp.bias,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass from the MobileNetV2."""
    return self._model(x)
