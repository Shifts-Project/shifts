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
"""Generic implementation of multi-layer perceptron."""

from typing import Callable
from typing import Optional
from typing import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
  """A simple multi-layer perceptron module."""

  def __init__(
      self,
      input_size: int,
      output_sizes: Sequence[int],
      activation_fn: Callable[[], nn.Module] = nn.ReLU,
      dropout_rate: Optional[float] = None,
      activate_final: bool = False,
  ) -> None:
    """Constructs a simple multi-layer-perceptron.

    Args:
      input_size: The size of the input features.
      output_sizes: Sequence of layer sizes.
      w_init: Initializer for Linear weights.
      activation: Activation function to apply between linear layers. Defaults
        to ReLU.
      dropout_rate: Dropout rate to apply, a rate of `None` (the default) or `0`
        means no dropout will be applied.
      activate_final: Whether or not to activate the final layer of the MLP.
    """
    super(MLP, self).__init__()

    layers = list()
    for in_features, out_features in zip(
        [input_size] + list(output_sizes)[:-2],
        output_sizes[:-1],
    ):
      # Fully connected layer.
      layers.append(nn.Linear(in_features, out_features))
      # Activation layer.
      layers.append(activation_fn(inplace=True))
      # (Optional) dropout layer.
      if dropout_rate is not None:
        layers.append(nn.Dropout(p=dropout_rate, inplace=True))
    # Final layer.
    layers.append(nn.Linear(output_sizes[-2], output_sizes[-1]))
    # (Optional) output activation layer.
    if activate_final:
      layers.append(activation_fn(inplace=True))

    self._model = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass from the MLP."""
    return self._model(x)
