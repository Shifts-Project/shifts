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

"""
Defines a behavioral cloning model with autoregressive RNN decoder as
used in Conditional Imitation Learning (CIL) [Codevilla et al., 2017].
"""

from typing import Mapping
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from sdc.oatomobile.torch.networks.perception import MobileNetV2


class BehaviouralModel(nn.Module):
    """A `PyTorch` implementation of a behavioural cloning model."""

    def __init__(
        self,
        in_channels: int = 16,
        output_shape: Tuple[int, int] = (25, 2),
    ) -> None:
        """Constructs a simple behavioural cloning model.

        Args:
          output_shape: The shape of the base and
            data distribution (a.k.a. event_shape).
        """
        super(BehaviouralModel, self).__init__()
        self._output_shape = output_shape

        # The convolutional encoder model.
        self._encoder = MobileNetV2(num_classes=64, in_channels=in_channels)

        # No need for an MLP merger, as all inputs (including static HD map
        # features) have been converted to an image representation.

        # The decoder recurrent network used for the sequence generation.
        self._decoder = nn.GRUCell(input_size=2, hidden_size=64)

        # The output head.
        self._output = nn.Linear(
            in_features=64,
            out_features=self._output_shape[-1],
        )

    def forward(self, **context: torch.Tensor) -> torch.Tensor:
        """Returns the expert plan."""

        # Parses context variables.
        feature_maps = context.get("feature_maps")

        # Encodes the visual input.
        z = self._encoder(feature_maps)

        # z is the decoder's initial state.

        # Output container.
        y = list()

        # Initial input variable.
        x = torch.zeros(  # pylint: disable=no-member
            size=(z.shape[0], self._output_shape[-1]),
            dtype=z.dtype,
        ).to(z.device)

        # Autoregressive generation of plan.
        for _ in range(self._output_shape[0]):
            # Unrolls the GRU.
            z = self._decoder(x, z)

            # Predicts the displacement (residual).
            dx = self._output(z)
            x = dx + x

            # Updates containers.
            y.append(x)

        return torch.stack(y, dim=1)  # pylint: disable=no-member


def train_step_bc(
  model: BehaviouralModel,
  optimizer: optim.Optimizer,
  criterion: torch.nn.Module,
  batch: Mapping[str, torch.Tensor],
  clip: bool = False,
) -> torch.Tensor:
    """Performs a single gradient-descent optimisation step."""
    # Resets optimizer's gradients.
    optimizer.zero_grad()

    # Forward pass from the model.
    predictions = model(**batch)

    # Calculates loss.
    loss = criterion(predictions, batch["ground_truth_trajectory"])
    loss = torch.sum(loss, dim=[-2, -1])  # pylint: disable=no-member
    loss = torch.mean(loss, dim=0)  # pylint: disable=no-member

    # Backward pass.
    loss.backward()

    # Clips gradients norm.
    if clip:
          torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

    # Performs a gradient descent step.
    optimizer.step()
    return loss


def evaluate_step_bc(
    criterion: torch.nn.Module.Loss,
    model: BehaviouralModel,
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Evaluates `model` on a `batch`."""
    # Forward pass from the model.
    predictions = model(**batch)

    # Calculates loss on mini-batch.
    loss = criterion(predictions, batch["ground_truth_trajectory"])
    loss = torch.sum(loss, dim=[-2, -1])  # pylint: disable=no-member
    loss = torch.mean(loss, dim=0)  # pylint: disable=no-member

    return loss
