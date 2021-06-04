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
Defines a behavioral cloning model with autoregressive RNN decoder,
based on Conditional Imitation Learning (CIL) [Codevilla et al., 2017].
"""

from typing import Mapping
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from sdc.metrics import SDCLoss

from sdc.oatomobile.torch.networks.perception import MobileNetV2


class BehaviouralModel(nn.Module):
    """A `PyTorch` implementation of a behavioural cloning model."""

    def __init__(
        self,
        in_channels: int,
        dim_hidden: int = 128,
        output_shape: Tuple[int, int] = (25, 2),
        device: str = 'cpu',
        **kwargs
    ) -> None:
        """Constructs a simple behavioural cloning model.

        Args:
          output_shape: The shape of the base and
            data distribution (a.k.a. event_shape).
        """
        super(BehaviouralModel, self).__init__()
        self._output_shape = output_shape

        # The convolutional encoder model.
        self._encoder = MobileNetV2(
            in_channels=in_channels, num_classes=dim_hidden)

        # No need for an MLP merger, as all inputs (including static HD map
        # features) have been converted to an image representation.

        # The decoder recurrent network used for the sequence generation.
        self._decoder = nn.GRUCell(
            input_size=self._output_shape[-1], hidden_size=dim_hidden)

        # The output head, predicts the parameters of a Gaussian.
        self._output = nn.Linear(
            in_features=dim_hidden,
            out_features=(self._output_shape[-1] * 2))

        self._device = device

    def forward_deterministic(self, **context: torch.Tensor) -> torch.Tensor:
        """Returns the expert plan."""
        raise NotImplementedError('Deprecated, now sampling from a Gaussian.')

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

    def forward(self,
                **context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the expert plan."""
        # Parses context variables.
        feature_maps = context.get("feature_maps")

        # Encodes the visual input.
        z = self._encoder(feature_maps)

        # z is the decoder's initial state.

        # Output container.
        y = list()
        scales = list()

        # Initial input variable.
        y_tm1 = torch.zeros(  # pylint: disable=no-member
            size=(z.shape[0], self._output_shape[-1]),
            dtype=z.dtype,
        ).to(z.device)

        # Autoregressive generation of plan.
        for _ in range(self._output_shape[0]):
            # Unrolls the GRU.
            z = self._decoder(y_tm1, z)

            # Predicts the location and scale of the MVN distribution.
            dloc_scale = self._output(z)
            dloc = dloc_scale[..., :2]
            scale = F.softplus(dloc_scale[..., 2:]) + 1e-3

            # Data distribution corresponding sample from a std normal.
            y_t = (y_tm1 + dloc) + (
                scale * torch.normal(
                mean=torch.zeros((z.shape[0], self._output_shape[-1])),
                std=torch.ones((z.shape[0], self._output_shape[-1])),
                device=self._device))

            # Update containers.
            y.append(y_t)
            scales.append(scale)
            y_tm1 = y_t

        # Prepare tensors, reshape to [B, T, 2].
        y = torch.stack(y, dim=-2)  # pylint: disable=no-member
        scales = torch.stack(scales, dim=-2)  # pylint: disable=no-member
        return y, scales


def train_step_bc(
    sdc_loss: SDCLoss,
    model: BehaviouralModel,
    optimizer: optim.Optimizer,
    batch: Mapping[str, torch.Tensor],
    clip: bool = False,
) -> torch.Tensor:
    """Performs a single gradient-descent optimisation step."""
    # Resets optimizer's gradients.
    optimizer.zero_grad()

    # Forward pass from the model.
    predictions, _ = model(**batch)

    # Calculates loss.
    loss = sdc_loss.average_displacement_error(
        predictions=predictions,
        ground_truth=batch["ground_truth_trajectory"])

    # Backward pass.
    loss.backward()

    # Clips gradients norm.
    if clip:
        torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

    # Performs a gradient descent step.
    optimizer.step()
    return loss


def evaluate_step_bc(
    sdc_loss: SDCLoss,
    model: BehaviouralModel,
    batch: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    """Evaluates `model` on a `batch`."""
    # Forward pass from the model.
    predictions, _ = model(**batch)

    # Calculates loss on mini-batch.
    ade = sdc_loss.average_displacement_error(
        predictions=predictions,
        ground_truth=batch["ground_truth_trajectory"])
    fde = sdc_loss.final_displacement_error(
        predictions=predictions,
        ground_truth=batch["ground_truth_trajectory"])
    loss_dict = {
        'ade': ade,
        'fde': fde}
    return loss_dict
