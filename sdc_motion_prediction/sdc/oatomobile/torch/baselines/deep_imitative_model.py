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
"""Defines a Deep Imitative Model with autoregressive flow decoder."""

from typing import Mapping, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim

from sdc.metrics import SDCLoss
from sdc.oatomobile.torch.networks.perception import MobileNetV2
from sdc.oatomobile.torch.networks.sequence import AutoregressiveFlow


class ImitativeModel(nn.Module):
    """A `PyTorch` implementation of an imitative model."""

    def __init__(
        self,
        num_decoding_steps: int,
        decoding_lr: float,
        in_channels: int,
        dim_hidden: int = 128,
        output_shape: Tuple[int, int] = (25, 2),
        **kwargs
    ) -> None:
        """Constructs a simple imitative model.

        Args:
          output_shape: The shape of the base and
            data distribution (a.k.a. event_shape).
          num_decoding_steps: int, number of grad descent steps
            for finding the mode.
          decoding_lr: float, learning rate for finding the mode.
        """
        super(ImitativeModel, self).__init__()
        self._output_shape = output_shape

        # The convolutional encoder model.
        self._encoder = MobileNetV2(
            num_classes=dim_hidden, in_channels=in_channels)

        # No need for an MLP merger, as all inputs (including static HD map
        # features) have been converted to an image representation.

        # The decoder recurrent network used for the sequence generation.
        self._decoder = AutoregressiveFlow(
            output_shape=self._output_shape,
            hidden_size=dim_hidden,
        )

        # Forward pass decoding args
        self._num_decoding_steps = num_decoding_steps
        self._decoding_lr = decoding_lr

    def to(self, *args, **kwargs):
        """Handles non-parameter tensors when moved to a new device."""
        self = super().to(*args, **kwargs)
        self._decoder = self._decoder.to(*args, **kwargs)
        return self

    def forward(
        self,
        **context: torch.Tensor
    ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        """Returns a local mode from the posterior.

        Args:
          context: (keyword arguments) The conditioning
            variables used for the conditional flow.

        Returns:
          A mode from the posterior, with shape `[D, 2]`.
        """
        batch_size = context["feature_maps"].shape[0]

        # Sets initial sample to base distribution's mean.
        x = self._decoder._base_dist.sample().clone().detach().repeat(
            batch_size, 1).view(
            batch_size,
            *self._output_shape,
        )
        x.requires_grad = True

        # The contextual parameters, caches for efficiency.
        z = self._params(**context)

        # Initialises a gradient-based optimiser.
        optimizer = optim.Adam(params=[x], lr=self._decoding_lr)

        # Stores the best values.
        x_best = x.clone()
        loss_best = torch.ones(()).to(
            x.device) * 1000.0  # pylint: disable=no-member

        for _ in range(self._num_decoding_steps):
            # Resets optimizer's gradients.
            optimizer.zero_grad()

            # Operate on `y`-space.
            y, _ = self._decoder._forward(x=x, z=z)

            # Calculates imitation prior.
            _, log_prob, logabsdet = self._decoder._inverse(y=y, z=z)
            imitation_prior = torch.mean(
                log_prob - logabsdet)  # pylint: disable=no-member
            loss = -imitation_prior

            # Backward pass.
            loss.backward(retain_graph=True)

            # Performs a gradient descent step.
            optimizer.step()

            # Book-keeping
            if loss < loss_best:
                x_best = x.clone()
                loss_best = loss.clone()

        y, _ = self._decoder._forward(x=x_best, z=z)

        return y

    def _params(self, **context: torch.Tensor) -> torch.Tensor:
        """Returns the contextual parameters of the conditional density estimator.

        Args:
          feature_maps: Feature maps, with shape `[B, H, W, C]`.

        Returns:
          The contextual parameters of the conditional density estimator.
        """

        # Parses context variables.
        feature_maps = context.get("feature_maps")

        # Encodes the image-format input.
        return self._encoder(feature_maps)


def train_step_dim(
    model: ImitativeModel,
    optimizer: optim.Optimizer,
    batch: Mapping[str, torch.Tensor],
    noise_level: float,
    clip: bool = False,
) -> torch.Tensor:
    """Performs a single gradient-descent optimisation step."""
    # Resets optimizer's gradients.
    optimizer.zero_grad()

    # Perturb target.
    y = torch.normal(  # pylint: disable=no-member
        mean=batch["ground_truth_trajectory"],
        std=torch.ones_like(batch["ground_truth_trajectory"]) * noise_level,
        # pylint: disable=no-member
    )

    # Forward pass from the model.
    z = model._params(**batch)
    _, log_prob, logabsdet = model._decoder._inverse(y=y, z=z)

    # Calculates loss (NLL).
    loss = -torch.mean(log_prob - logabsdet,
                       dim=0)  # pylint: disable=no-member

    # Backward pass.
    loss.backward()

    # Clips gradients norm.
    if clip:
        torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

    # Performs a gradient descent step.
    optimizer.step()

    return loss


def evaluate_step_dim(
    sdc_loss: SDCLoss,
    model: ImitativeModel,
    batch: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    """Evaluates `model` on a `batch`."""
    # Forward pass from the model.
    z = model._params(**batch)
    _, log_prob, logabsdet = model._decoder._inverse(
        y=batch["ground_truth_trajectory"], z=z)

    # Calculates NLL.
    nll = -torch.mean(log_prob - logabsdet, dim=0)

    # Decode a trajectory from the posterior.
    predictions = model.forward(**batch)
    ade = sdc_loss.average_displacement_error(
        predictions=predictions,
        ground_truth=batch["ground_truth_trajectory"])
    fde = sdc_loss.final_displacement_error(
        predictions=predictions,
        ground_truth=batch["ground_truth_trajectory"])
    loss_dict = {
        'nll': nll,
        'ade': ade,
        'fde': fde}
    return loss_dict
