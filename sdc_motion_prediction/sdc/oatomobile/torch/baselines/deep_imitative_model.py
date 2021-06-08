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
        in_channels: int,
        dim_hidden: int = 128,
        output_shape: Tuple[int, int] = (25, 2),
        **kwargs
    ) -> None:
        """Constructs a simple imitative model.

        Args:
          output_shape: The shape of the base and
            data distribution (a.k.a. event_shape).
        """
        super(ImitativeModel, self).__init__()
        self._output_shape = output_shape

        # The convolutional encoder model.
        self._encoder = MobileNetV2(
            num_classes=dim_hidden, in_channels=in_channels)

        # No need for an MLP merger, as all inputs (including static HD map
        # features) have been converted to an image representation.

        # The autoregressive flow used for the sequence generation.
        self._flow = AutoregressiveFlow(
            output_shape=self._output_shape,
            hidden_size=dim_hidden,
        )

    def to(self, *args, **kwargs):
        """Handles non-parameter tensors when moved to a new device."""
        self = super().to(*args, **kwargs)
        self._flow = self._flow.to(*args, **kwargs)
        return self

    def forward(
        self,
        **context: torch.Tensor
    ) -> torch.Tensor:
        """Sample a local mode from the posterior.

        Args:
          context: (keyword arguments) The conditioning
            variables used for the conditional flow.

        Returns:
          A batch of trajectories with shape `[B, T, 2]`.
        """
        # batch_size = context["feature_maps"].shape[0]

        # Don't use this -- we want the latent sample to different for each
        # # Latent sample from the base distribution.
        # x = self._flow._base_dist.sample().clone().detach().repeat(
        #     batch_size, 1).view(
        #     batch_size,
        #     *self._output_shape,
        # )

        # The contextual parameters.
        # Cache them, because we may use them to score plans from
        # other DIM models in RIP.
        self._z = self._params(**context)

        # if True:
        #     a = 1

        # Operate on `y`-space.
        y = self._flow.forward(z=self._z)
        # if True:
        #     a = 1

        return y

    def score_plans(
        self,
        y: torch.Tensor
    ) -> torch.Tensor:
        """Scores plans given a context.
        NOTE: Context encoding is assumed to be stored in self._z,
            via execution of `forward`.
        Args:
            self._z: context encodings, shape `[B, K]`
            y: modes from the posterior of a DIM model, with shape `[B, T, 2]`.
        Returns:
            imitation_prior = log_prob - logabsdet
            for each plan in the batch, i.e., shape [B]
        """
        # Calculates imitation prior for each prediction in the batch.
        _, log_prob, logabsdet = self._flow._inverse(y=y, z=self._z)
        if True:
            a = 1
        imitation_priors = log_prob - logabsdet
        return imitation_priors

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
    sdc_loss: SDCLoss,
    model: ImitativeModel,
    optimizer: optim.Optimizer,
    batch: Mapping[str, torch.Tensor],
    noise_level: float,
    clip: bool = False,
    **kwargs
) -> Mapping[str, torch.Tensor]:
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
    _, log_prob, logabsdet = model._flow._inverse(y=y, z=z)

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

    # Decode a trajectory from the posterior on this train example.
    predictions = model.forward(**batch)
    ade = sdc_loss.batch_mean_metric(
        base_metric=sdc_loss.average_displacement_error,
        predictions=predictions,
        ground_truth=batch["ground_truth_trajectory"])
    fde = sdc_loss.batch_mean_metric(
        base_metric=sdc_loss.final_displacement_error,
        predictions=predictions,
        ground_truth=batch["ground_truth_trajectory"])
    loss_dict = {
        'nll': loss.detach(),
        'ade': ade.detach(),
        'fde': fde.detach()}

    return loss_dict


def evaluate_step_dim(
    sdc_loss: SDCLoss,
    model: ImitativeModel,
    batch: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    """Evaluates `model` on a `batch`."""
    # Forward pass from the model.
    # if True:
    #     a = 1

    z = model._params(**batch)
    _, log_prob, logabsdet = model._flow._inverse(
        y=batch["ground_truth_trajectory"], z=z)

    # Calculates NLL.
    nll = -torch.mean(log_prob - logabsdet, dim=0)

    # if True:
    #     a = 1

    # Decode a trajectory from the posterior.
    predictions = model.forward(**batch)

    # if True:
    #     a = 1

    ade = sdc_loss.batch_mean_metric(
        base_metric=sdc_loss.average_displacement_error,
        predictions=predictions,
        ground_truth=batch["ground_truth_trajectory"])
    fde = sdc_loss.batch_mean_metric(
        base_metric=sdc_loss.final_displacement_error,
        predictions=predictions,
        ground_truth=batch["ground_truth_trajectory"])
    loss_dict = {
        'nll': nll.detach(),
        'ade': ade.detach(),
        'fde': fde.detach()}
    return loss_dict
