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
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sdc.metrics import SDCLoss
from sdc.oatomobile.torch.networks.perception import MobileNetV2
from ysdc_dataset_api.evaluation.metrics import (
    average_displacement_error_torch, final_displacement_error_torch,
    batch_mean_metric_torch)

class BehaviouralModel(nn.Module):
    """A `PyTorch` implementation of a behavioural cloning model."""

    def __init__(
        self,
        in_channels: int,
        dim_hidden: int = 128,
        output_shape: Tuple[int, int] = (25, 2),
        scale_eps: float = 1e-7,
        bc_deterministic: bool = False,
        generation_mode: str = 'sampling',
        **kwargs
    ) -> None:
        """Constructs a behavioural cloning model.

        Args:
            in_channels: Number of channels in image-featurized context
            dim_hidden: Hidden layer size of encoder output / GRU
            output_shape: The shape of the data distribution
                (a.k.a. event_shape).
            scale_eps: Epsilon term to avoid numerical instability by
                predicting zero Gaussian scale.
            generation_mode: one of {sampling, teacher-forcing}.
                In the former case, the autoregressive likelihood is formed
                    by conditioning on samples.
                In the latter case, the likelihood is formed by conditioning
                    on ground-truth.
        """
        super(BehaviouralModel, self).__init__()
        assert generation_mode in {'sampling', 'teacher-forcing'}

        self._output_shape = output_shape

        # The convolutional encoder model.
        self._encoder = MobileNetV2(
            in_channels=in_channels, num_classes=dim_hidden)

        # All inputs (including static HD map features)
        # have been converted to an image representation;
        # No need for an MLP merger.

        # The decoder recurrent network used for the sequence generation.
        self._decoder = nn.GRUCell(
            input_size=self._output_shape[-1], hidden_size=dim_hidden)

        self._scale_eps = scale_eps

        if bc_deterministic:
            print('DEBUG: using deterministic training and eval for BC model.')
            # The output head, predicts displacement.
            self._output = nn.Linear(
                in_features=dim_hidden,
                out_features=(self._output_shape[-1]))
        else:
            print('BC Model: using Gaussian likelihood.')
            # The output head, predicts the parameters of a Gaussian.
            self._output = nn.Linear(
                in_features=dim_hidden,
                out_features=(self._output_shape[-1] * 2))

        self.bc_deterministic = bc_deterministic
        self._generation_mode = generation_mode
        print(f'BC Model: using generation mode {generation_mode}.')

    def forward_deterministic(self, **context: torch.Tensor) -> torch.Tensor:
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

    def decode(
        self,
        z: torch.Tensor,
    ):
        """Returns the expert plan."""
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
            scale = F.softplus(dloc_scale[..., 2:]) + self._scale_eps

            # Data distribution corresponding sample from a std normal.
            y_t = (y_tm1 + dloc) + (
                scale * torch.normal(
                    mean=torch.zeros((z.shape[0], self._output_shape[-1])),
                    std=torch.ones((z.shape[0], self._output_shape[-1]))).to(
                    device=z.device))

            # Update containers.
            y.append(y_t)
            scales.append(scale)
            y_tm1 = y_t

        # Prepare tensors, reshape to [B, T, 2].
        y = torch.stack(y, dim=-2)  # pylint: disable=no-member
        scales = torch.stack(scales, dim=-2)  # pylint: disable=no-member
        return y, scales

    def forward(
        self,
        **context: torch.Tensor
    ) -> torch.Tensor:
        """Sample a local mode from the posterior.

        Args:
          context: (keyword arguments) The conditioning
            variables used for the conditional sequence generation.

        Returns:
          A batch of trajectories with shape `[B, T, 2]`.
        """
        # The contextual parameters.
        feature_maps = context.get("feature_maps")

        # Encodes the visual input.
        # Cache embedding, because we may use it in scoring plans
        # from other ensemble members in RIP.
        self._z = self._encoder(feature_maps)

        # Decode a trajectory.
        y, _ = self.decode(z=self._z)
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
            y: modes from the posterior of a BC model, with shape `[B, T, 2]`.
        Returns:
            log-likelihood for each plan in the batch, i.e., shape `[B]`.
        """
        return self.log_likelihood(y=y, z=self._z)

    def log_likelihood(
        self,
        y: torch.Tensor,
        z: torch.Tensor
    ):
        """Obtain log likelihood of given sample from data distribution
            (ground-truth or generated plan).

          Args:
            y: Samples from the data distribution, with shape `[B, D]`.
            z: The contextual parameters obtained from the CNN-based encoder,
                with shape `[B, K]`.

          Returns:
            log_prob: The log-likelihood of the samples, with shape `[B]`.
          """
        # Keep track of predicted (delta) locations and scales,
        # which will parameterize a MVN distribution.
        dlocs = list()
        scales = list()

        # Initial input variable.
        y_tm1 = torch.zeros(  # pylint: disable=no-member
            size=(y.shape[0], self._output_shape[-1]),
            dtype=y.dtype,
        ).to(y.device)

        for t in range(y.shape[-2]):
            # Unrolls the GRU.
            z = self._decoder(y_tm1, z)

            # Predicts the location and scale of the MVN distribution.
            dloc_scale = self._output(z)
            dloc = dloc_scale[..., :2]
            scale = F.softplus(dloc_scale[..., 2:]) + self._scale_eps

            # Update containers.
            dlocs.append(dloc)
            scales.append(scale)

            # Condition on ground truth
            if self._generation_mode == 'teacher-forcing':
                y_t = y[:, t, :]
            # Condition on sample
            elif self._generation_mode == 'sampling':
                y_t = (y_tm1 + dloc) + (
                    scale * torch.normal(
                        mean=torch.zeros((z.shape[0], self._output_shape[-1])),
                        std=torch.ones((z.shape[0], self._output_shape[-1]))
                    ).to(device=y.device))
            else:
                raise NotImplementedError

            y_tm1 = y_t

        scales = torch.stack(scales, dim=1).reshape(y.shape[0], -1)

        # Pred likelihood for each timestep
        likelihood = D.MultivariateNormal(
            loc=torch.stack(dlocs, dim=1).reshape(y.shape[0], -1),
            scale_tril=torch.stack([
                torch.diag(scales[batch_index, :])
                for batch_index in range(scales.shape[0])], dim=0))

        # Convert ground-truth to displacements at each time t
        y_copy = y.clone().detach()
        y_copy[:, 1:] = y_copy[:, 1:] - y_copy[:, :-1]
        y_copy = y_copy.reshape(y_copy.shape[0], -1)

        return likelihood.log_prob(y_copy)


def train_step_bc(
    model: BehaviouralModel,
    optimizer: optim.Optimizer,
    batch: Mapping[str, torch.Tensor],
    clip: bool = False,
) -> Mapping[str, torch.Tensor]:
    """Performs a single gradient-descent optimization step."""
    # Resets optimizer's gradients.
    optimizer.zero_grad()

    if model.bc_deterministic:
        predictions = model.forward_deterministic(**batch)

        # Compute ADE loss
        y = batch["ground_truth_trajectory"]
        ade = batch_mean_metric_torch(
            base_metric=average_displacement_error_torch,
            predictions=predictions,
            ground_truth=y)

        # Backward pass.
        ade.backward()

        # Clips gradients norm.
        if clip:
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

        # Performs a gradient descent step.
        optimizer.step()

        # Calculates other losses.
        fde = batch_mean_metric_torch(
            base_metric=final_displacement_error_torch,
            predictions=predictions,
            ground_truth=y)

        loss_dict = {
            'ade': ade.detach(),
            'fde': fde.detach()}
    else:
        # Model forward pass. Caches scene context in self._z.
        predictions = model.forward(**batch)

        # Compute NLL.
        y = batch["ground_truth_trajectory"]
        log_likelihood = model.score_plans(y=y)
        nll = -torch.mean(log_likelihood)

        # Backward pass.
        nll.backward()

        # Clips gradients norm.
        if clip:
            torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

        # Performs a gradient descent step.
        optimizer.step()

        # Calculates other losses.
        ade = batch_mean_metric_torch(
            base_metric=average_displacement_error_torch,
            predictions=predictions,
            ground_truth=y)
        fde = batch_mean_metric_torch(
            base_metric=final_displacement_error_torch,
            predictions=predictions,
            ground_truth=y)
        loss_dict = {
            'nll': nll.detach(),
            'ade': ade.detach(),
            'fde': fde.detach()}

    return loss_dict


def evaluate_step_bc(
    model: BehaviouralModel,
    batch: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    """Evaluates `model` on a `batch`."""
    if model.bc_deterministic:
        predictions = model.forward_deterministic(**batch)

        # Compute losses.
        y = batch["ground_truth_trajectory"]

        # Calculates other losses.
        ade = batch_mean_metric_torch(
            base_metric=average_displacement_error_torch,
            predictions=predictions,
            ground_truth=y)
        fde = batch_mean_metric_torch(
            base_metric=final_displacement_error_torch,
            predictions=predictions,
            ground_truth=y)
        loss_dict = {
            'ade': ade.detach(),
            'fde': fde.detach()}
    else:
        # Model forward pass. Caches scene context in self._z.
        predictions = model.forward(**batch)

        # Compute NLL.
        y = batch["ground_truth_trajectory"]
        log_likelihood = model.score_plans(y=y)
        nll = -torch.mean(log_likelihood)

        # Calculates other losses.
        ade = batch_mean_metric_torch(
            base_metric=average_displacement_error_torch,
            predictions=predictions,
            ground_truth=y)
        fde = batch_mean_metric_torch(
            base_metric=final_displacement_error_torch,
            predictions=predictions,
            ground_truth=y)
        loss_dict = {
            'nll': nll.detach(),
            'ade': ade.detach(),
            'fde': fde.detach()}

    return loss_dict
