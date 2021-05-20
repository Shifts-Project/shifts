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
"""Sequence generation."""

from typing import Tuple

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from sdc.oatomobile.torch import types
from sdc.oatomobile.torch.networks.mlp import MLP


class AutoregressiveFlow(nn.Module):
  """An autoregressive flow-based sequence generator."""

  def __init__(
      self,
      output_shape: types.Shape = (25, 2),
      hidden_size: int = 64,
  ):
    """Constructs a simple autoregressive flow-based sequence generator.

    Args:
      output_shape: The shape of the base and
        data distribution (a.k.a. event_shape).
      hidden_size: The dimensionality of the GRU hidden state.
    """
    super(AutoregressiveFlow, self).__init__()
    self._output_shape = output_shape

    # Initialises the base distribution.
    self._base_dist = D.MultivariateNormal(
        loc=torch.zeros(self._output_shape[-2] * self._output_shape[-1]),  # pylint: disable=no-member
        scale_tril=torch.eye(self._output_shape[-2] * self._output_shape[-1]),  # pylint: disable=no-member
    )

    # The decoder recurrent network used for the sequence generation.
    self._decoder = nn.GRUCell(
        input_size=self._output_shape[-1],
        hidden_size=hidden_size,
    )

    # The output head.
    self._locscale = MLP(
        input_size=hidden_size,
        output_sizes=[32, self._output_shape[-1] * 2],
        activation_fn=nn.ReLU,
        dropout_rate=None,
        activate_final=False,
    )

  def to(self, *args, **kwargs):
    """Handles non-parameter tensors when moved to a new device."""
    self = super().to(*args, **kwargs)
    self._base_dist = D.MultivariateNormal(
        loc=self._base_dist.mean.to(*args, **kwargs),
        scale_tril=self._base_dist.scale_tril.to(*args, **kwargs),
    )
    return self

  def forward(
      self,
      z: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward-pass, stochastic generation of a sequence.

    Args:
      z: The contextual parameters of the conditional density estimator, with
        shape `[B, K]`.

    Returns:
      The samples from the push-forward distribution, with shape `[B, D]`.
    """
    # Samples from the base distribution.
    x = self._base_dist.sample_n(n=z.shape[0])
    x = x.reshape(-1, *self._output_shape)

    return self._forward(x, z)[0]

  def _forward(
      self,
      x: torch.Tensor,
      z: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transforms samples from the base distribution to the data distribution.

    Args:
      x: Samples from the base distribution, with shape `[B, D]`.
      z: The contextual parameters of the conditional density estimator, with
        shape `[B, K]`.

    Returns:
      y: The sampels from the push-forward distribution,
        with shape `[B, D]`.
      logabsdet: The log absolute determinant of the Jacobian,
        with shape `[B]`.
    """

    # Output containers.
    y = list()
    scales = list()

    # Initial input variable.
    y_tm1 = torch.zeros(  # pylint: disable=no-member
        size=(z.shape[0], self._output_shape[-1]),
        dtype=z.dtype,
    ).to(z.device)

    for t in range(x.shape[-2]):
      x_t = x[:, t, :]

      # Unrolls the GRU.
      z = self._decoder(y_tm1, z)

      # Predicts the location and scale of the MVN distribution.
      dloc_scale = self._locscale(z)
      dloc = dloc_scale[..., :2]
      scale = F.softplus(dloc_scale[..., 2:]) + 1e-3

      # Data distribution corresponding sample.
      y_t = (y_tm1 + dloc) + scale * x_t

      # Update containers.
      y.append(y_t)
      scales.append(scale)
      y_tm1 = y_t

    # Prepare tensors, reshape to [B, T, 2].
    y = torch.stack(y, dim=-2)  # pylint: disable=no-member
    scales = torch.stack(scales, dim=-2)  # pylint: disable=no-member

    # Log absolute determinant of Jacobian.
    logabsdet = torch.log(torch.abs(torch.prod(scales, dim=-2)))  # pylint: disable=no-member
    logabsdet = torch.sum(logabsdet, dim=-1)  # pylint: disable=no-member

    return y, logabsdet

  def _inverse(
      self,
      y: torch.Tensor,
      z: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transforms samples from the data distribution to the base distribution.

    Args:
      y: Samples from the data distribution, with shape `[B, D]`.
      z: The contextual parameters of the conditional density estimator, with shape
        `[B, K]`.

    Returns:
      x: The sampels from the base distribution,
        with shape `[B, D]`.
      log_prob: The log-likelihood of the samples under
        the base distibution probability, with shape `[B]`.
      logabsdet: The log absolute determinant of the Jacobian,
        with shape `[B]`.
    """

    # Output containers.
    x = list()
    scales = list()

    # Initial input variable.
    y_tm1 = torch.zeros(  # pylint: disable=no-member
        size=(z.shape[0], self._output_shape[-1]),
        dtype=z.dtype,
    ).to(z.device)

    for t in range(y.shape[-2]):
      y_t = y[:, t, :]

      # Unrolls the GRU.
      z = self._decoder(y_tm1, z)

      # Predicts the location and scale of the MVN distribution.
      dloc_scale = self._locscale(z)
      dloc = dloc_scale[..., :2]
      scale = F.softplus(dloc_scale[..., 2:]) + 1e-3

      # Base distribution corresponding sample.
      x_t = (y_t - (y_tm1 + dloc)) / scale

      # Update containers.
      x.append(x_t)
      scales.append(scale)
      y_tm1 = y_t

    # Prepare tensors, reshape to [B, T, 2].
    x = torch.stack(x, dim=-2)  # pylint: disable=no-member
    scales = torch.stack(scales, dim=-2)  # pylint: disable=no-member

    # Log likelihood under base distribution.
    log_prob = self._base_dist.log_prob(x.view(x.shape[0], -1))

    # Log absolute determinant of Jacobian.
    logabsdet = torch.log(  # pylint: disable=no-member
        torch.abs(torch.prod(  # pylint: disable=no-member
            scales, dim=-1)))  # determinant == product over xy-coordinates
    logabsdet = torch.sum(logabsdet, dim=-1)  # sum over T dimension # pylint: disable=no-member

    return x, log_prob, logabsdet
