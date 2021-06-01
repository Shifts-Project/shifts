# Copyright 2021 The OATomobile Authors. All Rights Reserved.
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
"""Implements the robust imitative planning agent."""

from typing import Mapping, Union, Sequence

import numpy as np
import scipy.interpolate
import torch
import torch.optim as optim

from sdc.oatomobile.torch.baselines.behavioral_cloning import BehaviouralModel
from sdc.oatomobile.torch.baselines.deep_imitative_model import ImitativeModel


class RIPAgent:
    """The robust imitative planning agent."""

    def __init__(self,
                 algorithm: str,
                 model_name: str,
                 models: Sequence[Union[BehaviouralModel, ImitativeModel]],
                 device: str,
                 dim_lr: float = 1e-1,
                 dim_num_steps: int = 10,
                 **kwargs) -> None:
        """Constructs a robust imitative planning agent.

        Args:
          algorithm: The RIP variant used, one of {"WCM", "MA", "BCM"}.
          model_name: name of the backbone model, one of {"dim", "bc"}
          models: The deep imitative models.
        """
        # Specifices the RIP variant.
        assert algorithm in ("WCM", "MA", "BCM")
        self._algorithm = algorithm

        # Determines device, accelerator.
        self._device = device
        self._models = [model.to(self._device) for model in models]

        # Determines backbone model.
        assert model_name in ("bc", "dim")
        self._model_name = model_name

        self._dim_lr = dim_lr
        self._dim_num_steps = dim_num_steps

    def call_dim_models(self,
                        observation: Mapping[str, np.ndarray]) -> np.ndarray:
        # TODO(filangel) move this in `ImitativeModel.imitation_posterior`.
        lr = self._dim_lr
        num_steps = self._dim_num_steps
        batch_size = observation["feature_maps"].shape[0]

        # Sets initial sample to base distribution's mean.
        x = self._models[0]._decoder._base_dist.mean.clone().detach().repeat(
            batch_size, 1).view(
            batch_size,
            *self._models[0]._output_shape,
        )
        x.requires_grad = True

        # The contextual parameters, caches for efficiency.
        zs = [model._params(**observation) for model in self._models]

        # Initialises a gradient-based optimiser.
        optimizer = optim.Adam(params=[x], lr=lr)

        # Stores the best values.
        x_best = x.clone()
        loss_best = torch.ones(()).to(
            x.device) * 1000.0  # pylint: disable=no-member

        for _ in range(num_steps):
            # Resets optimizer's gradients.
            optimizer.zero_grad()

            # Operate on `y`-space.
            y, _ = self._models[0]._forward(x=x, z=zs[0])

            # No goal likelihood here, so we are just calculating the
            # imitiation prior for each of the `K` models.
            imitation_priors = list()
            for model, z in zip(self._models, zs):
                # Calculates imitation prior.
                _, log_prob, logabsdet = model._inverse(y=y,
                                                        z=z)  # should this be model.decoder.inverse?
                imitation_prior = torch.mean(
                    log_prob - logabsdet)  # pylint: disable=no-member
                imitation_priors.append(imitation_prior)

            # Aggregate scores from the `K` models.
            imitation_priors = torch.stack(imitation_priors,
                                           dim=0)  # pylint: disable=no-member

            if self._algorithm == "WCM":
                loss, _ = torch.min(-imitation_priors,
                                    dim=0)  # pylint: disable=no-member
            elif self._algorithm == "BCM":
                loss, _ = torch.max(-imitation_priors,
                                    dim=0)  # pylint: disable=no-member
            else:
                loss = torch.mean(-imitation_priors,
                                  dim=0)  # pylint: disable=no-member

            # Backward pass.
            loss.backward(retain_graph=True)

            # Performs a gradient descent step.
            optimizer.step()

            # Book-keeping
            if loss < loss_best:
                x_best = x.clone()
                loss_best = loss.clone()

        plan, _ = self._models[0]._forward(x=x_best, z=zs[0])

        ######
        plan = plan.detach().cpu().numpy()[0]  # [T, 2]

        # # TODO(filangel): clean API.
        # # Interpolates plan.
        # player_future_length = 40
        # increments = player_future_length // plan.shape[0]
        # time_index = list(range(0, player_future_length, increments))  # [T]
        # plan_interp = scipy.interpolate.interp1d(x=time_index, y=plan, axis=0)
        # xy = plan_interp(np.arange(0, time_index[-1]))
        #
        # # Appends z dimension.
        # z = np.zeros(shape=(xy.shape[0], 1))
        # return np.c_[xy, z]

        return plan

    def call_bc_models(self,
                       observation: Mapping[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def __call__(
        self,
        observation: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        """Returns the imitative prior."""
        if self._model_name == 'dim':
            return self.call_dim_models(observation)
        elif self._model_name == 'bc':
            return self.call_bc_models(observation)
        else:
            raise NotImplementedError
