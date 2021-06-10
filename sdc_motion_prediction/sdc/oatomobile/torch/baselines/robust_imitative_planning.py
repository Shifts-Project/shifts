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

import os
from typing import Mapping, Union, Sequence, Tuple

import torch

from sdc.metrics import SDCLoss
from sdc.oatomobile.torch.baselines.behavioral_cloning import (
    BehaviouralModel)
from sdc.oatomobile.torch.baselines.deep_imitative_model import (
    ImitativeModel)


class RIPAgent:
    """The robust imitative planning agent."""

    def __init__(self,
                 algorithm: str,
                 model_name: str,
                 models: Sequence[Union[BehaviouralModel, ImitativeModel]],
                 device: str,
                 eval_samples_per_model: int,
                 num_preds: int,
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

        # Number of generated trajectories per ensemble member.
        self._eval_samples_per_model = eval_samples_per_model

        # After applying RIP algorithm, report this number of trajectories.
        self._num_preds = num_preds

        # Determines device, accelerator.
        self._device = device
        self._models = [model.to(self._device) for model in models]

        # Determines backbone model.
        assert model_name in ("bc", "dim")
        self._model_name = model_name

    def call_dim_models(
        self,
        **observation: Mapping[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO(filangel) move this in `ImitativeModel.imitation_posterior`.
        Q = self._eval_samples_per_model
        K = len(self._models)  # Number of ensemble members

        # Obtain Q predicted plans for each of the K ensemble members.
        predictions = []
        for model in self._models:
            for _ in range(Q):
                predictions.append(model.forward(**observation))

        # Shape (D=K*Q, B, T, 2)
        # D: total number of plans
        # B: batch size
        # T: number of predicted timesteps (default: 25)
        predictions = torch.stack(predictions, dim=0)
        D, B, T, _ = predictions.size()

        # For each element in the batch, we have K models and Q predictions.
        # Score each plan under the imitation prior of each model:
        scores = []
        for i in range(K):
            model_i_scores = []
            for j in range(D):
                model_i_scores.append(
                    self._models[i].score_plans(predictions[j, :, :, :]))

            scores.append(torch.stack(model_i_scores, dim=0))

        # These are the imitation priors.
        # A high score at i, j corresponds to a high likelihood
        # of plan j under the imitation prior of model i.
        scores = torch.stack(scores, dim=0)

        # NO LONGER DO THIS; just use RIP aggregators
        # # Get per-plan confidence scores by computing the variance over the
        # # K model scores for each plan, and negating
        # plan_confidence_scores = -torch.var(scores, dim=0)  # shape: (D, B)

        # Aggregate scores from the `K` models.
        if self._algorithm == 'WCM':
            scores = torch.min(scores, dim=0).values
        elif self._algorithm == 'BCM':
            scores = torch.max(scores, dim=0).values
        elif self._algorithm == 'MA':
            scores = torch.mean(scores, dim=0)
        else:
            raise NotImplementedError

        # Get per--prediction request confidence scores by taking the mean
        # over all plans
        # TODO: maybe we should only take the mean over the topk plans instead?
        pred_request_confidence_scores = torch.mean(scores, dim=0)

        # Get indices of the plans with highest RIP-aggregated imitation priors
        # for each prediction request (scene input) in the batch
        best_plan_indices = torch.topk(scores, k=self._num_preds, dim=0).indices
        best_plans = [
            predictions[best_plan_indices[:, b], b, :, :]
            for b in range(B)]

        # Shape (B, num_preds, T, 2)
        best_plans = torch.stack(best_plans, dim=0)
        best_plans = best_plans.detach()

        # Report the confidence scores corresponding to our top num_preds plans
        plan_confidence_scores = [
            scores[best_plan_indices[:, b], b]
            for b in range(B)]

        # Shape (B, num_preds)
        plan_confidence_scores = torch.stack(plan_confidence_scores, dim=0)
        plan_confidence_scores = plan_confidence_scores.detach()
        return (best_plans, plan_confidence_scores,
                pred_request_confidence_scores)

        #     # imitation_priors = torch.stack(imitation_priors,
        # #                                    dim=0)  # pylint: disable=no-member
        #
        #     if self._algorithm == "WCM":
        #         loss, _ = torch.min(-imitation_priors,
        #                             dim=0)  # pylint: disable=no-member
        #     elif self._algorithm == "BCM":
        #         loss, _ = torch.max(-imitation_priors,
        #                             dim=0)  # pylint: disable=no-member
        #     else:
        #         loss = torch.mean(-imitation_priors,
        #                           dim=0)  # pylint: disable=no-member
        #
        #     # Backward pass.
        #     loss.backward(retain_graph=True)
        #
        #     # Performs a gradient descent step.
        #     optimizer.step()
        #
        #     # Book-keeping
        #     if loss < loss_best:
        #         x_best = x.clone()
        #         loss_best = loss.clone()
        #
        # plan, _ = self._models[0]._forward(x=x_best, z=zs[0])
        # plan = plan.detach().cpu()[0]  # [T, 2]
        # return plan

    def call_bc_models(
        self,
        **observation: Mapping[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def __call__(
        self,
        **observation: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the imitative prior."""
        if self._model_name == 'dim':
            return self.call_dim_models(**observation)
        elif self._model_name == 'bc':
            return self.call_bc_models(**observation)
        else:
            raise NotImplementedError

    def train(self):
        for model in self._models:
            model.train()

    def eval(self):
        for model in self._models:
            model.eval()


def evaluate_step_rip(
    sdc_loss: SDCLoss,
    model: RIPAgent,
    batch: Mapping[str, torch.Tensor],
):
    model.eval()
    predictions, plan_confidence_scores, pred_request_confidence_scores = model(
        **batch)
    ground_truth = batch['ground_truth_trajectory']
    sdc_loss.cache_batch_losses(
        predictions_list=predictions,
        ground_truth_batch=ground_truth,
        plan_confidence_scores_list=plan_confidence_scores,
        pred_request_confidence_scores=pred_request_confidence_scores)


def load_rip_checkpoints(
    model: RIPAgent,
    device: str,
    k: int,
    checkpoint_dir: str
) -> RIPAgent:
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        raise ValueError(
            f'Expected trained model checkpoints for RIP at '
            f'checkpoint_dir {checkpoint_dir}. The directory was created '
            f'for your convenience.\n.')

    # Load RIP ensemble members from checkpoint dir

    checkpoint_file_names = os.listdir(checkpoint_dir)

    models_loaded = 0
    for file_name in checkpoint_file_names:
        ckpt_path = os.path.join(checkpoint_dir, file_name)
        try:
            model._models[
                models_loaded].load_state_dict(
                torch.load(ckpt_path, map_location=device))
            models_loaded += 1
            print(f'Loaded ensemble member {models_loaded} '
                  f'from path {ckpt_path}')
        except Exception as e:
            raise Exception(
                f"Failed in loading checkpoint at path "
                f"{ckpt_path} with exception:\n{e}")

    assert models_loaded == len(model._models), (
        f'Failed to load all {k} ensemble members. '
        f'Does the checkpoint directory at {checkpoint_dir} '
        f'contain all of them?')

    print(f'Successfully loaded all {k} ensemble members.')
    return model
