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
from typing import Mapping, Union, Sequence, Tuple, Optional, Dict

import numpy as np
import torch

from sdc.cache_metadata import MetadataCache
from sdc.metrics import SDCLoss
from sdc.oatomobile.torch.baselines.behavioral_cloning import (
    BehaviouralModel)
from sdc.oatomobile.torch.baselines.deep_imitative_model import (
    ImitativeModel)


class RIPAgent:
    """The robust imitative planning agent."""

    def __init__(self,
                 per_plan_algorithm: str,
                 per_scene_algorithm: str,
                 model_name: str,
                 models: Sequence[Union[BehaviouralModel, ImitativeModel]],
                 device: str,
                 samples_per_model: int,
                 num_preds: int,
                 cache_all_preds: bool,
                 **kwargs) -> None:
        """Constructs a robust imitative planning agent.

        Args:
          algorithm: The RIP variant used.
          model_name: name of the backbone model, one of {"dim", "bc"}
          models: The deep imitative models.
        """
        # Specifies the RIP variant.
        assert per_plan_algorithm in ("WCM", "MA", "BCM", "LQ", "UQ")
        assert per_scene_algorithm in ("WCM", "MA", "BCM", "LQ", "UQ")
        self._per_plan_algorithm = per_plan_algorithm
        self._per_scene_algorithm = per_scene_algorithm

        # Number of generated trajectories per ensemble member.
        self._samples_per_model = samples_per_model

        # After applying RIP algorithm, report this number of trajectories.
        self._num_preds = num_preds

        # Determines device, accelerator.
        self._device = device
        self._models = [model.to(self._device) for model in models]

        # Determines backbone model.
        assert model_name in ("bc", "dim")
        self._model_name = model_name

        # If enabled, don't use aggregation -- just cache all
        # predictions and scores
        self.cache_all_preds = cache_all_preds

    @staticmethod
    def run_rip_aggregation(algorithm, scores_to_aggregate):
        if algorithm == 'WCM':
            return torch.min(scores_to_aggregate, dim=0).values
        elif algorithm == 'BCM':
            return torch.max(scores_to_aggregate, dim=0).values
        elif algorithm == 'MA':
            return torch.mean(scores_to_aggregate, dim=0)
        elif algorithm == 'UQ':
            return (torch.mean(scores_to_aggregate, dim=0) +
                    torch.std(scores_to_aggregate, dim=0))
        elif algorithm == 'LQ':
            return (torch.mean(scores_to_aggregate, dim=0) -
                    torch.std(scores_to_aggregate, dim=0))
        else:
            raise NotImplementedError

    def call_ensemble_members(
        self,
        **observation: Mapping[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        Q = self._samples_per_model
        K = len(self._models)  # Number of ensemble members

        # Obtain Q predicted plans from each of the K ensemble members.
        predictions = []
        for model in self._models:
            for _ in range(Q):
                predictions.append(model.forward(**observation))

        # Shape (G=K*Q, B, T, 2)
        # G: total number of plans generated
        # B: batch size
        # T: number of predicted timesteps (default: 25)
        predictions = torch.stack(predictions, dim=0)
        G, B, T, _ = predictions.size()

        # For each element in the batch, we have K models and G predictions.
        # Score each plan under each model.
        # (MC sampling over the model posterior)
        scores = []
        for i in range(K):
            model_i_scores = []
            for j in range(G):
                model_i_scores.append(
                    self._models[i].score_plans(predictions[j, :, :, :]))

            scores.append(torch.stack(model_i_scores, dim=0))

        # A high score at i, j denotes a high log probability
        # of plan j under the likelihood of model i.
        scores = torch.stack(scores, dim=0)

        if self.cache_all_preds:
            # permute axes to (B, G, T, 2)
            # predictions = np.transpose(predictions, axes=(1, 0, 2, 3))
            predictions = predictions.permute((1, 0, 2, 3))

            # permute axes to (B, G, K)
            scores = scores.permute((2, 1, 0))
            return predictions, scores, None

        # Aggregate scores from the `K` models.
        scores = self.run_rip_aggregation(
            algorithm=self._per_plan_algorithm,
            scores_to_aggregate=scores)

        # Get indices of the plans with highest RIP-aggregated scores
        # for each prediction request (scene input) in the batch
        best_plan_indices = torch.topk(scores, k=self._num_preds, dim=0).indices
        best_plans = [
            predictions[best_plan_indices[:, b], b, :, :] for b in range(B)]

        # Shape (B, num_preds, T, 2)
        best_plans = torch.stack(best_plans, dim=0)
        best_plans = best_plans.detach()

        # Report the confidence scores corresponding to our top num_preds plans
        plan_confidence_scores = [
            scores[best_plan_indices[:, b], b] for b in range(B)]

        # Shape (B, num_preds)
        plan_confidence_scores = torch.stack(plan_confidence_scores, dim=0)

        # Get per--prediction request (per-scene) confidence scores by
        # aggregating over topk plans
        pred_request_confidence_scores = self.run_rip_aggregation(
            algorithm=self._per_scene_algorithm,
            scores_to_aggregate=plan_confidence_scores.permute((1, 0)))

        plan_confidence_scores = plan_confidence_scores.detach()
        return (best_plans, plan_confidence_scores,
                pred_request_confidence_scores)

    def __call__(
        self,
        **observation: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._model_name not in {'bc', 'dim'}:
            raise NotImplementedError

        return self.call_ensemble_members(**observation)

    def train(self):
        for model in self._models:
            model.train()

    def eval(self):
        for model in self._models:
            model.eval()


def evaluate_step_rip(
    sdc_loss: SDCLoss,
    model: RIPAgent,
    batch: Dict[str, torch.Tensor],
    metadata_cache: Optional[MetadataCache],
    **kwargs
):
    model.eval()
    predictions, plan_confidence_scores, pred_request_confidence_scores = (
        model(**batch))

    for key, attr in batch.items():
        if isinstance(attr, torch.Tensor):
            batch[key] = attr.detach().cpu().numpy()

    ground_truth = batch['ground_truth_trajectory']
    predictions = predictions.detach().cpu().numpy()
    plan_confidence_scores = plan_confidence_scores.detach().cpu().numpy()

    # If we are caching all predictions, this is not computed yet
    if model.cache_all_preds:
        pred_request_confidence_scores = np.zeros(0)
    else:
        pred_request_confidence_scores = (
            pred_request_confidence_scores.detach().cpu().numpy())

    if metadata_cache is not None:
        metadata_cache.collect_batch_stats(
            predictions=predictions, batch=batch,
            plan_confidence_scores=plan_confidence_scores,
            pred_request_confidence_scores=pred_request_confidence_scores)

    if not model.cache_all_preds:
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
