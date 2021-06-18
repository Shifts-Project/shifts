import datetime
import os
from collections import defaultdict
from typing import Text, Mapping, List, Union

import pandas as pd
import torch

from sdc.constants import (
    VALID_TRAJECTORY_TAGS, SCENE_TAG_TYPE_TO_OPTIONS,
    VALID_BASE_METRICS, VALID_AGGREGATORS)
from sdc.metrics import SDCLoss


class MetadataCache:
    def __init__(self, full_model_name, c):
        self.full_model_name = full_model_name
        self.c = c
        self.softmax = torch.nn.Softmax(dim=0)

        # ** Caching Full Results **
        # Done if we wish to perform post-hoc analyses of model predictions
        # e.g., how does confidence/ADE correlate with trajectory types?
        metadata_cache_dir = (
            f'{c.dir_data}/metadata_cache'
            if not c.dir_metadata_cache else c.dir_metadata_cache)
        self.metadata_cache_dir = (
            os.path.join(metadata_cache_dir, full_model_name))
        os.makedirs(self.metadata_cache_dir, exist_ok=True)

        # Will ultimately have shape (M, D_i), where
        #   M = number of datapoints in evaluation set
        #   D_i = number of predictions per scene, which can vary per scene i
        # TODO: Extend to variant D_i
        self.plan_confidence_scores = []

        # Will ultimately have shape (M,)
        self.pred_request_confidence_scores = []

        # Per-Trajectory Metadata
        self.predictions = []
        self.ground_truth_trajectories = []
        self.scene_ids = []
        self.request_ids = []

        # Don't include 'scene_ids', which is already aggregated
        self.request_attributes = [
            'pred_request_confidence_scores',
            'request_ids'] + VALID_TRAJECTORY_TAGS

        metric_attributes = [
            f'{aggregator}{base_metric.upper()}'
            for aggregator in VALID_AGGREGATORS
            for base_metric in VALID_BASE_METRICS]

        self.request_attributes += metric_attributes

        # Non-mutually exclusive trajectory tags and metrics (e.g., minADE)
        for attr in VALID_TRAJECTORY_TAGS + metric_attributes:
            setattr(self, attr, [])

        # Per-Scene Metadata
        self.scene_id_to_num_vehicles = defaultdict(int)

        for scene_tag_type in SCENE_TAG_TYPE_TO_OPTIONS.keys():
            scene_tag_options = SCENE_TAG_TYPE_TO_OPTIONS[scene_tag_type]
            for scene_tag_option in scene_tag_options:
                setattr(
                    self, f'{scene_tag_type}__{scene_tag_option}',
                    defaultdict(bool))

    def collect_batch_stats(
        self,
        predictions: torch.Tensor,
        batch: Mapping[str, Union[torch.Tensor, List[Text]]],
        pred_request_confidence_scores: torch.Tensor,
        plan_confidence_scores: torch.Tensor,
    ):
        # TODO: support for varying # plans per prediction request
        # TODO: switch all to numpy arrs
        for obj in [predictions, plan_confidence_scores,
                    pred_request_confidence_scores]:
            if not isinstance(obj, torch.Tensor):
                raise NotImplementedError

        # Cache predictions + ground truth
        # self.predictions.append(predictions)
        #
        ground_truth = batch['ground_truth_trajectory']
        # self.ground_truth_trajectories.append(ground_truth)

        # Cache losses
        metrics_dict = SDCLoss.compute_all_aggregator_metrics(
            per_plan_confidences=plan_confidence_scores,
            predictions=predictions,
            ground_truth=ground_truth)
        for metrics_key, losses in metrics_dict.items():
            metric_attr = getattr(self, metrics_key)
            metric_attr.append(losses.detach().cpu())

        # Cache confidence estimates
        self.pred_request_confidence_scores.append(
            pred_request_confidence_scores.detach().cpu())
        # self.plan_confidence_scores.append(plan_confidence_scores)

        # Cache scene and request IDs
        batch_scene_ids = batch['scene_id']  # type: List[Text]
        batch_request_ids = batch['request_id']
        self.scene_ids += batch_scene_ids
        self.request_ids.append(batch_request_ids.detach().cpu())

        # Cache scene tags
        for batch_index, scene_id in enumerate(batch_scene_ids):
            self.scene_id_to_num_vehicles[
                scene_id] = batch['num_vehicles'][batch_index].item()

        for scene_tag_type in SCENE_TAG_TYPE_TO_OPTIONS.keys():
            scene_tag_options = SCENE_TAG_TYPE_TO_OPTIONS[scene_tag_type]
            for scene_tag_option in scene_tag_options:
                option_key = f'{scene_tag_type}__{scene_tag_option}'
                scene_id_to_scene_tag_option = getattr(self, option_key)
                for batch_index, scene_id in enumerate(batch_scene_ids):
                    scene_id_to_scene_tag_option[scene_id] = batch[
                        option_key][batch_index].detach().cpu()

        # Cache trajectory tags
        for trajectory_tag in VALID_TRAJECTORY_TAGS:
            trajectory_tag_attr = getattr(self, trajectory_tag)
            trajectory_tag_attr.append(batch[trajectory_tag])

    def cache_dataset_stats(self, dataset_key):
        # TODO: Extend to variant D_i
        request_data = {}  # Each attribute is a 1D list

        for request_attribute_name in self.request_attributes:
            request_attribute = getattr(self, request_attribute_name, None)
            if request_attribute is None:
                raise ValueError(
                    f'Request attribute not found: {request_attribute}')
            assert isinstance(request_attribute, list)
            assert len(request_attribute) > 0
            if isinstance(request_attribute[0], torch.Tensor):
                cat_attr = torch.cat(
                        request_attribute, dim=0).numpy()
                request_data[request_attribute_name] = cat_attr

        request_data['scene_ids'] = self.scene_ids
        request_df = pd.DataFrame(data=request_data)

        sorted_scene_ids = list(sorted(list(set(self.scene_ids))))
        scene_data = defaultdict(list)

        for scene_id in sorted_scene_ids:
            scene_data['num_vehicles'].append(
                self.scene_id_to_num_vehicles[scene_id])

        for scene_tag_type in SCENE_TAG_TYPE_TO_OPTIONS.keys():
            scene_tag_options = SCENE_TAG_TYPE_TO_OPTIONS[scene_tag_type]
            for scene_tag_option in scene_tag_options:
                attr_key = f'{scene_tag_type}__{scene_tag_option}'
                scene_id_to_attr = getattr(self, attr_key)
                for scene_id in sorted_scene_ids:
                    scene_data[attr_key].append(
                        scene_id_to_attr[scene_id].numpy())

        scene_data['scene_ids'] = sorted_scene_ids

        scene_df = pd.DataFrame(data=scene_data)
        request_df['dataset_key'] = dataset_key
        scene_df['dataset_key'] = dataset_key
        self.store_request_and_scene_dfs(request_df, scene_df)
        self.__init__(self.full_model_name, self.c)

    def store_request_and_scene_dfs(self, request_df, scene_df):
        request_path = os.path.join(self.metadata_cache_dir, 'request.tsv')
        scene_path = os.path.join(self.metadata_cache_dir, 'scene.tsv')

        df_arr = [
            ('request_df', request_path, request_df),
            ('scene_df', scene_path, scene_df)]

        for df in [request_df, scene_df]:
            df['run_datetime'] = datetime.datetime.now()
            df['run_datetime'] = pd.to_datetime(df['run_datetime'])

        for df_type, df_path, df in df_arr:
            # Update or initialize DataFrames
            try:
                with open(df_path, 'r') as f:
                    previous_df = pd.read_csv(f, sep='\t')
                df = pd.concat([previous_df, df])
                action_str = 'updated'
            except FileNotFoundError:
                print(f'No previous results found at path {df_path}. '
                      f'Storing a new {df_type}.')
                action_str = 'stored initial'

            # Store to file
            with open(df_path, 'w') as f:
                df.to_csv(path_or_buf=f, sep='\t', index=False)

            print(f'Successfully {action_str} {df_type} at {df_path}.')
