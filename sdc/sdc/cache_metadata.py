import datetime
import os
from collections import defaultdict
from functools import partial
from typing import Text, Mapping, List, Sequence, Union, Dict

import numpy as np
import pandas as pd
from scipy.special import softmax

from sdc.constants import (
    VALID_TRAJECTORY_TAGS, SCENE_TAG_TYPE_TO_OPTIONS,
    VALID_BASE_METRICS, VALID_AGGREGATORS)
from ysdc_dataset_api.evaluation.metrics import compute_all_aggregator_metrics


class MetadataCache:
    def __init__(self, full_model_name, c):
        self.full_model_name = full_model_name
        self.c = c

        # ** Caching Full Results **
        # Done if we wish to perform post-hoc analyses of model predictions
        # e.g., how does confidence/ADE correlate with trajectory types?
        metadata_cache_dir = (
            f'{c.dir_data}/metadata_cache'
            if not c.dir_metadata_cache else c.dir_metadata_cache)
        self.metadata_cache_dir = (
            os.path.join(metadata_cache_dir, full_model_name))
        os.makedirs(self.metadata_cache_dir, exist_ok=True)

        # Per-Trajectory Metadata
        # These two are needed to uniquely identify a given prediction request
        self.scene_ids = []
        self.request_ids = []

        # Specify per-prediction request attributes that will need to be
        # aggregated (because they are cached per batch)
        # Don't include 'scene_ids', which is already aggregated
        self.request_attributes = [
            'request_ids'] + VALID_TRAJECTORY_TAGS

        if self.c.rip_cache_all_preds:
            self.predictions = []
            self.ground_truth_trajectories = []

            # Will ultimately have shape (M, D_i), where
            #   M = number of datapoints in evaluation set
            #   D_i = number of predictions per scene,
            #       which can vary per scene i
            # TODO: Extend to variant D_i
            self.plan_confidence_scores = []

            # Since we are collecting everything here, don't compute metrics

        else:
            # Will ultimately have shape (M,)
            self.pred_request_confidence_scores = []

            metric_attributes = [
                f'{aggregator}{base_metric.upper()}'
                for aggregator in VALID_AGGREGATORS
                for base_metric in VALID_BASE_METRICS]
            self.request_attributes += ['pred_request_confidence_scores']
            self.request_attributes += metric_attributes

            # Metrics (e.g., minADE)
            for attr in metric_attributes:
                setattr(self, attr, [])

        # Non-mutually exclusive trajectory tags
        for attr in VALID_TRAJECTORY_TAGS:
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
        predictions: np.ndarray,
        batch: Mapping[str, Union[np.ndarray, List[Text]]],
        pred_request_confidence_scores: np.ndarray,
        plan_confidence_scores: np.ndarray,
    ):
        # TODO: support for varying # plans per prediction request
        for obj in [predictions, plan_confidence_scores,
                    pred_request_confidence_scores]:
            if not isinstance(obj, np.ndarray):
                raise NotImplementedError

        ground_truth = batch['ground_truth_trajectory']

        # Just cache predictions, plan confidences and ground truths --
        # we will compute losses post-hoc
        # (see example/compare_rip_models.ipynb)
        if self.c.rip_cache_all_preds:
            self.predictions.append(predictions)
            self.plan_confidence_scores.append(plan_confidence_scores)
            self.ground_truth_trajectories.append(ground_truth)
        else:
            # Cache losses
            metrics_dict = compute_all_aggregator_metrics(
                per_plan_confidences=plan_confidence_scores,
                predictions=predictions,
                ground_truth=ground_truth)
            for metrics_key, losses in metrics_dict.items():
                metric_attr = getattr(self, metrics_key)
                metric_attr.append(losses)

            # Cache pred request confidence estimates
            self.pred_request_confidence_scores.append(
                pred_request_confidence_scores)

        # Cache scene and request IDs
        batch_scene_ids = batch['scene_id']  # type: List[Text]
        self.scene_ids += batch_scene_ids
        batch_request_ids = batch['request_id']
        self.request_ids.append(batch_request_ids)

        # Cache scene tags
        for batch_index, scene_id in enumerate(batch_scene_ids):
            self.scene_id_to_num_vehicles[
                scene_id] = batch['num_vehicles'][batch_index]

        for scene_tag_type in SCENE_TAG_TYPE_TO_OPTIONS.keys():
            scene_tag_options = SCENE_TAG_TYPE_TO_OPTIONS[scene_tag_type]
            for scene_tag_option in scene_tag_options:
                option_key = f'{scene_tag_type}__{scene_tag_option}'
                scene_id_to_scene_tag_option = getattr(self, option_key)
                for batch_index, scene_id in enumerate(batch_scene_ids):
                    scene_id_to_scene_tag_option[scene_id] = batch[
                        option_key][batch_index]

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
            if isinstance(request_attribute[0], np.ndarray):
                cat_attr = np.concatenate(request_attribute, axis=0)
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
                    scene_data[attr_key].append(scene_id_to_attr[scene_id])

        scene_data['scene_ids'] = sorted_scene_ids

        scene_df = pd.DataFrame(data=scene_data)
        request_df['dataset_key'] = dataset_key
        scene_df['dataset_key'] = dataset_key
        self.store_request_and_scene_dfs(request_df, scene_df)

        if self.c.rip_cache_all_preds:
            self.store_preds_gt_scores(dataset_key)

        self.__init__(self.full_model_name, self.c)

    def store_preds_gt_scores(self, dataset_key):
        predictions = np.concatenate(self.predictions, axis=0)
        plan_confidence_scores = np.concatenate(
            self.plan_confidence_scores, axis=0)
        ground_truth_trajectories = np.concatenate(
            self.ground_truth_trajectories, axis=0)
        request_ids = np.concatenate(
            self.request_ids, axis=0)
        with open(os.path.join(self.metadata_cache_dir,
                               f'{dataset_key}__predictions.npy'), 'wb') as f:
            np.save(f, predictions)
        with open(
            os.path.join(self.metadata_cache_dir,
                         f'{dataset_key}__plan_conf_scores.npy'), 'wb') as f:
            np.save(f, plan_confidence_scores)
        with open(
            os.path.join(self.metadata_cache_dir,
                         f'{dataset_key}__gt_trajectories.npy'), 'wb') as f:
            np.save(f, ground_truth_trajectories)
        with open(
            os.path.join(self.metadata_cache_dir,
                         f'{dataset_key}__request_ids.npy'), 'wb') as f:
            np.save(f, request_ids)
        print(f'Saved predictions, conf scores, gt, request ids to '
              f'{self.metadata_cache_dir}')

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


"""
*** Helper Functions: Loading Data for Post-Hoc Analysis ***

Used with the --rip_cache_all_preds=True flag to load all predictions,
ground-truth values, and per-plan confidence scores from storage.
"""


def load_dataset_key_to_arrs(
    metadata_cache_dir: str,
    chosen_dataset_keys: Sequence[str] = (
        'moscow__validation', 'ood__validation', 'moscow__test', 'ood__test')
) -> Dict[str, Dict]:
    # TODO: remove test from these keys
    dataset_key_to_arrs = defaultdict(dict)

    for file_name in os.listdir(metadata_cache_dir):
        if '.npy' in file_name:
            is_ood, dataset_split, arr_type_and_suffix = file_name.split('__')
            arr_type, suffix = arr_type_and_suffix.split('.')
            dataset_key = f'{is_ood}__{dataset_split}'
            if dataset_key in chosen_dataset_keys:
                with open(os.path.join(metadata_cache_dir, file_name),
                          'rb') as f:
                    dataset_key_to_arrs[dataset_key][arr_type] = np.load(
                        f, allow_pickle=True)

    return dataset_key_to_arrs


def construct_full_dev_eval_sets(
    dataset_key_to_arrs: Dict[str, Dict]
) -> Dict[str, Dict]:
    """Given a dataset_key_to_arrs object as loaded from
    `load_dataset_key_to_arrs` above, add full__validation and
    full__test datasets, composed of the corresponding moscow and ood sets.
    # TODO: remove test dataset stuff
    """
    field_keys = [
        'predictions', 'plan_conf_scores', 'gt_trajectories', 'request_ids']
    full_dataset_to_subdataset_names = {
        'full__validation': ['moscow__validation', 'ood__validation'],
        'full__test': ['moscow__test', 'ood__test'],
    }
    for full_dataset_key, subdataset_keys in (
            full_dataset_to_subdataset_names.items()):
        is_ood_arr = []
        for subdataset_key in subdataset_keys:
            if 'moscow' in subdataset_key:
                is_ood_arr.append(np.zeros_like(
                    dataset_key_to_arrs[subdataset_key]['request_ids']))
            else:
                is_ood_arr.append(np.ones_like(
                    dataset_key_to_arrs[subdataset_key]['request_ids']))

        dataset_key_to_arrs[full_dataset_key]['is_ood'] = np.concatenate(
            is_ood_arr, axis=0)

        for field in field_keys:
            field_arr = []

            for subdataset_key in subdataset_keys:
                field_arr.append(dataset_key_to_arrs[subdataset_key][field])

            dataset_key_to_arrs[full_dataset_key][field] = np.concatenate(
                field_arr, axis=0)

    return dataset_key_to_arrs


def load_request_df(metadata_cache_path):
    with open(os.path.join(metadata_cache_path, 'request.tsv')) as f:
        return pd.read_csv(f, sep='\t')
