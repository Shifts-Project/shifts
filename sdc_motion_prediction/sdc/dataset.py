from collections import defaultdict
from pprint import pprint
from typing import Optional, List

import torch

from sdc.constants import *
from sdc.filters import *
from ysdc_dataset_api.dataset import MotionPredictionDataset
from ysdc_dataset_api.features import FeatureRenderer
from ysdc_dataset_api.utils import get_file_paths
import json


def load_renderer():
    return FeatureRenderer(RENDERER_CONFIG)


def load_overfit_set_file_paths(
        dataset_path, scene_tags_fpath, filter, n_overfit_examples=100):
    file_paths = get_file_paths(dataset_path=dataset_path)
    valid_indices = []

    with open(scene_tags_fpath, 'r') as f:
        for i, line in enumerate(f):
            tags = json.loads(line.strip())
            if filter(tags):
                valid_indices.append(i)

            if len(valid_indices) >= n_overfit_examples:
                break

    print(
        f'Built overfit dataset: used '
        f'{len(valid_indices)}/{len(file_paths)} '
        f'total scenes.')
    return [file_paths[i] for i in valid_indices]


def load_datasets(c, splits: Optional[List[str]] = None):
    if c.debug_overfit_test_data_only:
        splits = ['test']
    elif c.data_use_prerendered:
        splits = ['train', 'test']
    if splits is not None:
        print(f'Loading datasets for splits {splits}.')

    dataset_args = {}
    if not c.data_use_prerendered:
        dataset_args['feature_producer'] = load_renderer()
        dataset_args['transform_ground_truth_to_agent_frame'] = True

    datasets = defaultdict(dict)
    for dataset_split, split_dict in DATASETS_TO_FILTERS.items():
        if splits is not None:
            if dataset_split not in splits:
                continue

        print(f'\nLoading {dataset_split} dataset(s).')
        split_dataset_path = SPLIT_TO_PB_DATASET_PATH[dataset_split]
        split_scene_tags_fpath = SPLIT_TO_SCENE_TAGS_PATH[dataset_split]
        split_dataset_path = f'{c.dir_data}{split_dataset_path}'
        split_scene_tags_fpath = f'{c.dir_data}{split_scene_tags_fpath}'
        split_prerendered_dataset_path = (
            f'{c.dir_data}{SPLIT_TO_RENDERED_DATASET_PATH[dataset_split]}')

        for dataset_key, scene_tags_filter_fn in split_dict.items():
            print(f'Loading dataset {dataset_key}.')

            # Only return a small subset of each set (10 examples)
            if c.debug_overfit_eval:
                overfit_set_file_paths = load_overfit_set_file_paths(
                    dataset_path=split_dataset_path,
                    scene_tags_fpath=split_scene_tags_fpath,
                    filter=scene_tags_filter_fn,
                    n_overfit_examples=c.debug_overfit_n_examples)
                dataset_args[
                    'pre_filtered_scene_file_paths'] = overfit_set_file_paths

            if c.data_use_prerendered:
                dataset_args['prerendered_dataset_path'] = (
                    split_prerendered_dataset_path)

            datasets[dataset_split][dataset_key] = MotionPredictionDataset(
                dataset_path=split_dataset_path,
                scene_tags_fpath=split_scene_tags_fpath,
                scene_tags_filter=scene_tags_filter_fn,
                trajectory_tags_filter=None,
                **dataset_args)
            print(f'Loaded dataset {dataset_key}.')

    print('Finished loading all datasets.')
    pprint(datasets)
    return datasets


def load_dataloaders(datasets, c):
    batch_size = c.exp_batch_size
    num_workers = c.data_num_workers
    prefetch_factor = c.data_prefetch_factor

    if c.debug_overfit_test_data_only:
        train_dataloader = torch.utils.data.DataLoader(
            datasets['test']['moscow__test'], batch_size=batch_size,
            num_workers=num_workers, pin_memory=True,
            prefetch_factor=prefetch_factor)
        eval_dataloaders = {}
    else:
        train_dataloader = torch.utils.data.DataLoader(
            datasets['train']['moscow__train'], batch_size=batch_size,
            num_workers=num_workers, pin_memory=True,
            prefetch_factor=prefetch_factor)

        # Load dataloaders for in- and out-of-domain
        # validation and test datasets.
        eval_dataloaders = defaultdict(dict)
        for eval_mode in ['validation', 'test']:
            if eval_mode not in datasets.keys():
                continue

            eval_dataset_dict = datasets[eval_mode]
            for dataset_key, dataset in eval_dataset_dict.items():
                eval_dataloaders[
                    eval_mode][dataset_key] = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, num_workers=num_workers,
                    pin_memory=True, prefetch_factor=prefetch_factor)

    return train_dataloader, eval_dataloaders


def get_torch_dtype(dtype_name):
    if dtype_name == 'float32':
        dtype = torch.float32
    elif dtype_name == 'float64':
        dtype = torch.float64
    else:
        raise NotImplementedError

    return dtype


def torch_cast_to_dtype(obj, dtype_name):
    if dtype_name == 'float32':
        obj = obj.float()
    elif dtype_name == 'float64':
        obj = obj.double()
    elif dtype_name == 'long':
        obj = obj.long()
    else:
        raise NotImplementedError

    return obj
