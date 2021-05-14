from collections import defaultdict
from pprint import pprint
from typing import Optional

import torch

from sdc.constants import *
from sdc.filters import *
from ysdc_dataset_api.dataset import MotionPredictionDataset
from ysdc_dataset_api.features import FeatureRenderer


def load_renderer():
    return FeatureRenderer(RENDERER_CONFIG)


def load_datasets(c, split: Optional[str] = None):
    if split is not None:
        print(f'Loading datasets for split {split}.')

    renderer = load_renderer()
    datasets = defaultdict(dict)
    for dataset_split, split_dict in DATASETS_TO_FILTERS.items():
        if split is not None:
            if dataset_split != split:
                continue

        print(f'\nLoading {dataset_split} dataset(s).')
        split_dataset_path = SPLIT_TO_DATASET_PATH[dataset_split]
        split_scene_tags_fpath = SPLIT_TO_SCENE_TAGS_PATH[dataset_split]
        split_dataset_path = f'{c.dir_data}{split_dataset_path}'
        split_scene_tags_fpath = f'{c.dir_data}{split_scene_tags_fpath}'
        for dataset_key, scene_tags_filter_fn in split_dict.items():
            print(f'Loading dataset {dataset_key}.')
            datasets[dataset_split][dataset_key] = MotionPredictionDataset(
                dataset_path=split_dataset_path,
                scene_tags_fpath=split_scene_tags_fpath,
                feature_producer=renderer,
                transform_ground_truth_to_agent_frame=True,
                scene_tags_filter=scene_tags_filter_fn,
                trajectory_tags_filter=None)
            print(f'Loaded dataset {dataset_key}.')

    print('Finished loading all datasets.')
    pprint(datasets)
    return datasets


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