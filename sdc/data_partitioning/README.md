# How get the canonical data partitioning
To get the canonical data partitioning we provide a set of scene tags filters. The filters are located in [sdc/filters.py](https://github.com/yandex-research/shifts/blob/c4969c643b4a97d69dc1a3cfb6fb3e1af17b33bc/sdc/sdc/filters.py) file. The expected usage is as follows.
```
from sdc.filters import DATASETS_TO_FILTERS
from ysdc_dataset_api.dataset import MotionPredictionDataset


train_dataset = MotionPredictionDataset(
    dataset_path='/path/to/train/dataset/dir',
    scene_tags_fpath='/path/to/train/tags/file',
    scene_tags_filter=DATASETS_TO_FILTERS['moscow__train'],
)

dev_in_dataset = MotionPredictionDataset(
    dataset_path='/path/to/development/dataset/dir',
    scene_tags_fpath='/path/to/development/tags/file',
    scene_tags_filter=DATASETS_TO_FILTERS['moscow__development'],
)
dev_out_dataset = MotionPredictionDataset(
    dataset_path='/path/to/development/dataset/dir',
    scene_tags_fpath='/path/to/development/tags/file',
    scene_tags_filter=DATASETS_TO_FILTERS['ood__development'],
)

eval_in_dataset = MotionPredictionDataset(
    dataset_path='/path/to/evaluation/dataset/dir',
    scene_tags_fpath='/path/to/evaluation/tags/file',
    scene_tags_filter=DATASETS_TO_FILTERS['moscow__evaluation'],
)
eval_out_dataset = MotionPredictionDataset(
    dataset_path='/path/to/evaluation/dataset/dir',
    scene_tags_fpath='/path/to/evaluation/tags/file',
    scene_tags_filter=DATASETS_TO_FILTERS['ood__evaluation'],
)
