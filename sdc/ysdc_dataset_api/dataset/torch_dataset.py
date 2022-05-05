import torch

from .dataset import DatasetBase
from .utils import data_generator


class MotionPredictionDataset(DatasetBase, torch.utils.data.IterableDataset):
    def __init__(self, *args, **kwargs):
        super(MotionPredictionDataset, self).__init__(*args, **kwargs)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            file_paths = self._scene_file_paths
        else:
            file_paths = self._split_filepaths_by_worker(
                worker_info.id, worker_info.num_workers)

        return data_generator(
            file_paths,
            self._feature_producers,
            self._prerendered_dataset_path,
            self._transform_ground_truth_to_agent_frame,
            self._trajectory_tags_filter,
            self._yield_metadata,
            yield_scene_tags=True)

    def _split_filepaths_by_worker(self, worker_id, num_workers):
        n_scenes_per_worker = self.num_scenes // num_workers
        split = list(range(0, self.num_scenes, n_scenes_per_worker))
        start = split[worker_id]
        if worker_id == num_workers - 1:
            stop = self.num_scenes
        else:
            stop = split[worker_id + 1]
        return self._scene_file_paths[start:stop]
