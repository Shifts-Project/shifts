import json
import os

import torch

from ..proto import get_tags_from_request
from ..utils import (
    get_file_paths,
    get_gt_trajectory,
    get_track_for_transform,
    get_to_track_frame_transform,
    read_feature_map_from_file,
    request_is_valid,
    scenes_generator,
    transform2dpoints,
)


class MotionPredictionDataset(torch.utils.data.IterableDataset):
    def __init__(
            self,
            dataset_path,
            scene_tags_fpath,
            feature_producer=None,
            prerendered_dataset_path=None,
            transform_ground_truth_to_agent_frame=True,
            scene_tags_filter=None,
            trajectory_tags_filter=None,
    ):
        super(MotionPredictionDataset, self).__init__()

        self._feature_producer = feature_producer
        self._prerendered_dataset_path = prerendered_dataset_path
        self._transform_ground_truth_to_agent_frame = transform_ground_truth_to_agent_frame

        self._scene_tags_filter = _callable_or_trivial_filter(scene_tags_filter)
        self._trajectory_tags_filter = _callable_or_trivial_filter(trajectory_tags_filter)

        self._scene_file_paths = self._filter_paths(
            get_file_paths(dataset_path), scene_tags_fpath)

    @property
    def num_scenes(self):
        return len(self._scene_file_paths)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            file_paths = self._scene_file_paths
        else:
            file_paths = self._split_filepaths_by_worker(
                worker_info.id, worker_info.num_workers)

        def data_gen(_file_paths):
            for scene, fpath in scenes_generator(_file_paths, yield_fpath=True):
                for request in scene.prediction_requests:
                    if not request_is_valid(scene, request):
                        continue
                    trajectory_tags = get_tags_from_request(request)
                    if not self._trajectory_tags_filter(trajectory_tags):
                        continue
                    track = get_track_for_transform(scene, request.track_id)
                    to_track_frame_tf = get_to_track_frame_transform(track)
                    ground_truth_trajectory = get_gt_trajectory(scene, request.track_id)
                    if self._transform_ground_truth_to_agent_frame:
                        ground_truth_trajectory = transform2dpoints(
                            ground_truth_trajectory, to_track_frame_tf)
                    result = {
                        'ground_truth_trajectory': ground_truth_trajectory,
                        'scene_id': scene.id,
                        'track_id': request.track_id,
                    }

                    if self._prerendered_dataset_path:
                        fm_path = self._get_serialized_fm_path(fpath, scene.id, request.track_id)
                        result['prerendered_feature_map'] = read_feature_map_from_file(fm_path)
                        result['fm_path'] = fm_path

                    if self._feature_producer:
                        result.update(
                            self._feature_producer.produce_features(scene, to_track_frame_tf))
                    yield result

        return data_gen(file_paths)

    def _get_serialized_fm_path(self, scene_fpath, scene_id, track_id):
        base, _ = os.path.split(scene_fpath)
        _, subdir = os.path.split(base)
        return os.path.join(self._prerendered_dataset_path, subdir, f'{scene_id}_{track_id}.npy')

    def _split_filepaths_by_worker(self, worker_id, num_workers):
        n_scenes_per_worker = self.num_scenes // num_workers
        split = list(range(0, self.num_scenes, n_scenes_per_worker))
        start = split[worker_id]
        if worker_id == num_workers - 1:
            stop = self.num_scenes
        else:
            stop = split[worker_id + 1]
        return self._scene_file_paths[start:stop]

    def _callable_or_lambda_true(self, f):
        if f is None:
            return lambda x: True
        if not callable(f):
            raise ValueError('Expected callable, got {}'.format(type(f)))
        return f

    def _filter_paths(self, file_paths, scene_tags_fpath):
        valid_indices = []
        with open(scene_tags_fpath, 'r') as f:
            for i, line in enumerate(f):
                tags = json.loads(line.strip())
                if self._scene_tags_filter(tags):
                    valid_indices.append(i)
        return [file_paths[i] for i in valid_indices]


def _callable_or_trivial_filter(f):
    if f is None:
        return _trivial_filter
    if not callable(f):
        raise ValueError('Expected callable, got {}'.format(type(f)))
    return f


def _trivial_filter(x):
    return True
