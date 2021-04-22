import os
import random

import numpy as np
from google.protobuf.internal.decoder import _DecodeVarint32
from torch.utils.data import Dataset, IterableDataset

from ysdc_dataset_api.proto import Scene, proto_to_dict
from ysdc_dataset_api.rendering import FeatureRenderer
from ysdc_dataset_api.utils import get_track_to_fm_transform, get_track_for_transform


N_SCENES_PER_FILE = 5000


class MotionPredictionDataset(IterableDataset):
    def __init__(
            self,
            dataset_path,
            renderer_config=None,
            scene_tags_filter=None,
            trajectory_tags_filter=None,
        ):
        super(MotionPredictionDataset, self).__init__()
        self._dataset_path = dataset_path
        self._file_names = [f for f in os.listdir(dataset_path) if f.endswith('.bin')]
        self._renderer = FeatureRenderer(renderer_config)

        self._scene_tags_filter = self._callable_or_lambda_true(scene_tags_filter)
        self._trajectory_tags_filter = self._callable_or_lambda_true(trajectory_tags_filter)

    def __iter__(self):
        def data_gen():
            for fname in self._file_names:
                fpath = os.path.join(self._dataset_path, fname)
                with open(fpath, 'rb') as f:
                    buf = f.read()
                    n = 0
                    while n < len(buf):
                        msg_len, new_pos = _DecodeVarint32(buf, n)
                        n = new_pos
                        msg_buf = buf[n:n+msg_len]
                        n += msg_len
                        scene = Scene()
                        scene.ParseFromString(msg_buf)
                        scene_tags = proto_to_dict(scene.scene_tags)
                        if not self._scene_tags_filter(scene_tags):
                            continue
                        for request in scene.prediction_requests:
                            request_tags = proto_to_dict(request)['trajectory_tags']
                            if not self._trajectory_tags_filter(request_tags):
                                continue
                            track = get_track_for_transform(scene, request.track_id)
                            track_to_fm_transform = get_track_to_fm_transform(track)
                            feature_maps = self._renderer.render_features(
                                scene, track_to_fm_transform)
                            gt_trajectory = transform_points(
                                get_gt_trajectory(scene, request.track_id), transform)
                            yield {
                                'feature_maps': feature_maps,
                                'gt_trajectory': gt_trajectory,
                            }
        return data_gen()

    def _callable_or_lambda_true(self, f):
        if f is None:
            return lambda x: True
        if not callable(f):
            raise ValueError('Expected callable, got {}'.format(type(f)))
        return f


class MotionPredictionDatasetV2(Dataset):
    def __init__(self, dataset_path):
        super(MotionPredictionDatasetV2, self).__init__()
        self._filepaths = self._get_filepaths(dataset_path)

    def __getitem__(self, idx):
        file_idx = idx // N_SCENES_PER_FILE
        scene_in_file_idx = idx % N_SCENES_PER_FILE
        scene = self._read_scene_by_idx(self._filepaths[file_idx], scene_in_file_idx)
        return {
            'pedestrians': np.random.rand(random.randint(0, 5), 25, 2),
            'vehicles': np.random.rand(random.randint(0, 10), 25, 2),
            'gt_vehicles': np.random.rand(random.randint(0, 3), 25, 2),
        }

    def __len__(self):
        return len(self._filepaths) * N_SCENES_PER_FILE

    def _read_scene_by_idx(self, filepath, idx):
        assert idx < N_SCENES_PER_FILE
        with open(filepath, 'rb') as f:
            file_buf = f.read()
        scene_idx = 0
        sep_pos = 0
        while sep_pos < len(file_buf):
            msg_len, new_sep_pos = _DecodeVarint32(file_buf, sep_pos)
            sep_pos = new_sep_pos
            msg_buf = file_buf[sep_pos:sep_pos + msg_len]
            sep_pos += msg_len
            if scene_idx == idx:
                scene = Scene()
                scene.ParseFromString(msg_buf)
                return scene
            scene_idx += 1

    def _get_filepaths(self, dataset_dir):
        return sorted([
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.endswith('.bin')
        ])
