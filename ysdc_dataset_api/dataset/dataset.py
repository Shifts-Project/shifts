import os

from google.protobuf.internal.decoder import _DecodeVarint32
from torch.utils.data import IterableDataset

from ysdc_dataset_api.proto import Scene
from ysdc_dataset_api.rendering import FeatureRenderer
from ysdc_dataset_api.utils import get_traack_to_fm_transform, get_track_for_transform


N_SCENES_PER_FILE = 5000


class MotionPredictionDataset(IterableDataset):
    def __init__(self, dataset_path, filtration_config=None, renderer_config=None):
        super(MotionPredictionDataset, self).__init__()
        self._dataset_path = dataset_path
        self._file_names = [f for f in os.listdir(dataset_path) if f.endswith('.bin')]
        self._filtration_config = filtration_config
        self._renderer = FeatureRenderer(renderer_config)

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
                        if not self._scene_is_valid(scene):
                            continue
                        for request in scene.prediction_requests:
                            if not self._request_is_valid(request):
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

    def _scene_is_valid(self, scene):
        # Filter by scene tags
        return True

    def _request_is_valid(self, request):
        # Filter by trajectory tags
        return True
