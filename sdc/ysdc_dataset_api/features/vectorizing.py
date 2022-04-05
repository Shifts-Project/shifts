from typing import Dict

import numpy as np

from .producing import FeatureProducerBase
from ..proto import PredictionRequest, Scene
from ..utils import (
    get_latest_track_state_by_id,
    get_to_track_frame_transform,
    transform_2d_points,
    transform_2d_vectors,
)


class FeatureVectorizer(FeatureProducerBase):
    FEATURE_NAME_TO_NUM_CHANNELS = {
        'position': 2,
        'velocity': 2,
        'acceleration': 2,
        'yaw': 1,
    }

    def __init__(self, config):
        self._config = config
        self._history_indices = self._get_history_indices()
        self._n_history_steps = len(self._history_indices)

    def _get_history_indices(self):
        return list(range(
            -self._config['time_grid_params']['stop'] - 1,
            -self._config['time_grid_params']['start'],
            self._config['time_grid_params']['step'],
        ))

    def _get_num_channels(self):
        return sum(self.FEATURE_NAME_TO_NUM_CHANNELS[name] for name in self._config['features'])

    def _get_track_at_ts(self, scene, track_id, ts):
        for track in scene.past_vehicle_tracks[ts].tracks:
            if track.track_id == track_id:
                return track
        return None

    def _get_features(self, track, transform):
        values = []
        if 'position' in self._config['features']:
            values.append(self._get_position_values(track, transform))
        if 'velocity' in self._config['features']:
            values.append(self._get_velocity_values(track, transform))
        if 'acceleration' in self._config['features']:
            values.append(self._get_acceleration_values(track, transform))
        if 'yaw' in self._config['features']:
            values.append([track.yaw])
        return np.concatenate(values).astype(np.float64)

    def _get_position_values(self, track, transform):
        position = np.asarray([[track.position.x, track.position.y]], dtype=np.float32)
        position = transform_2d_points(position, transform)
        return position[0]

    def _get_velocity_values(self, track, transform):
        velocity = np.asarray(
            [[track.linear_velocity.x, track.linear_velocity.y]], dtype=np.float32)
        velocity = transform_2d_vectors(velocity, transform)
        return velocity[0]

    def _get_acceleration_values(self, track, transform):
        acceleration = np.asarray(
            [[track.linear_acceleration.x, track.linear_acceleration.y]], dtype=np.float32)
        acceleration = transform_2d_vectors(acceleration, transform)
        return acceleration[0]

    def produce_features(
            self, scene: Scene, request: PredictionRequest) -> Dict[str, np.ndarray]:
        # The last channel is a binary indicator whether the state is known
        features = np.zeros((self._n_history_steps, self._get_num_channels() + 1))

        to_track_frame_tf = get_to_track_frame_transform(
            get_latest_track_state_by_id(scene, request.track_id))

        for i, ts_ind in enumerate(self._history_indices):
            track = self._get_track_at_ts(scene, request.track_id, ts_ind)
            if track is not None:
                features[i, :self._get_num_channels()] = self._get_features(
                    track, to_track_frame_tf)
                features[i, self._get_num_channels()] = 1

        return {
            'vector_features': features,
        }
