import cv2
import numpy as np

from ysdc_dataset_api.utils import get_track_polygon, transform2dpoints


def _create_feature_map(rows, cols, num_channels, data_format):
    shape = [rows, cols]
    if data_format == 'channels_first':
        shape.insert(0, num_channels)
    elif data_format == 'channels_last':
        shape.append(num_channels)
    else:
        raise ValueError('Unknown data format')
    return np.zeros(shape, dtype=np.float32)


class FeatureMapRendererBase:
    def __init__(self, config, feature_map_params, n_history_steps):
        self._config = config
        self._feature_map_params = feature_map_params
        self._n_history_steps = n_history_steps
        self._num_channels = self._get_num_channels()

    def render(self, scene, transform):
        raise NotImplementedError()

    def _get_num_channels(self):
        raise NotImplementedError()

    @property
    def n_history_steps(self):
        return self._n_history_steps

    @property
    def num_channels(self):
        return self._num_channels

    def _create_feature_map(self):
        return _create_feature_map(
            self._feature_map_params['rows'],
            self._feature_map_params['cols'],
            self._num_channels,
            self._feature_map_params['data_format'],
        )


class VehicleTracksRenderer(FeatureMapRendererBase):
    def _get_num_channels(self):
        num_channels = 0
        if 'occupancy' in self._config:
            num_channels += 1
        if 'velocity' in self._config:
            num_channels += 2
        if 'acceleration' in self._config:
            num_channels += 2
        if 'angular_velocity' in self._config:
            num_channels += 1
        return num_channels

    def render(self, scene, transform):
        feature_map = self._create_feature_map()
        for ts_ind in range(-self.n_history_steps, 0):
            for track in scene.past_vehicle_tracks[ts_ind].tracks:
                track_polygon = transform2dpoints(get_track_polygon(track), transform)
                track_polygon = np.int32(track_polygon).reshape(1, -1, 2)
                for v in self._get_fm_values(track):
                    cv2.fillPoly(feature_map[:, :, 0], track_polygon, v)
        return feature_map

    def _get_fm_values(self, track):
        values = []
        if 'occupancy' in self._config:
            values.append(1.)
        if 'velocity' in self._config:
            values.append(track.linear_velocity.x)
            values.append(track.linear_velocity.y)
        if 'acceleration' in self._config:
            values.append(track.linear_acceleration.x)
            values.append(track.linear_acceleration.y)
        return values


class PedestrianTracksRenderer(FeatureMapRendererBase):
    def _get_num_channels(self):
        num_channels = 0
        if 'occupancy' in self._config:
            num_channels += 1
        if 'velocity' in self._config:
            num_channels += 2
        return num_channels

    def render(self, scene, transform):
        feature_map = self._create_feature_map()
        return feature_map


class FeatureRenderer:
    def __init__(self, config):
        self._fm_params = config['fm_params']

        self._renderers = []
        for group in config['groups']:
            for renderer_config in group['renderers']:
                self._renderers.append(
                    self._create_renderer(renderer_config, self._fm_params, group['n_history_steps']))
        self._num_channels = self._get_num_channels()
        self._fm_shift_scale_transform = self._get_fm_shift_scale_transform()

    def render_features(self, scene, track_to_fm_transform):
        transform = self._fm_shift_scale_transform @ track_to_fm_transform
        fm = self._create_feature_map()
        fm_slice_start = 0
        for renderer in self._renderers:
            fm_slice_end = fm_slice_start + renderer.num_channels * renderer.n_history_steps
            if self._fm_params['data_format'] == 'channels_first':
                fm[fm_slice_start:fm_slice_end, :, :] = renderer.render(scene, transform)
            else:
                fm[:, :, fm_slice_start:fm_slice_end] = renderer.render(scene, transform)
            fm_slice_start = fm_slice_end
        return fm

    def _get_fm_shift_scale_transform(self):
        fm_scale = 1. / self._fm_params['resolution']
        fm_origin_x = 0.5 * self._fm_params['rows']
        fm_origin_y = 0.5 * self._fm_params['cols']
        return np.array([
            [fm_scale, 0, 0, fm_origin_x],
            [0, fm_scale, 0, fm_origin_y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

    def _create_feature_map(self):
        return _create_feature_map(
            self._fm_params['rows'],
            self._fm_params['cols'],
            self._num_channels,
            self._fm_params['data_format'],
        )

    def _get_num_channels(self):
        return sum(
            renderer.num_channels * renderer.n_history_steps
            for renderer in self._renderers
        )

    def _create_renderer(self, config, feature_map_params, n_history_steps):
        if 'vehicles' in config:
            return VehicleTracksRenderer(
                config['vehicles'], feature_map_params, n_history_steps)
        elif 'pedestrians' in config:
            return PedestrianTracksRenderer(
                config['pedestrians'], feature_map_params, n_history_steps)
        else:
            raise NotImplementedError()
