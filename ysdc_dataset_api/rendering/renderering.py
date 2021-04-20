import numpy as np


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
    def __init__(self, spec, feature_map_params, n_history_steps):
        self._spec = spec
        self._feature_map_params = feature_map_params
        self._n_history_steps = n_history_steps

    def render(self, scene, transform):
        raise NotImplementedError()

    def get_num_channels(self):
        raise NotImplementedError()

    def _create_feature_map(self):
        return _create_feature_map(
            self._feature_map_params['rows'],
            self._feature_map_params['cols'],
            self.get_num_channels(),
            self._feature_map_params['data_format'],
        )


class VehicleTracksRenderer(FeatureMapRendererBase):
    def get_num_channels(self):
        num_channels = 0
        if 'occupancy' in self._spec:
            num_channels += 1
        if 'velocity' in self._spec:
            num_channels += 2
        if 'acceleration' in self._spec:
            num_channels += 2
        if 'angular_velocity' in self._spec:
            num_channels += 1
        return num_channels * self._n_history_steps

    def render(self, scene, transform):
        feature_map = self._create_feature_map()
        return feature_map


class PedestrianTracksRenderer(FeatureMapRendererBase):
    def get_num_channels(self):
        num_channels = 0
        if 'occupancy' in self._spec:
            num_channels += 1
        if 'velocity' in self._spec:
            num_channels += 2
        return num_channels * self._n_history_steps

    def render(self, scene, transform):
        feature_map = self._create_feature_map()
        return feature_map


class FeatureRenderer:
    def __init__(self, spec):
        self._fm_params = spec['fm_params']

        self._renderers = []
        for group in spec['groups']:
            for renderer_spec in group['renderers']:
                self._renderers.append(
                    self._create_renderer(renderer_spec, self._fm_params, group['n_history_steps']))
        self._num_channels = self._get_num_channels()

    def render_features(self, scene, transform):
        fm = self._create_feature_map()
        fm_slice_start = 0
        for renderer in self._renderers:
            fm_slice_end = fm_slice_start + renderer.get_num_channels()
            if self._fm_params['data_format'] == 'channels_first':
                fm[fm_slice_start:fm_slice_end, :, :] = renderer.render(scene, transform)
            else:
                fm[:, :, fm_slice_start:fm_slice_end] = renderer.render(scene, transform)
            fm_slice_start = fm_slice_end
        return fm

    def _create_feature_map(self):
        return _create_feature_map(
            self._fm_params['rows'],
            self._fm_params['cols'],
            self._num_channels,
            self._fm_params['data_format'],
        )

    def _get_num_channels(self):
        return sum(renderer.get_num_channels() for renderer in self._renderers)

    def _create_renderer(self, spec, feature_map_params, n_history_steps):
        if 'vehicles' in spec:
            return VehicleTracksRenderer(
                spec['vehicles'], feature_map_params, n_history_steps)
        elif 'pedestrians' in spec:
            return PedestrianTracksRenderer(
                spec['pedestrians'], feature_map_params, n_history_steps)
        else:
            raise NotImplementedError()
