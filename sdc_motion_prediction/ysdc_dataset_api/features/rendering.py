import math

import cv2
import numpy as np

from .producing import FeatureProducerBase
from ..utils import (
    get_track_polygon,
    get_transformed_velocity,
    get_transformed_acceleration,
    transform2dpoints,
)
from ..utils.map import get_crosswalk_availability, get_polygon


MAX_HISTORY_LENGTH = 25


def _create_feature_maps(rows, cols, num_channels):
    shape = [num_channels, rows, cols]
    return np.zeros(shape, dtype=np.float32)


class FeatureMapRendererBase:
    def __init__(
            self,
            config,
            feature_map_params,
            time_grid_params,
            to_feature_map_tf,
    ):
        self._config = config
        self._feature_map_params = feature_map_params
        self._history_indices = self._get_history_indices(time_grid_params)
        self._num_channels = self._get_num_channels()
        self._to_feature_map_tf = to_feature_map_tf

    def render(self, scene, to_track_transform):
        raise NotImplementedError()

    def _get_num_channels(self):
        raise NotImplementedError()

    def _get_history_indices(self, time_grid_params):
        return list(range(
            -time_grid_params['stop'] - 1,
            -time_grid_params['start'],
            time_grid_params['step'],
        ))

    @property
    def n_history_steps(self):
        return len(self._history_indices)

    @property
    def num_channels(self):
        return self._num_channels

    def _create_feature_maps(self):
        return _create_feature_maps(
            self._feature_map_params['rows'],
            self._feature_map_params['cols'],
            self._num_channels * self.n_history_steps,
        )

    def _get_transformed_track_polygon(self, track, transform):
        polygon = transform2dpoints(get_track_polygon(track), transform)
        polygon = np.around(polygon.reshape(1, -1, 2) - 0.5).astype(np.int32)
        return polygon


class VehicleTracksRenderer(FeatureMapRendererBase):
    def _get_num_channels(self):
        num_channels = 0
        if 'occupancy' in self._config:
            num_channels += 1
        if 'velocity' in self._config:
            num_channels += 2
        if 'acceleration' in self._config:
            num_channels += 2
        if 'yaw' in self._config:
            num_channels += 1
        return num_channels

    def render(self, scene, to_track_transform):
        feature_map = self._create_feature_maps()
        transform = self._to_feature_map_tf @ to_track_transform
        for ts_ind in self._history_indices:
            for track in scene.past_vehicle_tracks[ts_ind].tracks:
                track_polygon = self._get_transformed_track_polygon(
                    track, transform)
                for i, v in enumerate(self._get_fm_values(track, to_track_transform)):
                    cv2.fillPoly(
                        feature_map[ts_ind * self.num_channels + i, :, :],
                        track_polygon,
                        v,
                        lineType=cv2.LINE_AA,
                    )
            ego_track = scene.past_ego_track[ts_ind]
            ego_track_polygon = self._get_transformed_track_polygon(ego_track, transform)
            for i, v in enumerate(self._get_fm_values(ego_track, to_track_transform)):
                cv2.fillPoly(
                    feature_map[ts_ind * self.num_channels + i, :, :],
                    ego_track_polygon,
                    v,
                    lineType=cv2.LINE_AA,
                )
        return feature_map

    def _get_fm_values(self, track, to_track_transform):
        values = []
        if 'occupancy' in self._config:
            values.append(1.)
        if 'velocity' in self._config:
            velocity_transformed = get_transformed_velocity(track, to_track_transform)
            values.append(velocity_transformed[0])
            values.append(velocity_transformed[1])
        if 'acceleration' in self._config:
            acceleration_transformed = get_transformed_acceleration(
                track, to_track_transform)
            values.append(acceleration_transformed[0])
            values.append(acceleration_transformed[1])
        if 'yaw' in self._config:
            values.append(track.yaw)
        return values


class PedestrianTracksRenderer(FeatureMapRendererBase):
    def _get_num_channels(self):
        num_channels = 0
        if 'occupancy' in self._config:
            num_channels += 1
        if 'velocity' in self._config:
            num_channels += 2
        return num_channels

    def render(self, scene, to_track_transform):
        feature_map = self._create_feature_maps()
        transform = self._to_feature_map_tf @ to_track_transform
        for ts_ind in self._history_indices:
            for track in scene.past_pedestrian_tracks[ts_ind].tracks:
                track_polygon = self._get_transformed_track_polygon(track, transform)
                for i, v in enumerate(self._get_fm_values(track, to_track_transform)):
                    cv2.fillPoly(
                        feature_map[ts_ind * self.num_channels + i, :, :],
                        track_polygon,
                        v,
                        lineType=cv2.LINE_AA,
                    )
        return feature_map

    def _get_fm_values(self, track, to_track_transform):
        values = []
        if 'occupancy' in self._config:
            values.append(1.)
        if 'velocity' in self._config:
            velocity_transformed = get_transformed_velocity(track, to_track_transform)
            values.append(velocity_transformed[0])
            values.append(velocity_transformed[1])
        return values


class RoadGraphRenderer(FeatureMapRendererBase):
    def render(self, scene, to_track_transform):
        feature_map = self._create_feature_maps()
        transform = self._to_feature_map_tf @ to_track_transform
        path_graph = scene.path_graph
        for channel_ind in range(len(self._history_indices)):
            traffic_light_sections = scene.traffic_lights[self._history_indices[channel_ind]]
            if self._get_crosswalk_feature_map_size() > 0:
                self._render_crosswalks(
                    feature_map[self._get_crosswalk_feature_map_slice(channel_ind), :, :],
                    path_graph,
                    traffic_light_sections,
                    transform,
                )
            if self._get_lane_feature_map_size() > 0:
                self._render_lanes(
                    feature_map[self._get_lanes_feature_map_slice(channel_ind), :, :],
                    path_graph,
                    traffic_light_sections,
                    transform,
                )
            if self._get_road_feature_map_size() > 0:
                self._render_road_polygons(
                    feature_map[self._get_road_polygon_feature_map_slice(channel_ind), :, :],
                    path_graph,
                    transform,
                )
        return feature_map

    def _render_crosswalks(self, feature_map, path_graph, traffic_light_sections, transform):
        for crosswalk in path_graph.crosswalks:
            polygon = get_polygon(crosswalk.geometry)
            polygon = transform2dpoints(polygon, transform)
            polygon = np.around(polygon.reshape(1, -1, 2) - 0.5).astype(np.int32)
            for i, v in enumerate(self._get_crosswalk_feature_map_values(
                    crosswalk, traffic_light_sections)):
                cv2.fillPoly(
                    feature_map[i, :, :],
                    polygon,
                    v,
                    lineType=cv2.LINE_AA,
                )

    def _render_lanes(self, feature_map, path_graph, traffic_light_sections, transform):
        for lane in path_graph.lanes:
            lane_centers = transform2dpoints(
                np.array([[p.x, p.y] for p in lane.centers]),
                transform
            )
            lane_centers = np.around(lane_centers - 0.5).astype(np.int32)
            for i in range(1, len(lane_centers)):
                for channel, value in enumerate(self._get_lane_feature_map_values(
                        lane_centers[i-1], lane_centers[i], lane, traffic_light_sections)):
                    cv2.polylines(
                        feature_map[channel, :, :],
                        [lane_centers],
                        isClosed=False,
                        color=value,
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

    def _render_road_polygons(self, feature_map, path_graph, transform):
        for road_polygon in path_graph.road_polygons:
            polygon = get_polygon(road_polygon.geometry)
            polygon = transform2dpoints(polygon, transform)
            polygon = np.around(polygon.reshape(1, -1, 2) - 0.5).astype(np.int32)
            for i, v in enumerate(self._get_road_polygon_feature_map_values()):
                cv2.fillPoly(
                    feature_map[i, :, :],
                    polygon,
                    v,
                    lineType=cv2.LINE_AA,
                )

    def _get_num_channels(self):
        return (
            self._get_crosswalk_feature_map_size() +
            self._get_lane_feature_map_size() +
            self._get_road_feature_map_size()
        )

    def _get_crosswalk_feature_map_size(self):
        num_channels = 0
        if 'crosswalk_occupancy' in self._config:
            num_channels += 1
        if 'crosswalk_availability' in self._config:
            num_channels += 1
        return num_channels

    def _get_crosswalk_feature_map_slice(self, ts_ind):
        return slice(
            ts_ind * self.num_channels,
            ts_ind * self.num_channels + self._get_crosswalk_feature_map_size()
        )

    def _get_crosswalk_feature_map_values(self, crosswalk, traffic_light_sections):
        values = []
        if 'crosswalk_occupancy' in self._config:
            values.append(1.)
        if 'crosswalk_avalability' in self._config:
            values.append(get_crosswalk_availability(crosswalk, traffic_light_sections))
        return values

    def _get_lane_feature_map_size(self):
        num_channels = 0
        if 'lane_availability' in self._config:
            raise NotImplementedError()
            num_channels += 1
        if 'lane_direction' in self._config:
            num_channels += 1
        if 'lane_occupancy' in self._config:
            num_channels += 1
        if 'lane_priority' in self._config:
            num_channels += 1
        if 'lane_speed_limit' in self._config:
            num_channels += 1
        return num_channels

    def _get_lanes_feature_map_slice(self, ts_ind):
        offset = (
            ts_ind * self._num_channels +
            self._get_crosswalk_feature_map_size()
        )
        return slice(offset, offset + self._get_lane_feature_map_size())

    def _get_lane_feature_map_values(
            self, segment_start, segment_end, lane, traffic_light_sections):
        values = []
        if 'lane_availability' in self._config:
            raise NotImplementedError()
        if 'lane_direction' in self._config:
            values.append(
                math.atan2(segment_end[1] - segment_start[1], segment_end[0] - segment_start[0])
            )
        if 'lane_occupancy' in self._config:
            values.append(1.0)
        if 'lane_priority' in self._config:
            values.append(float(lane.gives_way_to_some_lane))
        if 'lane_speed_limit' in self._config:
            values.append(lane.max_velocity / 15.0)
        return values

    def _get_road_feature_map_size(self):
        num_channels = 0
        if 'road_polygons' in self._config:
            num_channels += 1
        return num_channels

    def _get_road_polygon_feature_map_slice(self, ts_ind):
        offset = (
            ts_ind * self._num_channels +
            self._get_crosswalk_feature_map_size() +
            self._get_lane_feature_map_size()
        )
        return slice(offset, offset + self._get_road_feature_map_size())

    def _get_road_polygon_feature_map_values(self):
        values = []
        if 'road_polygons' in self._config:
            values.append(1.0)
        return values


class FeatureRenderer(FeatureProducerBase):
    def __init__(self, config):
        self._feature_map_params = config['feature_map_params']
        self._to_feature_map_tf = self._get_to_feature_map_transform()

        self._renderers = self._create_renderers_list(config)
        self._num_channels = self._get_num_channels()

    def produce_features(self, scene, to_track_frame_tf):
        feature_maps = self._create_feature_maps()
        slice_start = 0
        for renderer in self._renderers:
            slice_end = slice_start + renderer.num_channels * renderer.n_history_steps
            feature_maps[slice_start:slice_end, :, :] = renderer.render(scene, to_track_frame_tf)
            slice_start = slice_end
        return {
            'feature_maps': feature_maps,
        }

    @property
    def to_feature_map_tf(self):
        return self._to_feature_map_tf

    def _get_to_feature_map_transform(self):
        fm_scale = 1. / self._feature_map_params['resolution']
        fm_origin_x = 0.5 * self._feature_map_params['rows']
        fm_origin_y = 0.5 * self._feature_map_params['cols']
        return np.array([
            [fm_scale, 0, 0, fm_origin_x],
            [0, fm_scale, 0, fm_origin_y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

    def _create_feature_maps(self):
        return _create_feature_maps(
            self._feature_map_params['rows'],
            self._feature_map_params['cols'],
            self._num_channels,
        )

    def _get_num_channels(self):
        return sum(
            renderer.num_channels * renderer.n_history_steps
            for renderer in self._renderers
        )

    def _create_renderers_list(self, config):
        renderers = []
        for group in config['renderers_groups']:
            for renderer_config in group['renderers']:
                time_grid_params = self._validate_time_grid(group['time_grid_params'])
                renderers.append(
                    self._create_renderer(
                        renderer_config,
                        self._feature_map_params,
                        time_grid_params,
                        self._to_feature_map_tf,
                    )
                )
        return renderers

    def _create_renderer(
            self,
            config,
            feature_map_params,
            n_history_steps,
            to_feature_map_tf,
    ):
        if 'vehicles' in config:
            return VehicleTracksRenderer(
                config['vehicles'],
                feature_map_params,
                n_history_steps,
                to_feature_map_tf,
            )
        elif 'pedestrians' in config:
            return PedestrianTracksRenderer(
                config['pedestrians'],
                feature_map_params,
                n_history_steps,
                to_feature_map_tf
            )
        elif 'road_graph' in config:
            return RoadGraphRenderer(
                config['road_graph'],
                feature_map_params,
                n_history_steps,
                to_feature_map_tf,
            )
        else:
            raise NotImplementedError()

    @staticmethod
    def _validate_time_grid(time_grid_params):
        if time_grid_params['start'] < 0:
            raise ValueError('"start" value must be non-negative.')
        if time_grid_params['stop'] < 0:
            raise ValueError('"stop" value must be non-negative')
        if time_grid_params['start'] > time_grid_params['stop']:
            raise ValueError('"start" must be less or equal to "stop"')
        if time_grid_params['stop'] + 1 > MAX_HISTORY_LENGTH:
            raise ValueError(
                'Maximum history size is 25. Consider setting "stop" to 24 or less')
        return time_grid_params
