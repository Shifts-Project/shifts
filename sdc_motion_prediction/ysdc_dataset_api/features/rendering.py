import math
from collections import defaultdict

import cv2
import numpy as np

from .producing import FeatureProducerBase
from ..utils import (
    get_track_polygon,
    get_tracks_polygons,
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

    def render(self, feature_map, scene, to_track_transform):
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
        polygon = get_track_polygon(track)
        polygon = transform2dpoints(polygon, transform)
        polygon = np.around(polygon.reshape(1, -1, 2) - 0.5).astype(np.int32)
        return polygon


class TrackRendererBase(FeatureMapRendererBase):
    def _get_tracks_at_timestamp(self, scene, ts_ind):
        raise NotImplementedError

    def render(self, feature_map, scene, to_track_transform):
        transform = self._to_feature_map_tf @ to_track_transform
        for ts_ind in self._history_indices:
            tracks_at_frame = self._get_tracks_at_timestamp(scene, ts_ind)
            if not tracks_at_frame:
                continue
            polygons = get_tracks_polygons(tracks_at_frame)
            polygons = transform2dpoints(polygons.reshape(-1, 2), transform).reshape(-1, 4, 2)
            polygons = np.around(polygons - 0.5).astype(np.int32)

            for i, track in enumerate(tracks_at_frame):
                for j, v in enumerate(self._get_fm_values(track, to_track_transform)):
                    cv2.fillPoly(
                        feature_map[ts_ind * self.num_channels + j, :, :],
                        [polygons[i]],
                        v,
                        lineType=cv2.LINE_AA,
                    )
        return feature_map


class VehicleTracksRenderer(TrackRendererBase):
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

    def _get_tracks_at_timestamp(self, scene, ts_ind):
        return [
            track for track in scene.past_vehicle_tracks[ts_ind].tracks
        ] + [scene.past_ego_track[ts_ind]]

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


class PedestrianTracksRenderer(TrackRendererBase):
    def _get_num_channels(self):
        num_channels = 0
        if 'occupancy' in self._config:
            num_channels += 1
        if 'velocity' in self._config:
            num_channels += 2
        return num_channels

    def _get_tracks_at_timestamp(self, scene, ts_ind):
        return [
            track for track in scene.past_pedestrian_tracks[ts_ind].tracks
        ]

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
    LINE_TYPE = cv2.LINE_AA
    LINE_THICKNESS = 1

    def render(self, feature_map, scene, to_track_transform):
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
                    lineType=self.LINE_TYPE,
                )

    def _render_lanes(self, feature_map, path_graph, traffic_light_sections, transform):
        lane_lengths = []
        lane_centers_concatenated = []
        for lane in path_graph.lanes:
            lane_lengths.append(len(lane.centers))
            for p in lane.centers:
                lane_centers_concatenated.append([p.x, p.y])
        lane_centers_concatenated = np.array(lane_centers_concatenated, dtype=np.float32)
        lane_centers_concatenated = transform2dpoints(lane_centers_concatenated, transform)
        lane_centers_concatenated = np.around(lane_centers_concatenated - 0.5).astype(np.int32)

        lane_centers = []
        bounds = [0] + np.cumsum(lane_lengths).tolist()

        for i in range(1, len(bounds)):
            lane_centers.append(lane_centers_concatenated[bounds[i - 1]:bounds[i]])

        channel = 0
        if 'lane_availability' in self._config:
            raise NotImplementedError
            channel += 1
        if 'lane_direction' in self._config:
            self._render_lane_direction(feature_map[channel, ...], lane_centers)
            channel += 1
        if 'lane_occupancy' in self._config:
            self._render_lane_occupancy(feature_map[channel, ...], lane_centers)
            channel += 1
        if 'lane_priority' in self._config:
            self._render_lane_priority(feature_map[channel, ...], lane_centers, path_graph)
            channel += 1
        if 'lane_speed_limit' in self._config:
            self._render_lane_speed_limit(feature_map[channel, ...], lane_centers, path_graph)
            channel += 1

    def _render_lane_direction(self, feature_map, lane_centers):
        for lane in lane_centers:
            for i in range(1, lane.shape[0]):
                p1 = (lane[i - 1, 0], lane[i - 1, 1])
                p2 = (lane[i, 0], lane[i, 1])
                cv2.line(
                    feature_map,
                    p1,
                    p2,
                    math.atan2(p2[1] - p1[1], p2[0] - p1[0]),
                    thickness=self.LINE_THICKNESS,
                    lineType=self.LINE_TYPE,
                )

    def _render_lane_occupancy(self, feature_map, lane_centers):
        cv2.polylines(
            feature_map,
            lane_centers,
            isClosed=False,
            color=1.,
            thickness=self.LINE_THICKNESS,
            lineType=self.LINE_TYPE,
        )

    def _render_lane_priority(self, feature_map, lane_centers, path_graph):
        non_priority_lanes = []
        for i, lane in enumerate(path_graph.lanes):
            if lane.gives_way_to_some_lane:
                non_priority_lanes.append(lane_centers[i])
        cv2.polylines(
            feature_map,
            non_priority_lanes,
            isClosed=False,
            color=1.,
            thickness=self.LINE_THICKNESS,
            lineType=self.LINE_TYPE,
        )

    def _render_lane_speed_limit(self, feature_map, lane_centers, path_graph):
        limit_to_lanes = defaultdict(list)
        for i, lane in enumerate(path_graph.lanes):
            limit_to_lanes[lane.max_velocity].append(lane_centers[i])
        for limit, lanes in limit_to_lanes.items():
            cv2.polylines(
                feature_map,
                lanes,
                isClosed=False,
                color=limit / 15.0,
                thickness=self.LINE_THICKNESS,
                lineType=self.LINE_TYPE,
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
                    lineType=self.LINE_TYPE,
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
            renderer.render(feature_maps[slice_start:slice_end, :, :], scene, to_track_frame_tf)
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
        ], dtype=np.float32)

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
