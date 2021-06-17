import pathlib

from ..features import FeatureRenderer
from ..utils import get_to_track_frame_transform, get_latest_track_state_by_id, read_scene_from_file


TEST_CONFIG = {
    'feature_map_params': {
        'rows': 400,
        'cols': 400,
        'resolution': 0.25,
    },
    'renderers_groups': [
        {
            'time_grid_params': {
                'start': 0,
                'stop': 1,
                'step': 1,
            },
            'renderers': [
                {'vehicles': ['occupancy', 'velocity', 'acceleration', 'yaw']},
                {'pedestrians': ['occupancy', 'velocity']},
            ]
        },
        {
            'time_grid_params': {
                'start': 0,
                'stop': 0,
                'step': 1,
            },
            'renderers': [
                {
                    'road_graph': [
                        'crosswalk_availability',
                        'crosswalk_occupancy',
                        'lane_availability',
                        'lane_direction',
                        'lane_occupancy',
                        'lane_priority',
                        'lane_speed_limit',
                        'road_polygons',
                    ]
                }
            ]
        }
    ]
}


TEST_SCENE_FNAME = 'test_scene.pb'


def get_test_scene_path():
    return pathlib.Path(__file__).parent.absolute() / TEST_SCENE_FNAME


def test_produce_features():
    scene = read_scene_from_file(get_test_scene_path())
    renderer = FeatureRenderer(TEST_CONFIG)
    for request in scene.prediction_requests:
        track = get_latest_track_state_by_id(scene, request.track_id)
        to_track_frame_tf = get_to_track_frame_transform(track)
        feature_maps = renderer.produce_features(scene, to_track_frame_tf)
        assert feature_maps
