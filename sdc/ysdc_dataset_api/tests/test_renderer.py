import pathlib

from ..features import FeatureRenderer
from ..utils import read_scene_from_file


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
        feature_maps = renderer.produce_features(scene, request)
        assert feature_maps
        assert feature_maps['feature_maps'].shape == (26, 400, 400)
