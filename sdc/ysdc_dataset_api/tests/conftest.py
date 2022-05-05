import pathlib
import pytest

from ..features import FeatureRenderer, FeatureVectorizer


@pytest.fixture
def vectorizer_config():
    return {
        'time_grid_params': {
            'start': 0,
            'stop': 2,
            'step': 2,
        },
        'features': ['position', 'velocity', 'acceleration', 'yaw']
    }


@pytest.fixture
def renderer_config():
    return {
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


@pytest.fixture
def vectorizer(vectorizer_config):
    return FeatureVectorizer(vectorizer_config)


@pytest.fixture
def renderer(renderer_config):
    return FeatureRenderer(renderer_config)


TEST_SCENE_FNAME = 'test_scene.pb'


@pytest.fixture
def dataset_path():
    return pathlib.Path(__file__).parent.absolute() / 'dataset'


@pytest.fixture
def scene_file_path(dataset_path):
    return dataset_path / '0' / TEST_SCENE_FNAME
