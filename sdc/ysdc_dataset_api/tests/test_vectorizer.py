import pathlib

from ..features import FeatureVectorizer
from ..utils import read_scene_from_file


TEST_CONFIG = {
    'time_grid_params': {
        'start': 0,
        'stop': 2,
        'step': 2,
    },
    'features': ['position', 'velocity', 'acceleration', 'yaw']
}

TEST_SCENE_FNAME = 'test_scene.pb'


def get_test_scene_path():
    return pathlib.Path(__file__).parent.absolute() / TEST_SCENE_FNAME


def test_produce_vector_features():
    scene = read_scene_from_file(get_test_scene_path())
    vectorizer = FeatureVectorizer(TEST_CONFIG)
    for request in scene.prediction_requests:
        vector_features = vectorizer.produce_features(scene, request)
        assert vector_features
        assert vector_features['vector_features'].shape == (2, 8)
