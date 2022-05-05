from ..features import FeatureVectorizer
from ..utils import read_scene_from_file


def test_produce_vector_features(scene_file_path, vectorizer_config):
    scene = read_scene_from_file(scene_file_path)
    vectorizer = FeatureVectorizer(vectorizer_config)
    for request in scene.prediction_requests:
        vector_features = vectorizer.produce_features(scene, request)
        assert vector_features
        assert vector_features['vector_features'].shape == (2, 8)
