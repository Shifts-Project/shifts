from ..features import FeatureRenderer
from ..utils import read_scene_from_file


def test_produce_features(scene_file_path, renderer_config):
    scene = read_scene_from_file(scene_file_path)
    renderer = FeatureRenderer(renderer_config)
    for request in scene.prediction_requests:
        feature_maps = renderer.produce_features(scene, request)
        assert feature_maps
        assert feature_maps['feature_maps'].shape == (26, 400, 400)
