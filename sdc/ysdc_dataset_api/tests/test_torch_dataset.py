from ..dataset import MotionPredictionDataset


def test_dataset(dataset_path, renderer, vectorizer):
    dataset = MotionPredictionDataset(
        dataset_path,
        feature_producers=[renderer, vectorizer])
    for data_item in dataset:
        assert 'ground_truth_trajectory' in data_item
        assert 'vector_features' in data_item
        assert 'feature_maps' in data_item
