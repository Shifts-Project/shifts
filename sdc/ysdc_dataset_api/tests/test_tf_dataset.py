from ..dataset import TFMotionPredictionDataset


def test_dataset(dataset_path, renderer, vectorizer):
    dataset = TFMotionPredictionDataset(
        dataset_path,
        feature_producers=[renderer, vectorizer])
    for data_item in dataset.get_tf_dataset():
        assert 'ground_truth_trajectory' in data_item
        assert 'vector_features' in data_item
        assert 'feature_maps' in data_item
