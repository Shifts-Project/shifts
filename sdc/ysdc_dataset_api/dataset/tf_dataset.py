import tensorflow as tf

from .dataset import DatasetBase
from .utils import data_generator


class TFMotionPredictionDataset(DatasetBase):
    def __init__(self, *args, **kwargs):
        super(TFMotionPredictionDataset, self).__init__(*args, **kwargs)

    def _get_output_signature(self):
        signature = {
            'ground_truth_trajectory': tf.TensorSpec(shape=(25, 2), dtype=tf.float32),
            'scene_id': tf.TensorSpec(shape=(), dtype=tf.string),
            'track_id': tf.TensorSpec(shape=(), dtype=tf.int64),
        }
        if self._prerendered_dataset_path:
            signature['prerendered_feature_map'] = tf.TensorSpec((17, 128, 128), dtype=tf.float32)
        for producer in self._feature_producers:
            signature.update(producer.get_tf_signature())
        return signature

    def get_tf_dataset(self):
        return tf.data.Dataset.from_generator(
            lambda: data_generator(
                self._scene_file_paths,
                self._feature_producers,
                self._prerendered_dataset_path,
                self._transform_ground_truth_to_agent_frame,
                self._trajectory_tags_filter,
                self._yield_metadata,
                yield_scene_tags=False),
            output_signature=self._get_output_signature(),
        )
