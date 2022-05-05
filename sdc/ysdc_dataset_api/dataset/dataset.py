import json
from typing import Callable, Union
from typing import Optional, List

from .utils import callable_or_trivial_filter
from ..features import FeatureProducerBase
from ..utils import get_file_paths


class DatasetBase:
    def __init__(
            self,
            dataset_path: str,
            scene_tags_fpath: str = None,
            feature_producers: List[FeatureProducerBase] = None,
            prerendered_dataset_path: str = None,
            transform_ground_truth_to_agent_frame: bool = True,
            scene_tags_filter: Union[Callable, None] = None,
            trajectory_tags_filter: Union[Callable, None] = None,
            pre_filtered_scene_file_paths: Optional[List[str]] = None,
            yield_metadata=False
    ):
        """Base dataset class for the motion prediction task.

        Dataset iterator performs iteration over scenes in the dataset and individual prediction
        requests in each scene. Iterator yields dict that can have the following structure:
        {
            'scene_id': str,
            'track_id': int,
            'scene_tags': Dict[str, str],
            'ground_truth_trajectory': np.ndarray,
            'prerendered_feature_map': np.ndarray,
            'feature_maps': np.ndarray,
        }.
        'scene_id' unique scene identifier.
        'track_id' vehicle id of the current prediction request.
        'ground_truth_trajectory' the field contains ground truth trajectory for
        the current prediction request.
        'prerendered_feature_map' field would be present if prerendered_dataset_path was specified,
        contains pre-rendered feature maps.
        'feature_maps' field would be present if user passes an instance of
        ysdc_dataset_api.features.FeatureRenderer, contains feature maps rendered on the fly by
        specified renderer instance.

        Args:
            dataset_path: path to the dataset directory
            scene_tags_fpath: path to the tags file
            feature_producers: a list of instances of the FeatureProducerBase class,
                used to generate features for a data item. Defaults to None.
            prerendered_dataset_path: path to the pre-rendered dataset. Defaults to None.
            transform_ground_truth_to_agent_frame: whether to transform ground truth
                trajectory to an agent coordinate system or return global coordinates.
                Defaults to True.
            scene_tags_filter: function to filter dataset scenes by tags. Defaults to None.
            trajectory_tags_filter: function to filter prediction requests by trajectory tags.
                Defaults to None.

        Raises:
            ValueError: if none of feature_producer or prerendered_dataset_path was specified.
        """
        self._feature_producers = feature_producers or []
        self._prerendered_dataset_path = prerendered_dataset_path
        self._transform_ground_truth_to_agent_frame = transform_ground_truth_to_agent_frame

        self._scene_tags_filter = callable_or_trivial_filter(scene_tags_filter)
        self._trajectory_tags_filter = callable_or_trivial_filter(trajectory_tags_filter)
        self._yield_metadata = yield_metadata

        if pre_filtered_scene_file_paths is not None:
            print('Building MotionPredictionDataset with pre-filtered '
                  'scene file paths.')
            self._scene_file_paths = pre_filtered_scene_file_paths
        else:
            self._scene_file_paths = get_file_paths(dataset_path)
            if scene_tags_fpath is not None:
                self._scene_file_paths = self._filter_paths(
                    self._scene_file_paths, scene_tags_fpath)

    @property
    def num_scenes(self) -> int:
        """Number of scenes in the dataset"""
        return len(self._scene_file_paths)

    def _filter_paths(self, file_paths, scene_tags_fpath):
        valid_indices = []
        with open(scene_tags_fpath, 'r') as f:
            for i, line in enumerate(f):
                tags = json.loads(line.strip())
                if self._scene_tags_filter(tags):
                    valid_indices.append(i)

        print(
            f'{len(valid_indices)}/{len(file_paths)} '
            f'scenes fit the filter criteria.')
        return [file_paths[i] for i in valid_indices]
