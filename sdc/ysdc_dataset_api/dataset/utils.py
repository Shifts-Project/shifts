import os
from typing import Callable, List

from sdc.constants import SCENE_TAG_TYPE_TO_OPTIONS, VALID_TRAJECTORY_TAGS
from ..features import FeatureProducerBase
from ..proto import get_tags_from_request, proto_to_dict
from ..utils import (
    get_gt_trajectory,
    get_latest_track_state_by_id,
    get_to_track_frame_transform,
    read_feature_map_from_file,
    request_is_valid,
    scenes_generator,
    transform_2d_points,
)


def data_generator(
        file_paths: List[str],
        feature_producers: List[FeatureProducerBase],
        prerendered_dataset_path: str,
        transform_ground_truth_to_agent_frame: bool,
        trajectory_tags_filter: Callable,
        yield_metadata: bool,
        yield_scene_tags: bool
):
    for scene, fpath in scenes_generator(file_paths, yield_fpath=True):
        for request in scene.prediction_requests:
            if not request_is_valid(scene, request):
                continue
            trajectory_tags = get_tags_from_request(request)
            if not trajectory_tags_filter(trajectory_tags):
                continue
            track = get_latest_track_state_by_id(scene, request.track_id)
            to_track_frame_tf = get_to_track_frame_transform(track)

            result = {
                'scene_id': scene.id,
                'track_id': request.track_id,
            }

            if yield_scene_tags:
                result['scene_tags'] = proto_to_dict(scene.scene_tags)

            ground_truth_trajectory = get_gt_trajectory(scene, request.track_id)
            if ground_truth_trajectory.shape[0] > 0:
                if transform_ground_truth_to_agent_frame:
                    ground_truth_trajectory = transform_2d_points(
                        ground_truth_trajectory, to_track_frame_tf)
                result['ground_truth_trajectory'] = ground_truth_trajectory

            if prerendered_dataset_path:
                fm_path = _get_serialized_fm_path(
                    prerendered_dataset_path, fpath, scene.id, request.track_id)
                result['prerendered_feature_map'] = read_feature_map_from_file(fm_path)

            for producer in feature_producers:
                result.update(producer.produce_features(scene, request))

            if yield_metadata:
                result = _add_metadata_to_batch(
                    scene=scene, request=request,
                    trajectory_tags=trajectory_tags,
                    batch=result)

            yield result


def _get_serialized_fm_path(prerendered_dataset_path, scene_fpath, scene_id, track_id):
    base, _ = os.path.split(scene_fpath)
    _, subdir = os.path.split(base)
    return os.path.join(prerendered_dataset_path, subdir, f'{scene_id}_{track_id}.npy')


def _add_metadata_to_batch(scene, request, trajectory_tags, batch):
    batch['scene_id'] = scene.id
    batch['request_id'] = request.track_id

    # Note that some will be "invalid"
    batch['num_vehicles'] = len(scene.prediction_requests)

    scene_tags_dict = proto_to_dict(scene.scene_tags)
    for scene_tag_type in SCENE_TAG_TYPE_TO_OPTIONS.keys():
        scene_tag_options = SCENE_TAG_TYPE_TO_OPTIONS[scene_tag_type]

        for scene_tag_option in scene_tag_options:
            try:
                batch[f'{scene_tag_type}__{scene_tag_option}'] = int(
                    scene_tags_dict[scene_tag_type] == scene_tag_option)
            except KeyError:
                batch[f'{scene_tag_type}__{scene_tag_option}'] = -1

    trajectory_tags = set(trajectory_tags)
    for trajectory_tag in VALID_TRAJECTORY_TAGS:
        batch[trajectory_tag] = (trajectory_tag in trajectory_tags)

    return batch


def callable_or_trivial_filter(f):
    if f is None:
        return _trivial_filter
    if not callable(f):
        raise ValueError('Expected callable, got {}'.format(type(f)))
    return f


def _trivial_filter(x):
    return True
