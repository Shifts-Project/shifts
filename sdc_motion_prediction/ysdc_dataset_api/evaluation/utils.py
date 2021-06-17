from collections import defaultdict
from typing import Dict, Sequence, Tuple

import numpy as np

from ..dataset import MotionPredictionDataset
from ..proto import (ObjectPrediction, Submission, Trajectory, Vector3,
                     WeightedTrajectory)
from ..utils.map import repeated_points_to_array
from .metrics import (avg_ade, avg_fde, min_ade, min_fde, top1_ade, top1_fde,
                      weighted_ade, weighted_fde)


def save_submission_proto(filepath: str, submission: Submission) -> None:
    """Save serialized submission protobuf to file.

    Args:
        filepath (str): Path to output file.
        submission (Submission): Submission proto to save.
    """
    with open(filepath, 'wb') as fout:
        fout.write(submission.SerializeToString())


def load_submission_proto(filepath: str) -> Submission:
    """Load and deserialized submission proto from file.

    Args:
        filepath (str): File with serialized protobuf message.

    Returns:
        Submission: Deserialized message.
    """
    with open(filepath, 'rb') as fin:
        serialized = fin.read()
    submission = Submission()
    submission.ParseFromString(serialized)
    return submission


def evaluate_submission_with_proto(
        submission: Submission,
        ground_truth: Submission,
) -> Dict[str, float]:
    """Calculates various motion prediction metrics given
    the submission and ground truth protobuf messages.

    Args:
        submission (Submission): Proto message with predicted trajectories.
        ground_truth (Submission): Proto message with ground truth trajectories.

    Raises:
        ValueError:
            Number of objects in submission is not equal to number of objects in ground truth.
        ValueError:
            Objects order in submission violates objects order in ground truth.

    Returns:
        Dict[str, float]: Mapping from metric name to its aggregated value.
    """
    if len(submission.predictions) != len(ground_truth.predictions):
        raise ValueError(f'Check number of submitted predictions: \
            {len(submission.predictions)} != {len(ground_truth.predictions)}')
    metrics = defaultdict(list)
    for i in range(len(submission.predictions)):
        pred = submission.predictions[i]
        gt = ground_truth.predictions[i]
        if pred.scene_id != gt.scene_id:
            raise ValueError(f'Check scenes order: {pred.scene_id} != {gt.scene_id}')
        if pred.track_id != gt.track_id:
            raise ValueError(f'Check objects order: {pred.track_id} != {gt.track_id}')
        pred_trajectories, weights = get_trajectories_weights_arrays(pred.weighted_trajectories)
        gt_trajectory, _ = get_trajectories_weights_arrays(gt.weighted_trajectories)
        metrics['avg_ade'].append(avg_ade(gt_trajectory, pred_trajectories))
        metrics['avg_fde'].append(avg_fde(gt_trajectory, pred_trajectories))
        metrics['min_ade'].append(min_ade(gt_trajectory, pred_trajectories))
        metrics['min_fde'].append(min_fde(gt_trajectory, pred_trajectories))
        metrics['top1_ade'].append(top1_ade(gt_trajectory, pred_trajectories, weights))
        metrics['top1_fde'].append(top1_fde(gt_trajectory, pred_trajectories, weights))
        metrics['weighted_ade'].append(weighted_ade(gt_trajectory, pred_trajectories, weights))
        metrics['weighted_fde'].append(weighted_fde(gt_trajectory, pred_trajectories, weights))
    return metrics


def get_trajectories_weights_arrays(
        trajectories: Sequence[WeightedTrajectory],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return numpy array of trajectories and respective weights
    given the sequence of WeightedTrajectory protobuf messages.

    Args:
        trajectories (Sequence[WeightedTrajectory]): [description]

    Returns:
        Tuple[np.ndarray, np.ndarray]: trajectories of shape (n_modes, prediction_horizon, 2) and
            respective weights of shape (n_modes,)
    """
    n_modes = len(trajectories)
    prediction_horizon = get_prediction_horizon(trajectories)
    trajectories_array = np.empty((n_modes, prediction_horizon, 2))
    weights = np.empty(n_modes)
    for i, weighted_trajectory in enumerate(trajectories):
        trajectories_array[i] = repeated_points_to_array(weighted_trajectory.trajectory)
        weights[i] = weighted_trajectory.weight
    return trajectories_array, weights


def ground_truth_from_dataset(dataset: MotionPredictionDataset) -> Submission:
    """Generates a Submission protobuf instance with ground truth trajectories.

    Args:
        dataset (MotionPredictionDataset): Dataset to get trajectories from.

    Returns:
        Submission: Resulting protobuf message.
    """
    dataset_iter = iter(dataset)
    ground_truth = Submission()
    for data_item in dataset_iter:
        pred = ObjectPrediction()
        pred.track_id = data_item['track_id']
        pred.scene_id = data_item['scene_id']
        pred.weighted_trajectories.append(WeightedTrajectory(
            trajectory=trajectory_array_to_proto(data_item['ground_truth_trajectory']),
            weight=1.0,
        ))
        ground_truth.predictions.append(pred)
    return ground_truth


def trajectory_array_to_proto(trajectory: np.ndarray) -> Trajectory:
    """Transforms a numpy array with 2D trajectory to Trajectory proto message.

    Args:
        trajectory (np.ndarray): Trajectory array, shape (N, 2)

    Returns:
        Trajectory: Resulting protobuf messsage.
    """
    assert len(trajectory.shape) == 2
    trajectory_proto = Trajectory()
    for i in range(trajectory.shape[0]):
        trajectory_proto.points.append(Vector3(x=trajectory[i, 0], y=trajectory[i, 1]))
    return trajectory_proto


def get_prediction_horizon(trajectories: Sequence[WeightedTrajectory]) -> int:
    """Returns

    Args:
        trajectories (Sequence[WeightedTrajectory]): [description]

    Raises:
        ValueError: [description]

    Returns:
        int: [description]
    """
    horizon = len(trajectories[0].trajectory.points)
    if not all(len(w.trajectory.points) == horizon for w in trajectories):
        raise ValueError('All modes must have the same prediction horizon')
    return horizon
