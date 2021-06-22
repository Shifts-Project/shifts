import numpy as np
import pytest

from ..evaluation import (average_displacement_error, avg_ade, avg_fde,
                          evaluate_submission_with_proto,
                          final_displacement_error, get_prediction_horizon,
                          get_trajectories_weights_arrays, min_ade, min_fde,
                          top1_ade, top1_fde, trajectory_array_to_proto,
                          weighted_ade, weighted_fde)
from ..evaluation.metrics import _softmax_normalize
from ..proto import ObjectPrediction, Submission, WeightedTrajectory


def test_shapes():
    n_timestamps = 5
    n_modes = 3
    coord_dim = 2
    gt = np.zeros((n_timestamps, coord_dim))
    pred = np.zeros((n_modes, n_timestamps, coord_dim))

    assert average_displacement_error(gt, pred).shape == (n_modes, )
    assert final_displacement_error(gt, pred).shape == (n_modes, )


def test_metrics():
    n_timestamps = 3
    n_modes = 2
    coord_dim = 1
    gt = np.zeros((n_timestamps, coord_dim))
    pred = np.zeros((n_modes, n_timestamps, coord_dim))
    pred[1] = np.array([[0], [1], [2]])

    assert np.allclose(average_displacement_error(gt, pred), np.array([0, 1]))
    assert np.allclose(final_displacement_error(gt, pred), np.array([0, 2]))

    assert np.allclose(min_ade(gt, pred), 0)
    assert np.allclose(min_fde(gt, pred), 0)
    assert np.allclose(avg_ade(gt, pred), 0.5)
    assert np.allclose(avg_fde(gt, pred), 1.)

    weights = np.array([0.6, 0.4])
    assert np.allclose(top1_ade(gt, pred, weights), 0)
    assert np.allclose(top1_fde(gt, pred, weights), 0)
    assert np.allclose(weighted_ade(gt, pred, weights), 0.22508300134376105)
    assert np.allclose(weighted_fde(gt, pred, weights), 0.4501660026875221)

    weights = np.array([0.4, 0.6])
    assert np.allclose(top1_ade(gt, pred, weights), 1.)
    assert np.allclose(top1_fde(gt, pred, weights), 2.)
    assert np.allclose(weighted_ade(gt, pred, weights), 0.2749169986562389)
    assert np.allclose(weighted_fde(gt, pred, weights), 0.5498339973124778)


def test_evaluate_submission():
    gt_array = np.array([
        [0, 0],
        [1, 1],
    ])

    gt = Submission()
    object_prediction = ObjectPrediction()
    object_prediction.scene_id = 'test_scene_id'
    object_prediction.track_id = 42
    trajectory = WeightedTrajectory(
        trajectory=trajectory_array_to_proto(gt_array),
        weight=1.0
    )
    object_prediction.weighted_trajectories.append(trajectory)
    gt.predictions.append(object_prediction)

    mode1_array = np.zeros_like(gt_array)
    mode1_weight = 0.4
    mode2_array = np.zeros_like(gt_array)
    mode2_array[1, 0] = 1
    mode2_array[1, 1] = 1
    mode2_weight = 0.6

    pred = Submission()
    object_prediction = ObjectPrediction()
    object_prediction.scene_id = 'test_scene_id'
    object_prediction.track_id = 42
    object_prediction.weighted_trajectories.append(
        WeightedTrajectory(
            trajectory=trajectory_array_to_proto(mode1_array),
            weight=mode1_weight,
        )
    )
    object_prediction.weighted_trajectories.append(
        WeightedTrajectory(
            trajectory=trajectory_array_to_proto(mode2_array),
            weight=mode2_weight,
        )
    )
    pred.predictions.append(object_prediction)

    metrics = evaluate_submission_with_proto(pred, gt)

    assert metrics['avg_ade'][0] == pytest.approx(2 ** 0.5 / 2 / 2)
    assert metrics['avg_fde'][0] == pytest.approx(2 ** 0.5 / 2)
    assert metrics['min_ade'][0] == pytest.approx(0.)
    assert metrics['min_fde'][0] == pytest.approx(0.)
    assert metrics['top1_ade'][0] == pytest.approx(0.)
    assert metrics['top1_fde'][0] == pytest.approx(0.)

    normalized_weights = _softmax_normalize(np.array([mode1_weight, mode2_weight]))
    assert metrics['weighted_ade'][0] == pytest.approx(2 ** 0.5 / 2 / 2 * normalized_weights[0])
    assert metrics['weighted_fde'][0] == pytest.approx(2 ** 0.5 / 2 * normalized_weights[0])


def test_trajectory_array_to_proto():
    array = np.arange(4).reshape(2, 2)
    trajectory_proto = trajectory_array_to_proto(array)
    assert trajectory_proto.points[0].x == pytest.approx(0)
    assert trajectory_proto.points[0].y == pytest.approx(1)
    assert trajectory_proto.points[1].x == pytest.approx(2)
    assert trajectory_proto.points[1].y == pytest.approx(3)


def test_get_trajectories_weights_arrays():
    horizon = 7
    mode1 = np.zeros((horizon, 2))
    mode2 = np.ones((horizon, 2))
    mode1_weight = 0.1
    mode2_weight = 0.9

    trajectories = []
    trajectories.append(
        WeightedTrajectory(
            trajectory=trajectory_array_to_proto(mode1),
            weight=mode1_weight))
    trajectories.append(
        WeightedTrajectory(
            trajectory=trajectory_array_to_proto(mode2),
            weight=mode2_weight))

    trajectories_array, weights = get_trajectories_weights_arrays(trajectories)
    assert np.allclose(
        trajectories_array,
        np.concatenate((mode1[np.newaxis, ...], mode2[np.newaxis, ...])),
    )
    assert np.allclose(weights, np.array([mode1_weight, mode2_weight]))


def test_get_prediction_horizon():
    horizon = 7
    mode1 = np.zeros((horizon, 2))
    mode2 = np.ones((horizon, 2))

    trajectories = []
    trajectories.append(
        WeightedTrajectory(
            trajectory=trajectory_array_to_proto(mode1)))
    trajectories.append(
        WeightedTrajectory(
            trajectory=trajectory_array_to_proto(mode2)))
    assert get_prediction_horizon(trajectories) == horizon
