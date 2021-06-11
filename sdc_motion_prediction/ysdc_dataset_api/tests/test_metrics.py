from ysdc_dataset_api.evaluation.metrics import average_displacement_error, final_displacement_error
import numpy as np

from ..evaluation import (
    avg_ade, avg_fde, min_ade, min_fde, top1_ade, top1_fde, weighted_ade, weighted_fde,
)


def test_shapes():
    n_timestamps = 5
    n_modes = 3
    coord_dim = 2
    gt = np.zeros((n_timestamps, coord_dim))
    pred = np.zeros((n_modes, n_timestamps, coord_dim))

    assert average_displacement_error(gt, pred).shape == (n_modes, )
    assert final_displacement_error(gt, pred).shape == (n_modes, )


def test_ade():
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
    assert np.allclose(weighted_ade(gt, pred, weights), 0.2)
    assert np.allclose(weighted_fde(gt, pred, weights), 0.4)

    weights = np.array([0.4, 0.6])
    assert np.allclose(top1_ade(gt, pred, weights), 1.)
    assert np.allclose(top1_fde(gt, pred, weights), 2.)
    assert np.allclose(weighted_ade(gt, pred, weights), 0.3)
    assert np.allclose(weighted_fde(gt, pred, weights), 0.6)
