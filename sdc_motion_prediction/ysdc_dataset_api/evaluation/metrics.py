import numpy as np


def average_displacement_error(ground_truth, predicted):
    """Calculates average displacement error.
    Does not perform any mode aggregation.

    Args:
        ground_truth (np.ndarray): array of shape (n_timestamps, 2)
        predicted (np.ndarray): array of shape (n_modes, n_timestamps, 2)

        Returns:
        np.ndarray: array of shape (n_modes,)
    """
    return np.linalg.norm(ground_truth - predicted, axis=-1).mean(axis=-1)


def final_displacement_error(ground_truth, predicted):
    """Calculates final displacement error.
    Does not performs any mode aggregation

    Args:
        ground_truth (np.ndarray): array of shape (n_timestamps, 2)
        predicted (np.ndarray): array of shape (n_modes, n_timestamps, 2)

    Returns:
        np.ndarray: array of shape (n_modes,)
    """
    return np.linalg.norm(ground_truth - predicted, axis=-1)[:, -1]


def min_ade(ground_truth, predicted):
    return np.min(average_displacement_error(ground_truth, predicted), axis=-1)


def min_fde(ground_truth, predicted):
    return np.min(final_displacement_error(ground_truth, predicted), axis=-1)


def avg_ade(ground_truth, predicted):
    return np.mean(average_displacement_error(ground_truth, predicted), axis=-1)


def avg_fde(ground_truth, predicted):
    return np.mean(final_displacement_error(ground_truth, predicted), axis=-1)


def top1_ade(ground_truth, predicted, weights):
    argmax = np.argmax(weights)
    return average_displacement_error(ground_truth, predicted)[argmax]


def top1_fde(ground_truth, predicted, weights):
    argmax = np.argmax(weights)
    return final_displacement_error(ground_truth, predicted)[argmax]


def weighted_ade(ground_truth, predicted, weights):
    weights = _softmax_normalize(weights)
    ade = average_displacement_error(ground_truth, predicted) * weights
    return np.sum(ade, axis=-1)


def weighted_fde(ground_truth, predicted, weights):
    weights = _softmax_normalize(weights)
    fde = final_displacement_error(ground_truth, predicted) * weights
    return np.sum(fde, axis=-1)


def _assert_weights_non_negative(weights: np.ndarray):
    if (weights < 0).sum() > 0:
        raise ValueError('Weights are expected to be non-negative.')


def _softmax_normalize(weights: np.ndarray) -> np.ndarray:
    weights = np.exp(weights - np.max(weights))
    return weights / weights.sum(axis=0)
