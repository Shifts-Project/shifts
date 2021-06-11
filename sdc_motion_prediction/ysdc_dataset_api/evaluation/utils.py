from collections import defaultdict

import numpy as np

from ..proto import Submission
from ..evaluation import min_ade, min_fde, top1_ade, top1_fde
from ..utils.map import repeated_points_to_array


def save_submission(filepath, submission):
    with open(filepath, 'wb') as fout:
        fout.write(submission.SerializeToString())


def load_submission(filepath):
    with open(filepath, 'rb') as fin:
        serialized = fin.read()
    submission = Submission()
    submission.ParseFromString(serialized)
    return submission


def evaluate_submission(submission, dataset):
    dataset_iter = iter(dataset)
    metrics = defaultdict(list)
    for prediction in submission.predictions:
        data_item = next(dataset_iter)
        if (
            prediction.scene_id != data_item['scene_id']
            or prediction.track_id != data_item['track_id']
        ):
            raise ValueError('Order of predictions does not coincide with the dataset')
        n_modes = len(prediction.weighted_trajectories)
        modes = np.empty((n_modes, 25, 2))
        weights = np.empty(n_modes)
        for i, weighted_trajectory in enumerate(prediction.weighted_trajectories):
            modes[i] = repeated_points_to_array(weighted_trajectory.trajectory)
            weights[i] = weighted_trajectory.weight
        metrics['min_ade'] = min_ade(data_item['ground_truth_trajectory'], modes)
        metrics['min_fde'] = min_fde(data_item['ground_truth_trajectory'], modes)
        metrics['top1_ade'] = top1_ade(data_item['ground_truth_trajectory'], modes, weights)
        metrics['top1_fde'] = top1_fde(data_item['ground_truth_trajectory'], modes, weights)
    return {
        metric_name: np.mean(metric_values)
        for metric_name, metric_values in metrics.items()
    }
