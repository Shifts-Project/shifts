import pandas as pd
import numpy as np
from sklearn.utils.multiclass import type_of_target
from sklearn.utils import check_consistent_length, column_or_1d, assert_all_finite
from sklearn.utils.extmath import stable_cumsum
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, auc


def calc_uncertainty_regection_curve(errors, uncertainty, group_by_uncertainty=True, rank_type="ordered"):
    """
    Calculates the mean error for retention in range [0,1] given a specified rank_type ordering of errors
    Taken from here: https://github.com/yandex-research/shifts .
    Modifications/additions:
    The code is extended to support random and optimal error ordering
    :param errors: Per sample errors - array [n_samples]
    :param uncertainty: Uncertainties associated with each prediction. array [n_samples]
    :param group_by_uncertainty: Whether to group errors by uncertainty
    :param rank_type: The type of error ordering. Available options are:
                        1. ordered: Errors are ordered based on the rank of their uncertainties
                        2. random: Errors are shuffled randomly, resulting in no error ordering.
                        3. optimal: Errors are ordered based on their own rank (in this way we simulate the case of
                        absolute correlation of errors with uncertainties)
    :return: The mean error for retention in range [0,1]
    """
    n_objects = errors.shape[0]
    if group_by_uncertainty:
        data = pd.DataFrame(dict(
            errors=errors,
            uncertainty=uncertainty
        ))
        mean_errors = data.groupby("uncertainty").mean()
        mean_errors.rename(columns={"errors": "mean_errors"}, inplace=True)
        data = data.join(mean_errors, "uncertainty")
        data.drop("errors", axis=1, inplace=True)

        uncertainty_order = data["uncertainty"].argsort()
        errors = data["mean_errors"][uncertainty_order]
    else:
        if rank_type == "random":
            uncertainty_order = np.arange(errors.shape[0])
            np.random.shuffle(uncertainty_order)

        elif rank_type == "optimal":
            uncertainty_order = errors.argsort()

        elif rank_type == "ordered":
            uncertainty_order = uncertainty.argsort()

        else:
            raise ValueError("Not supported ranking type")

        errors = errors[uncertainty_order]

    error_rates = np.zeros(n_objects + 1)
    error_rates[:-1] = np.cumsum(errors)[::-1] / n_objects
    return error_rates


assert np.allclose(
    calc_uncertainty_regection_curve(np.array([2, 1]), np.array([1, 0])).mean(),
    2 / 3
)
assert np.allclose(
    calc_uncertainty_regection_curve(np.arange(5), np.array([0, 0, 2, 1, 1])).mean(),
    0.8
)
debug_errors = np.random.rand(10)
assert np.allclose(
    calc_uncertainty_regection_curve(debug_errors, np.zeros_like(debug_errors)).mean(),
    debug_errors.mean() / 2
)


def _check_pos_label_consistency(pos_label, y_true):
    """
    Taken from here: https://github.com/yandex-research/shifts
    """
    # ensure binary classification if pos_label is not specified
    # classes.dtype.kind in ('O', 'U', 'S') is required to avoid
    # triggering a FutureWarning by calling np.array_equal(a, b)
    # when elements in the two arrays are not comparable.
    classes = np.unique(y_true)
    if (pos_label is None and (
            classes.dtype.kind in 'OUS' or
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1])))):
        classes_repr = ", ".join(repr(c) for c in classes)
        raise ValueError(
            f"y_true takes value in {{{classes_repr}}} and pos_label is not "
            f"specified: either make y_true take value in {{0, 1}} or "
            f"{{-1, 1}} or pass pos_label explicitly."
        )
    elif pos_label is None:
        pos_label = 1.0

    return pos_label


def _binary_clf_curve_ret(y_true, y_score, error_inverse, pos_label=None, sample_weight=None, rank_type="ordered"):
    """
    Taken from here: https://github.com/yandex-research/shifts.
    Modifications/additions:
    The code is extended to support random and optimal error ordering.
    """
    # Check to make sure y_true is valid
    y_type = type_of_target(y_true)
    if not (y_type == "binary" or
            (y_type == "multiclass" and pos_label is not None)):
        raise ValueError("{0} format is not supported".format(y_type))

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    error_inverse = column_or_1d(error_inverse)
    assert_all_finite(y_true)
    assert_all_finite(y_score)
    assert_all_finite(error_inverse)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    pos_label = _check_pos_label_consistency(pos_label, y_true)

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    if rank_type == "random":
        desc_score_indices = np.arange(y_score.shape[0])
        np.random.shuffle(desc_score_indices)
    if rank_type == "optimal":
        desc_score_indices = np.argsort(error_inverse, kind="mergesort")[::-1]
    if rank_type == "ordered":
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    # distinct_value_indices = np.where(np.diff(y_score))[0]
    # threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)  # [threshold_idxs]
    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = stable_cumsum((1 - y_true) * weight)  # [threshold_idxs]
    else:
        fps = stable_cumsum((1 - y_true))  # [threshold_idxs]
    return fps, tps, y_score  # [threshold_idxs]


def _precision_recall_curve_retention(y_true, probas_pred, error_inverse, *, pos_label=None,
                                      sample_weight=None, rank_type="ordered"):
    """
    Taken from here: https://github.com/yandex-research/shifts.
    Modifications/additions:
    The code is extended to support random and optimal error ordering.
    """
    fps, tps, thresholds = _binary_clf_curve_ret(y_true, probas_pred,
                                                 pos_label=pos_label,
                                                 sample_weight=sample_weight,
                                                 error_inverse=error_inverse,
                                                 rank_type=rank_type)

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(-1, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def _acceptable_error(errors, threshold):
    """
    Taken from here: https://github.com/yandex-research/shifts
    """
    return np.asarray(errors <= threshold, dtype=np.float32)


def _calc_fbeta_regection_curve(errors, uncertainty, rank_type, threshold, beta=1.0, eps=1e-10):
    """
    Taken from here: https://github.com/yandex-research/shifts
    Modifications/additions:
    The code is extended to support random and optimal error ordering.
    """
    ae = _acceptable_error(errors, threshold)
    pr, rec, _ = _precision_recall_curve_retention(ae, -uncertainty, error_inverse=-errors, rank_type=rank_type)
    pr = np.asarray(pr)
    rec = np.asarray(rec)
    f_scores = (1 + beta ** 2) * pr * rec / (pr * beta ** 2 + rec + eps)

    return f_scores, pr, rec


def f_beta_metrics(errors, uncertainty, threshold, beta=1.0, rank_type="ordered"):
    """
    Taken from here: https://github.com/yandex-research/shifts.
    Modifications/additions:
    The code is extended to support random and optimal error ordering
    :param errors: Per sample errors - array [n_samples]
    :param uncertainty: Uncertainties associated with each prediction. array [n_samples]
    :param threshold: The error threshold below which we consider the prediction acceptable
    :param beta: The beta value for the F_beta metric. Defaults to 1
    :param rank_type: The type of error ordering. Available options are:
                        1. ordered: Errors are ordered based on the rank of their uncertainties
                        2. random: Errors are shuffled randomly, resulting in no error ordering.
                        3. optimal: Errors are ordered based on their own rank (in this way we simulate the case of
                        absolute correlation of errors with uncertainties)
    :return: fbeta_auc, fbeta_95, retention
    """
    f_scores, pr, rec = _calc_fbeta_regection_curve(errors=errors, uncertainty=uncertainty,
                                                    threshold=threshold, beta=beta, rank_type=rank_type)
    ret = np.arange(pr.shape[0]) / pr.shape[0]

    f_auc = auc(ret[::-1], f_scores)
    f95 = f_scores[::-1][np.int(0.95 * pr.shape[0])]

    return f_auc, f95, f_scores[::-1]


def get_ensemble_errors(predictions, y_true, squared=True):
    """
    Get the prediction errors of an ensemble, defined as the differences (or squared differences) of actual target
    values and the ensemble average predicted output
    :param predictions: array [ensemble_size x num_samples x num_params]
           where
           - ensemble size is the number of members of the ensemble
           - num_samples is the number of records
           - num_params is the number of parameters the model outputs (ex. in the case of a probabilistic model with
           Normal distribution at the output, num_params=2 for mean and std of Normal)
    :param y_true: array of the target values
    :param squared: Whether to return the squared difference of y_true and ensemble average predicted output
    :return:
    """
    ens_avg_preds = np.squeeze(np.mean(predictions[:, :, 0], axis=0))
    y_true = np.asarray(y_true)
    if not squared:
        errors = y_true - ens_avg_preds
    else:
        errors = (y_true - ens_avg_preds) ** 2

    return errors


def get_ensemble_params(pred: np.array):
    """
    An ensemble of probabilistic models that have a normal distribution at the output, is treated as a mixture of
    uniformly weighted Normal distributions. The mixture is approximated by the marginal normal distribution
    N(ensemble_mean, ensemble_std)
    :param pred: Predictions
    :return: The parameters of the marginal normal distribution N(ensemble_mean, ensemble_std)
    """
    ensemble_mean = np.squeeze(np.mean(pred[:, :, 0], axis=0))
    ensemble_std = np.sqrt(
        np.squeeze(
            np.mean(
                pred[:, :, 1] + np.square(pred[:, :, 0]), axis=0
            )
        ) - np.square(ensemble_mean)
    )

    return ensemble_mean, ensemble_std


def get_ensemble_metric(pred: np.array, y_true: np.array, metric: str) -> float:
    """
    Calculates the requested performance metric of an ensemble of models
    :param pred: Predictions
    :param y_true: Actual target values
    :param metric: Supported metrics are "mae", "mse", "rmse", "mape"
    :return: The requested performance metric of ensemble
    """
    mean_pred = np.squeeze(np.mean(pred[:, :, 0], axis=0))

    if metric == "mae":
        error = mean_absolute_error(y_true=y_true, y_pred=mean_pred)
    elif metric == "rmse":
        error = mean_squared_error(y_true=y_true, y_pred=mean_pred, squared=False)
    elif metric == "mse":
        error = mean_squared_error(y_true=y_true, y_pred=mean_pred, squared=True)
    elif metric == "mape":
        error = mean_absolute_percentage_error(y_true=y_true, y_pred=mean_pred)
    return error
