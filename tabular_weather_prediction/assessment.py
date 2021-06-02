from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve

import matplotlib

matplotlib.use('agg')
import numpy as np
from sklearn.metrics import auc
import seaborn as sns

sns.set()
sns.set(font_scale=1.25)


def prr_classification(labels, probs, measure, rev: bool):
    # Get predictions
    preds = np.argmax(probs, axis=1)

    if rev:
        inds = np.argsort(measure)[::-1]
    else:
        inds = np.argsort(measure)

    total_data = np.float(preds.shape[0])
    errors, percentages = [], []

    for i in range(preds.shape[0]):
        errors.append(np.sum(
            np.asarray(labels[inds[:i]] != preds[inds[:i]], dtype=np.float32)) * 100.0 / total_data)
        percentages.append(float(i + 1) / total_data * 100.0)
    errors, percentages = np.asarray(errors)[:, np.newaxis], np.asarray(percentages)

    base_error = errors[-1]
    n_items = errors.shape[0]
    auc_uns = 1.0 - auc(percentages / 100.0, errors[::-1] / 100.0)

    random_rejection = np.asarray(
        [base_error * (1.0 - float(i) / float(n_items)) for i in range(n_items)],
        dtype=np.float32)
    auc_rnd = 1.0 - auc(percentages / 100.0, random_rejection / 100.0)
    orc_rejection = np.asarray(
        [base_error * (1.0 - float(i) / float(base_error / 100.0 * n_items)) for i in
         range(int(base_error / 100.0 * n_items))], dtype=np.float32)
    orc = np.zeros_like(errors)
    orc[0:orc_rejection.shape[0]] = orc_rejection
    auc_orc = 1.0 - auc(percentages / 100.0, orc / 100.0)

    rejection_ratio = (auc_uns - auc_rnd) / (auc_orc - auc_rnd) * 100.0
    return rejection_ratio, auc_uns


def prr_regression(targets, preds, measure):
    measure_loc = measure
    preds = np.squeeze(preds)

    # Compute total MSE
    error = (preds - targets) ** 2
    MSE_0 = np.mean(error)

    # Create array
    array = np.concatenate(
        (preds[:, np.newaxis], targets[:, np.newaxis], error[:, np.newaxis], measure_loc[:, np.newaxis]), axis=1)

    # Results arrays
    results_max = [[0.0, 0.0]]
    results_var = [[0.0, 0.0]]
    results_min = [[0.0, 0.0]]

    optimal_ranking = array[:, 2].argsort()
    sorted_array = array[optimal_ranking]  # Sort by error

    for i in range(1, array.shape[0]):
        x = np.concatenate((sorted_array[:-i, 0], sorted_array[-i:, 1]), axis=0)
        mse = np.mean((x - sorted_array[:, 1]) ** 2)
        # Best rejection
        results_max.append([float(i) / float(array.shape[0]), (MSE_0 - mse) / MSE_0])
        # Random Rejection
        results_min.append([float(i) / float(array.shape[0]), float(i) / float(array.shape[0])])

    uncertainty_ranking = array[:, 3].argsort()
    sorted_array = array[uncertainty_ranking]  # Sort by uncertainty

    for i in range(1, array.shape[0]):
        x = np.concatenate((sorted_array[:-i, 0], sorted_array[-i:, 1]), axis=0)
        mse = np.mean((x - sorted_array[:, 1]) ** 2)
        results_var.append([float(i) / float(array.shape[0]), (MSE_0 - mse) / MSE_0])

    max_auc = auc([x[0] for x in results_max], [x[1] for x in results_max])
    var_auc = auc([x[0] for x in results_var], [x[1] for x in results_var])
    min_auc = auc([x[0] for x in results_min], [x[1] for x in results_min])

    AUC_RR = (var_auc - min_auc) / (max_auc - min_auc)

    return AUC_RR, var_auc


def ood_detect(domain_labels, in_measure, out_measure, mode='ROC', pos_label=1):
    scores = np.concatenate((in_measure, out_measure), axis=0)
    scores = np.asarray(scores, dtype=np.float128)
    if pos_label != 1:
        scores *= -1.0

    if mode == 'PR':
        precision, recall, thresholds = precision_recall_curve(domain_labels, scores)
        aupr = auc(recall, precision)
        return aupr

    elif mode == 'ROC':
        roc_auc = roc_auc_score(domain_labels, scores)
        return roc_auc


def nll_regression(target, mu, var, epsilon=1e-8, raw=False):
    nll = (target - mu) ** 2 / (2.0 * var + epsilon) + np.log(var + epsilon) / 2.0 + np.log(2 * np.pi) / 2.0
    if raw:
        return nll
    return np.mean(nll)


def nll_class(target, probs, epsilon=1e-10):
    log_p = -np.log(probs + epsilon)
    return target * log_p[:, 1] + (1 - target) * log_p[:, 0]


def ens_nll_regression(target, preds, epsilon=1e-8, raw=False):
    mu = preds[:, :, 0]
    var = preds[:, :, 1]
    nll = (target - mu) ** 2 / (2.0 * var + epsilon) + np.log(var + epsilon) / 2.0 + np.log(2 * np.pi) / 2.0
    proba = np.exp(-1 * nll)
    if raw:
        return -1 * np.log(np.mean(proba, axis=0))
    return np.mean(-1 * np.log(np.mean(proba, axis=0)))


def calc_rmse(preds, target, raw=False):
    if raw:
        return (preds - target) ** 2
    return np.sqrt(np.mean((preds - target) ** 2))


def ens_rmse(target, preds, epsilon=1e-8, raw=False):
    means = preds[:, :, 0]  # mean values predicted by all models
    avg_mean = np.mean(means, axis=0)  # average predicted mean value
    if raw:
        return calc_rmse(avg_mean, target, raw=True)
    return calc_rmse(avg_mean, target)