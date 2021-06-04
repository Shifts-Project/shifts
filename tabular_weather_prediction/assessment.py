from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve
import pandas as pd

import matplotlib

matplotlib.use('agg')
import numpy as np
from sklearn.metrics import auc
import seaborn as sns

sns.set()
sns.set(font_scale=1.25)

def calc_uncertainty_regection_curve(errors, uncertainty, group_by_uncertainty=True):
    n_objects = errors.shape[0]
    if group_by_uncertainty:
        data = pd.DataFrame(dict(
            errors = errors,
            uncertainty=uncertainty
        ))
        mean_errors = data.groupby("uncertainty").mean()
        mean_errors.rename(columns={"errors": "mean_errors"}, inplace=True)    
        data = data.join(mean_errors, "uncertainty")
        data.drop("errors", axis=1, inplace=True)

        uncertainty_order = data["uncertainty"].argsort()
        errors = data["mean_errors"][uncertainty_order]
    else:
        uncertainty_order = uncertainty.argsort()
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

def calc_aucs(errors, uncertainty):
    uncertainty_rejection_curve = calc_uncertainty_regection_curve(errors, uncertainty)
    uncertainty_rejection_auc = uncertainty_rejection_curve.mean()
    random_rejection_auc = uncertainty_rejection_curve[0] / 2
    ideal_rejection_auc = calc_uncertainty_regection_curve(errors, errors).mean()

    rejection_ratio = (uncertainty_rejection_auc - random_rejection_auc) / (
        ideal_rejection_auc - random_rejection_auc) * 100.0
    return rejection_ratio, uncertainty_rejection_auc
    

def prr_classification(labels, probs, measure, rev: bool):
    if rev:
        measure = -measure
    preds = np.argmax(probs, axis=1)
    errors = (labels != preds).astype("float32")
    return calc_aucs(errors, measure)

def prr_regression(targets, preds, measure):
    preds = np.squeeze(preds)
    # Compute MSE errors
    errors = (preds - targets) ** 2
    return calc_aucs(errors, measure)


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