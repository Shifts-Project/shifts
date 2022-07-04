import numpy as np
from typing import Dict


def epkl_reg(preds: np.array, epsilon=1e-10) -> np.array:
    """
    Taken from here: https://github.com/yandex-research/shifts.
    Modification/additions:
    Rename vars variable to vars_.
    :param: preds: array [n_models, n_samples, 2] - mean and var along last axis.
    """
    means = preds[:, :, 0]
    vars_ = preds[:, :, 1] + epsilon
    logvars = np.log(vars_)

    avg_means = np.mean(means, axis=0)
    avg_second_moments = np.mean(means * means + vars_, axis=0)

    inv_vars = 1.0 / vars_
    avg_inv_vars = np.mean(inv_vars, axis=0)
    mean_inv_var = inv_vars * means
    avg_mean_inv_var = np.mean(mean_inv_var, axis=0)
    avg_mean2_inv_var = np.mean(means * mean_inv_var + logvars, axis=0) + np.log(2 * np.pi)

    epkl = 0.5 * (avg_second_moments * avg_inv_vars - 2 * avg_means * avg_mean_inv_var + avg_mean2_inv_var)

    return epkl


def ensemble_uncertainties_regression(preds: np.array) -> Dict:
    """
    Taken from here: https://github.com/yandex-research/shifts
    :param: preds: array [n_models, n_samples, 2] - last dim is mean, var
    """
    epkl = epkl_reg(preds=preds)
    var_mean = np.var(preds[:, :, 0], axis=0)
    mean_var = np.mean(preds[:, :, 1], axis=0)

    uncertainty = {'tvar': var_mean + mean_var,
                   'mvar': mean_var,
                   'varm': var_mean,
                   'epkl': epkl}

    return uncertainty
