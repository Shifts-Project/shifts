import numpy as np


def entropy_of_expected_class(probs, epsilon=1e-10):
    """
    :param probs: array [num_models, num_examples, num_classes]
    :return: array [num_examples}
    """
    mean_probs = np.mean(probs, axis=0)
    log_probs = -np.log(mean_probs + epsilon)
    return np.sum(mean_probs * log_probs, axis=1)


def expected_entropy_class(probs, epsilon=1e-10):
    """
    :param probs: array [num_models, num_examples, num_classes]
    :return: array [num_examples}
    """
    log_probs = -np.log(probs + epsilon)

    return np.mean(np.sum(probs * log_probs, axis=2), axis=0)


def ensemble_uncertainties_classification(probs, epsilon=1e-10):
    """
    :param probs: array [num_models, num_examples, num_classes]
    :return: Dictionary of uncertaintties
    """
    mean_probs = np.mean(probs, axis=0)
    mean_lprobs = np.mean(np.log(probs + epsilon), axis=0)
    conf = np.max(mean_probs, axis=1)

    eoe = entropy_of_expected_class(probs, epsilon)
    exe = expected_entropy_class(probs, epsilon)

    mutual_info = eoe - exe

    epkl = -np.sum(mean_probs * mean_lprobs, axis=1) - exe

    uncertainty = {'confidence': conf,
                   'entropy_of_expected': eoe,
                   'expected_entropy': exe,
                   'mutual_information': mutual_info,
                   'epkl': epkl,
                   'reverse_mutual_information': epkl - mutual_info
                   }

    return uncertainty


def epkl_reg(preds, epsilon=1e-10):
    """
    preds: array [n_models, n_samples, 2] - mean and var along last axis.
    """
    means = preds[:, :, 0]
    vars = preds[:, :, 1] + epsilon
    logvars = np.log(vars)

    avg_means = np.mean(means, axis=0)
    avg_second_moments = np.mean(means * means + vars, axis=0)

    inv_vars = 1.0 / vars
    avg_inv_vars = np.mean(inv_vars, axis=0)
    mean_inv_var = inv_vars * means
    avg_mean_inv_var = np.mean(mean_inv_var, axis=0)
    avg_mean2_inv_var = np.mean(means * mean_inv_var + logvars, axis=0) + np.log(2 * np.pi)

    epkl = 0.5 * (avg_second_moments * avg_inv_vars - 2 * avg_means * avg_mean_inv_var + avg_mean2_inv_var)

    return epkl


def ensemble_uncertainties_regression(preds):
    """
    preds: array [n_models, n_samples, 2] - last dim is mean, var
    """
    epkl = epkl_reg(preds)

    var_mean = np.var(preds[:, :, 0], axis=0)
    mean_var = np.mean(preds[:, :, 1], axis=0)

    uncertainty = {'tvar': var_mean + mean_var,
                   'mvar': mean_var,
                   'varm': var_mean,
                   'epkl': epkl}

    return uncertainty
