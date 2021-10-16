import argparse
import os, sys

sys.path.append('../../weather/')

import numpy as np
from joblib import Parallel, delayed
from sacrebleu import corpus_bleu
from nltk.translate import gleu_score
import pandas as pd

from sklearn.metrics import *
from sklearn.utils.multiclass import type_of_target
from sklearn.utils import check_consistent_length, column_or_1d, assert_all_finite
from sklearn.utils.extmath import stable_cumsum

import matplotlib

matplotlib.use('agg')
import seaborn as sns

sns.set()

parser = argparse.ArgumentParser(description='Assess translation performance')
parser.add_argument('path', type=str,
                    help='Path of directory containing data.')
parser.add_argument('domain_path', type=str,
                    help='Path of file containing domain labels.')
parser.add_argument('--save_path', type=str, default='./results.txt',
                    help='Path to where to save output.')
parser.add_argument('--beam_width', type=int, default=5,
                    help='Path of directory where to save results.')
parser.add_argument('--nbest', type=int, default=5,
                    help='Path of directory where to save results.')
parser.add_argument('--ensemble', action='store_true',
                    help='Whether to load in additional ensemble-based measures.')


def _check_pos_label_consistency(pos_label, y_true):
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


def _binary_clf_curve_ret(y_true, y_score, pos_label=None, sample_weight=None):
    # Check to make sure y_true is valid
    y_type = type_of_target(y_true)
    if not (y_type == "binary" or
            (y_type == "multiclass" and pos_label is not None)):
        raise ValueError("{0} format is not supported".format(y_type))

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    pos_label = _check_pos_label_consistency(pos_label, y_true)

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
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


def _precision_recall_curve_retention(y_true, probas_pred, *, pos_label=None,
                                      sample_weight=None):
    fps, tps, thresholds = _binary_clf_curve_ret(y_true, probas_pred,
                                                 pos_label=pos_label,
                                                 sample_weight=sample_weight)

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(-1, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def _acceptable_error(errors, threshold):
    return np.asarray(errors <= threshold, dtype=np.float32)


def _calc_fbeta_regection_curve(errors, uncertainty, threshold, beta=1.0, group_by_uncertainty=True, eps=1e-10):
    ae = _acceptable_error(errors, threshold)
    pr, rec, _ = _precision_recall_curve_retention(ae, -uncertainty)
    pr = np.asarray(pr)
    rec = np.asarray(rec)
    f_scores = (1 + beta ** 2) * pr * rec / (pr * beta ** 2 + rec + eps)

    return f_scores, pr, rec


def f_beta_metrics(errors, uncertainty, threshold, beta=1.0):
    """

    :param errors: Per sample errors - array [n_samples]
    :param uncertainty: Uncertainties associated with each prediction. rray [n_samples]
    :param threshold: The error threshold below which we consider the prediction acceptable
    :param beta: The beta value for the F_beta metric. Defaults to 1
    :return: fbeta_auc, fbeta_95, retention
    """
    f_scores, pr, rec = _calc_fbeta_regection_curve(errors, uncertainty, threshold, beta)
    ret = np.arange(pr.shape[0]) / pr.shape[0]

    f_auc = auc(ret[::-1], f_scores)
    f95 = f_scores[::-1][np.int(0.95 * pr.shape[0])]

    return f_auc, f95, f_scores[::-1]


def load_uncertainties(path, n_best=5, beam_width=5):
    eoe = np.loadtxt(os.path.join(path, 'entropy_expected.txt'), dtype=np.float32)
    exe = np.loadtxt(os.path.join(path, 'expected_entropy.txt'), dtype=np.float32)
    mi = np.loadtxt(os.path.join(path, 'mutual_information.txt'), dtype=np.float32)
    epkl = np.loadtxt(os.path.join(path, 'epkl.txt'), dtype=np.float32)
    mkl = np.loadtxt(os.path.join(path, 'mkl.txt'), dtype=np.float32)
    score = np.loadtxt(os.path.join(path, 'score.txt'), dtype=np.float32)
    aep_tu = np.loadtxt(os.path.join(path, 'aep_tu.txt'), dtype=np.float32)
    aep_du = np.loadtxt(os.path.join(path, 'aep_du.txt'), dtype=np.float32)
    npmi = np.loadtxt(os.path.join(path, 'npmi.txt'), dtype=np.float32)
    lprobs = np.loadtxt(os.path.join(path, 'log_probs.txt'), dtype=np.float32)
    sMKL_pe = np.loadtxt(os.path.join(path, 'score_npmi.txt'), dtype=np.float32)

    # Expectation of Products Measures
    ep_eoe = np.loadtxt(os.path.join(path, 'ep_entropy_expected.txt'), dtype=np.float32)
    ep_mi = np.loadtxt(os.path.join(path, 'ep_mutual_information.txt'), dtype=np.float32)
    ep_epkl = np.loadtxt(os.path.join(path, 'ep_epkl.txt'), dtype=np.float32)
    ep_mkl = np.loadtxt(os.path.join(path, 'ep_mkl.txt'), dtype=np.float32)

    # Heuristic Measures
    var = np.loadtxt(os.path.join(path, 'var.txt'), dtype=np.float32)
    varcombo = np.loadtxt(os.path.join(path, 'varcombo.txt'), dtype=np.float32)
    logvar = np.loadtxt(os.path.join(path, 'logvar.txt'), dtype=np.float32)
    logcombo = np.loadtxt(os.path.join(path, 'logcombo.txt'), dtype=np.float32)

    unc_dict = {'Total Uncertainty-PE': eoe,
                'Total Uncertainty-EP': ep_eoe,
                'SCR-PE': score,
                'SCR-EP': aep_tu,
                'Data Uncertainty': exe,
                'Mutual Information-PE': mi,
                'Mutual Information-EP': ep_mi,
                'EPKL-PE': epkl,
                'EPKL-EP': ep_epkl,
                'Reverse Mutual Information': mkl,
                'ep_MKL': ep_mkl,
                'sMKL-PE': sMKL_pe,
                'sMKL-EP': npmi,
                'var': var,
                'varcombo': varcombo,
                'logvar': logvar,
                'logcomvo': logcombo
                }

    for key in unc_dict.keys():
        uncertainties = unc_dict[key]
        unc_dict[key] = np.mean(np.reshape(uncertainties, [-1, beam_width])[:, :n_best], axis=1)
    return unc_dict


def eval_ood_detect(in_uncertainties, out_uncertainties, save_path):
    for key in in_uncertainties.keys():
        ood_detect(in_uncertainties[key],
                   out_uncertainties[key],
                   measure_name=key,
                   save_path=save_path)


def ood_detect(in_measure, out_measure, measure_name, save_path):
    scores = np.concatenate((in_measure, out_measure), axis=0)
    scores = np.asarray(scores, dtype=np.float128)
    results_path = save_path
    save_path = os.path.split(save_path)[0]

    domain_labels = np.concatenate((np.zeros_like(in_measure, dtype=np.int32),
                                    np.ones_like(out_measure, dtype=np.int32)), axis=0)

    fpr, tpr, thresholds = roc_curve(domain_labels, scores)
    roc_auc = roc_auc_score(domain_labels, scores)
    with open(results_path, 'a') as f:
        f.write('AUROC using ' + measure_name + ": " + str(np.round(roc_auc * 100.0, 1)) + '\n')


def eval_predictions(refs, hypos, nlls, nbest=5):
    weights = np.exp(-nlls)
    weights /= weights.sum(axis=1, keepdims=True)

    hyposb = [hypo[0] for hypo in hypos]
    bleu = corpus_bleu(sys_stream=hyposb, ref_streams=[refs]).score

    refsg = [[ref.split()] for ref in refs]
    gleus = []
    for i in range(nbest):
        hyposg = [hypo[i].split() for hypo in hypos]
        gleu = np.asarray(
            [gleu_score.sentence_gleu(references=r, hypothesis=h) * 100.0 for r, h in zip(refsg, hyposg)])[:,
               np.newaxis]
        gleus.append(gleu)
    gleus = np.concatenate(gleus, axis=1)
    return bleu, gleus, weights


def calc_uncertainty_regection_curve(errors, uncertainty, group_by_uncertainty=True):
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
        uncertainty_order = uncertainty.argsort()
        errors = errors[uncertainty_order]

    error_rates = np.zeros(n_objects + 1)
    error_rates[:-1] = np.cumsum(errors)[::-1] / n_objects
    return error_rates


def calc_aucs(errors, uncertainty):
    uncertainty_rejection_curve = calc_uncertainty_regection_curve(errors, uncertainty)
    uncertainty_rejection_auc = uncertainty_rejection_curve.mean()
    random_rejection_auc = uncertainty_rejection_curve[0] / 2
    ideal_rejection_auc = calc_uncertainty_regection_curve(errors, errors).mean()

    rejection_ratio = (uncertainty_rejection_auc - random_rejection_auc) / (
            ideal_rejection_auc - random_rejection_auc) * 100.0
    return rejection_ratio, uncertainty_rejection_auc


def load_text(path, beam_width=5):
    refs, hypos = [], []
    with open(os.path.join(path, 'refs.txt'), 'r') as f:
        for line in f.readlines():
            refs.append(line[:-1])

    with open(os.path.join(path, 'hypos.txt'), 'r') as f:
        count = 0
        hypos = []
        hypos_joint = []
        for line in f.readlines():
            count += 1
            hypos_joint.append(line[:-1])
            if count % beam_width == 0:
                hypos.append(hypos_joint)
                hypos_joint = []

    ids = np.loadtxt(os.path.join(path, 'ref_ids.txt'), dtype=np.int32)
    nlls = -np.loadtxt(os.path.join(path, 'hypo_likelihoods.txt'), dtype=np.float32).reshape([-1, beam_width])

    return refs, hypos, ids, nlls


def eval_gleu_retention(errors, in_uncertainties, out_uncertainties, save_path):
    with open(save_path, 'a') as f:
        for key in in_uncertainties.keys():
            uncertainties = np.concatenate([in_uncertainties[key], out_uncertainties[key]], axis=0)
            prr, r_auc = calc_aucs(errors, uncertainties)
            f.write('R-ROC (SCORE) using ' + key + ": " + str(np.round(r_auc, 3)) + '\n')
        for key in in_uncertainties.keys():
            uncertainties = np.concatenate([in_uncertainties[key], out_uncertainties[key]], axis=0)
            prr, r_auc = calc_aucs(errors, uncertainties)
            f.write('PRR using ' + key + ": " + str(np.round(prr, 3)) + '\n')


def eval_fbeta(errors, threshold, in_uncertainties, out_uncertainties, save_path):
    with open(save_path, 'a') as f:
        for key in in_uncertainties.keys():
            uncertainties = np.concatenate([in_uncertainties[key], out_uncertainties[key]], axis=0)
            f_auc, f95, f_scores = f_beta_metrics(errors, uncertainties, threshold=threshold)
            f.write('F1-AUC  using ' + key + ": " + str(np.round(f_auc, 3)) + '\n')
        for key in in_uncertainties.keys():
            uncertainties = np.concatenate([in_uncertainties[key], out_uncertainties[key]], axis=0)
            f_auc, f95, f_scores = f_beta_metrics(errors, uncertainties, threshold=threshold)
            f.write('F1@95  using ' + key + ": " + str(np.round(f95, 3)) + '\n')


def main():
    args = parser.parse_args()

    # Load refs and hypos
    refs, hypos, ids, nlls = load_text(args.path, beam_width=args.beam_width)
    bleu = corpus_bleu(sys_stream=[h[0] for h in hypos], ref_streams=[refs]).score
    domain_labels = np.loadtxt(args.domain_path)

    nlls_in = nlls[np.where(domain_labels == 0)]
    nlls_out = nlls[np.where(domain_labels == 1)]

    refs_in = [refs[i] for i in np.where(domain_labels == 0)[0]]
    refs_out = [refs[i] for i in np.where(domain_labels == 1)[0]]

    hypos_in = [hypos[i] for i in np.where(domain_labels == 0)[0]]
    hypos_out = [hypos[i] for i in np.where(domain_labels == 1)[0]]

    bleu_in, gleus_in, weights_in = eval_predictions(refs_in, hypos_in, nlls_in)
    bleu_out, gleus_out, weights_out = eval_predictions(refs_out, hypos_out, nlls_out)

    gleus = np.concatenate([gleus_in, gleus_out], axis=0)
    weights = np.concatenate([weights_in, weights_out], axis=0)
    gleu_errors = 100.0 - gleus

    egleus = np.mean(np.sum(gleus * weights, axis=1))
    mgleus = np.mean(np.max(gleus, axis=1))

    with open(args.save_path, 'a') as f:
        f.write('BLEU eval-in: ' + str(np.round(bleu_in, 2)) + '\n')
        f.write('BLEU eval-out: ' + str(np.round(bleu_out, 2)) + '\n')
        f.write('BLEU eval: ' + str(np.round(bleu, 2)) + '\n')
        f.write('eGLEU eval-in: ' + str(np.round(np.mean(np.sum(gleus_in * weights_in, axis=1)), 2)) + '\n')
        f.write('eGLEU eval-out: ' + str(np.round(np.mean(np.sum(gleus_out * weights_out, axis=1)), 2)) + '\n')
        f.write('mGLEU eval-in: ' + str(np.round(np.mean(np.max(gleus_in, axis=1)), 2)) + '\n')
        f.write('mGLEU eval-out: ' + str(np.round(np.mean(np.max(gleus_out, axis=1)), 2)) + '\n')
        f.write('eGLEU eval: ' + str(np.round(egleus, 2)) + '\n')
        f.write('mGLEU eval: ' + str(np.round(mgleus, 2)) + '\n')

    # Load uncertainties
    if args.ensemble:
        uncertainties = load_uncertainties(args.path, beam_width=args.beam_width, n_best=args.nbest)

        uncertainties_in, uncertainties_out = {}, {}
        for key in uncertainties.keys():
            uncertainties_in[key] = uncertainties[key][np.where(domain_labels == 0)]
            uncertainties_out[key] = uncertainties[key][np.where(domain_labels == 1)]

    else:
        uncertainties_in = {'NLL': np.mean(nlls_in, axis=1)}
        uncertainties_out = {'NLL': np.mean(nlls_out, axis=1)}

    eval_gleu_retention(errors=np.sum(gleu_errors * weights, axis=1),
                        in_uncertainties=uncertainties_in,
                        out_uncertainties=uncertainties_out,
                        save_path=args.save_path)

    eval_fbeta(errors=np.sum(gleu_errors * weights, axis=1),
               threshold=60.0,
               in_uncertainties=uncertainties_in,
               out_uncertainties=uncertainties_out,
               save_path=args.save_path)

    eval_ood_detect(in_uncertainties=uncertainties_in,
                    out_uncertainties=uncertainties_out,
                    save_path=args.save_path)


if __name__ == '__main__':
    main()
