import json
import numpy as np
import sacrebleu
from nltk.translate import gleu_score
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, fbeta_score
from sklearn.metrics import precision_recall_curve
import pandas as pd

import numpy as np
from sklearn.metrics import auc
from sklearn.utils.multiclass import type_of_target
from sklearn.utils import check_consistent_length, column_or_1d, check_array, assert_all_finite
from sklearn.utils.extmath import stable_cumsum
import argparse


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


def precision_recall_curve_retention(y_true, probas_pred, *, pos_label=None,
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


def acceptable_error(errors, threshold):
    return np.asarray(errors <= threshold, dtype=np.float32)


def calc_fbeta_regection_curve(errors, uncertainty, threshold, beta=1.0, group_by_uncertainty=True, eps=1e-10):
    ae = acceptable_error(errors, threshold)
    pr, rec, _ = precision_recall_curve_retention(ae, -uncertainty)
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
    f_scores, pr, rec = calc_fbeta_regection_curve(errors, uncertainty, threshold, beta)
    ret = np.arange(pr.shape[0]) / pr.shape[0]

    f_auc = auc(ret[::-1], f_scores)
    f95 = f_scores[::-1][np.int(0.95 * pr.shape[0])]

    return f_auc, f95, f_scores[::-1]


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


def ood_detect(domain_labels, measure):
    scores = np.asarray(measure, dtype=np.float128)

    roc_auc = roc_auc_score(domain_labels, scores)
    return roc_auc


def eval():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("preds")
        parser.add_argument("refs")
        parser.add_argument("domain_labels")
        try:
            args = parser.parse_args()
        except:
            raise Exception('Missing sumbmit or target.')

        decoder = json.JSONDecoder()
        refs = []
        preds = []

        hypo_len = None
        preds_ids = []
        refs_ids = []

        domain_labels_orig = np.loadtxt(args.domain_labels, dtype=np.int32)
        with open(args.preds, 'r') as jp, open(args.refs, 'r') as jr:
            for line in jp.readlines():
                pred = decoder.decode(line)
                preds.append(pred)
                if not hypo_len:
                    hypo_len = len(pred['hypos'])
                    assert hypo_len < 11, 'The number of hypotheses exceeds 10.'
                else:
                    assert hypo_len == len(pred['hypos']), 'The number of hypotheses is not constant.'
                assert abs(np.sum([hypo['confidence'] for hypo in
                                   pred['hypos']]) - 1) < 1e-5, 'The sum of the confidence scores is not equal to 1.'
            for line in jr.readlines():
                ref = decoder.decode(line)
                refs.append(ref)

        refs = sorted(refs, key=lambda x: x['id'])
        preds = sorted(preds, key=lambda x: x['id'])

        assert len(refs) == len(preds), 'Missing some IDs.'

        for r, p in zip(refs, preds):
            assert r['id'] == p['id'], 'Wrong order of predictions.'

        refsb = [ref['ref'] for ref in refs]
        hyposb = [hypo['hypos'][0]['text'] for hypo in preds]

        refsg = [[ref['ref'].split()] for ref in refs]
        gleus = []
        for r, pr in zip(refsg, preds):
            score = 0
            for h in pr['hypos']:
                hypo = h['text'].split()
                score += h['confidence'] * gleu_score.sentence_gleu(references=r, hypothesis=hypo) * 100
            gleus.append(score)
        gleu_errors = np.asarray([100.0 - g for g in gleus])

        uncertainties = np.asarray([hypo['uncertainty'] for hypo in preds])

        domain_labels = np.asarray([domain_labels_orig[r['id']] for r in refs])

        bleu = sacrebleu.corpus_bleu(sys_stream=hyposb, ref_streams=[refsb], force=True).score
        gleu = np.mean(gleus)
        prr, auc = calc_aucs(errors=gleu_errors, uncertainty=uncertainties)
        roc_auc = ood_detect(domain_labels, uncertainties)

        f_auc, f95, _ = f_beta_metrics(gleu_errors, np.asarray([hypo['uncertainty'] for hypo in preds]), threshold=60.0)

        scores = json.dumps({
            'BLEU': bleu,
            'sGLEU': gleu,
            'AUC-F1': f_auc,
            'F1 @ 95%': f95,
            'ROC-AUC': roc_auc * 100,
        })

        print(auc, scores)
    except Exception as err:
        print('$ ' + str(err) + ' $')


eval()
