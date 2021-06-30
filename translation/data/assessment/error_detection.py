import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from sacrebleu import corpus_bleu, sentence_bleu
from sklearn.metrics import auc

sns.set()

parser = argparse.ArgumentParser(description='Assess ood detection performance')
parser.add_argument('path', type=str,
                    help='Path of directory containing in-domain uncertainties.')
parser.add_argument('--wer', action='store_true',
                    help='Whether to evaluate using WER instead of BLEU')
parser.add_argument('--beam_width', type=int, default=5,
                    help='Path of directory where to save results.')
parser.add_argument('--nbest', type=int, default=5,
                    help='Path of directory where to save results.')
parser.add_argument('--beam_search', action='store_true',
                    help='Path of directory where to save results.')
parser.add_argument('--calibrate', action='store_true',
                    help='Path of directory where to save results.')
parser.add_argument('--noln', action='store_true',
                    help='Path of directory where to save results.')


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

    ep_eoe = np.loadtxt(os.path.join(path, 'ep_entropy_expected.txt'), dtype=np.float32)
    ep_mi = np.loadtxt(os.path.join(path, 'ep_mutual_information.txt'), dtype=np.float32)
    ep_epkl = np.loadtxt(os.path.join(path, 'ep_epkl.txt'), dtype=np.float32)
    ep_mkl = np.loadtxt(os.path.join(path, 'ep_mkl.txt'), dtype=np.float32)

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
                'MKL': mkl,
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


def get_corpus_bleu(refs, hypos, i):
    tmp_hypo = np.concatenate((refs[:i], hypos[i:]), axis=0)
    return corpus_bleu(tmp_hypo, [refs]).score


def get_bleu(refs, hypos, i):
    tmp_hypo = np.concatenate((refs[:i], hypos[i:]), axis=0)
    return np.mean([sentence_bleu([tmp_hypo[i]], [[refs[i]]], smooth_value=1.0).score for i in range(hypos.shape[0])])
    # return corpus_bleu(tmp_hypo, [refs]).score


def reject_predictions_bleu(refs, hypos, measure, path,
                            rev: bool = False, corpus_bleu=False):  # , measure_name: str, save_path: str,  show=True):
    # Get predictions

    refs = np.asarray(refs)
    hypos = np.asarray(hypos)

    if rev:
        inds = np.argsort(measure)
    else:
        inds = np.argsort(measure)[::-1]

    assert measure[inds[0]] >= measure[inds[-1]]

    sorted_refs = refs[inds]
    sorted_hypos = hypos[inds]

    # bleus = np.asarray([sentence_bleu([hypos[i]], [[refs[i]]], smooth_value=1.1).score for i in range(hypos.shape[0])])
    if not os.path.exists(f"{path}/sbleus.txt"):
        sbleus = np.asarray(
            [sentence_bleu([hypos[i]], [[refs[i]]], smooth_value=1.1).score for i in range(hypos.shape[0])])
        np.savetxt(f"{path}/sbleus.txt", sbleus)
    else:
        sbleus = np.loadtxt(f"{path}/sbleus.txt", dtype=np.float32)

    sorted_sbleus = sbleus[inds]
    # sorted_refs = refs[inds]
    # sorted_hypos = hypos[inds]

    inds = np.squeeze(inds)

    total_data = np.float(len(hypos))
    bleus, percentages = [], []

    if corpus_bleu:
        bleus = Parallel(n_jobs=-1)(delayed(get_bleu)(sorted_refs, sorted_hypos, i) for i in range(len(hypos)))
    else:
        max_sbleus = 100 * np.ones_like(sorted_sbleus)
        bleus = [np.mean(np.concatenate((max_sbleus[:i], sorted_sbleus[i:]), axis=0)) for i in range(len(hypos))]

    for i in range(len(hypos)):
        # tmp_ref = np.concatenate((sorted_hypos[:i], sorted_refs[i:]), axis=0)
        # score = corpus_bleu(tmp_ref, [sorted_hypos]).score
        # bleus.append(score)
        percentages.append(float(i + 1) / total_data * 100.0)

    bleus, percentages = np.asarray(bleus), np.asarray(percentages)

    base_bleu = bleus[0]
    n_items = bleus.shape[0]
    # auc_uns = 1.0 - auc(percentages / 100.0, bleus[::-1] / 100.0)

    random_bleu = np.asarray(
        [base_bleu * (1.0 - float(i) / float(n_items)) + 100.0 * (float(i) / float(n_items)) for i in range(n_items)],
        dtype=np.float32)

    sorted_sbleus = np.sort(sbleus)
    perfect_bleu = np.asarray(
        [np.mean(np.concatenate((max_sbleus[:i], sorted_sbleus[i:]), axis=0)) for i in range(len(hypos))])

    auc_rnd = 1.0 - auc(percentages / 100.0, random_bleu / 100.0)
    auc_uns = 1.0 - auc(percentages / 100.0, bleus / 100.0)
    auc_orc = 1.0 - auc(percentages / 100.0, perfect_bleu / 100.0)
    rejection_ratio = (auc_uns - auc_rnd) / (auc_orc - auc_rnd) * 100.0

    return bleus, random_bleu, percentages, perfect_bleu, rejection_ratio


def load_text(path, beam_width=5):
    refs, hypos = [], []
    with open(os.path.join(path, 'refs.txt'), 'r') as f:
        for line in f.readlines():
            refs.append(line[1:-1])

    with open(os.path.join(path, 'hypos.txt'), 'r') as f:
        count = 0
        for line in f.readlines():
            if count % beam_width == 0:
                hypos.append(line[1:-1])
            count += 1

    return refs, hypos


def main():
    args = parser.parse_args()

    uncertainties = load_uncertainties(args.path, beam_width=args.beam_width, n_best=args.nbest)
    out_dict = {}
    refs, hypos = load_text(args.path, beam_width=args.beam_width)
    for key in uncertainties.keys():
        bleus, random_bleu, percentage, perfect_bleu, auc_rr = reject_predictions_bleu(refs, hypos,
                                                                                       uncertainties[key],
                                                                                       args.path)
        out_dict[key] = [bleus, percentage, random_bleu, perfect_bleu, auc_rr]
    plt.plot(out_dict[key][1], out_dict[key][3], 'r--', lw=2)
    for key in out_dict.keys():
        if key == 'SCR-PE':
            plt.plot(out_dict[key][1], out_dict[key][0])
    plt.plot(out_dict[key][1], out_dict[key][2], 'k--', lw=2)
    plt.ylim(out_dict[key][2][0], 100.0)
    plt.xlim(0.0, 100.0)
    plt.xlabel('Percentage Rejected')
    plt.ylabel('Bleu')
    plt.legend(['Oracle', 'Joint-Seq TU', 'Expected Random'])
    plt.savefig(os.path.join(args.path, 'seq_reject.png'), bbox_inches='tight', dpi=300)
    plt.close()

    results = 'results_seq_mc.txt'
    with open(os.path.join(args.path, results), 'a') as f:
        f.write(f'--SEQ ERROR DETECT N-BEST  {args.nbest} BW: {args.beam_width}, BS:{args.beam_search}--\n')
        for key in out_dict.keys():
            f.write('AUC-RR using ' + key + ": " + str(np.round(out_dict[key][-1], 1)) + '\n')


if __name__ == '__main__':
    main()
