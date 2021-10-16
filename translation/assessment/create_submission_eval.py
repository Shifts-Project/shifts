import argparse
import os
import json
import numpy as np


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


parser = argparse.ArgumentParser(description='Assess translation performance')
parser.add_argument('path', type=str,
                    help='Path of directory containing in-domain data.')
parser.add_argument('--domain_path', type=str, default=None,
                    help='Path of file containing domain labels.')
parser.add_argument('--save_path', type=str,  default='./submission.json',
                    help='Path to where to save output.')
parser.add_argument('--beam_width', type=int, default=5,
                    help='Path of directory where to save results.')
parser.add_argument('--nbest', type=int, default=5,
                    help='Path of directory where to save results.')
parser.add_argument('--ensemble', action='store_true',
                    help='Whether to load in additional ensemble-based measures.')
parser.add_argument('--uncertainty_metric', type=str, default='SCR-PE',
                    help='Whether to load in additional ensemble-based measures.')
parser.add_argument('--create_ref', action='store_true',
                    help='Whether to create a reference JSON - not for challenge participants.')


def main():
    args = parser.parse_args()

    # Load refs and hypos
    refs, hypos, ids, nlls = load_text(args.path, beam_width=args.beam_width)

    weights = np.exp(-nlls)
    weights /= weights.sum(axis=1, keepdims=True)

    if args.ensemble:
        uncertainties = load_uncertainties(args.path, beam_width=args.beam_width, n_best=args.nbest)
        uncertainties = uncertainties[args.uncertainty_metric]
    else:
        uncertainties = {'NLL': np.mean(nlls, axis=1)}

    encoder = json.JSONEncoder(ensure_ascii=False)

    if args.create_ref:
        domain_labels = np.loadtxt(args.domain_path)
        with open(args.save_path, 'w') as jp, open('refs_eval.json', 'w') as jr:
            for idx, ref, hypo, un, dl, weight in zip(ids, refs, hypos, uncertainties, domain_labels, weights):
                hypo = [{'text': t, 'confidence': float(c)} for t, c in zip(hypo, weight)]
                dct = {'id': int(idx), 'uncertainty': float(un), 'hypos': hypo}
                jsn = encoder.encode(dct)
                jp.write(jsn + '\n')

                dct = {'id': int(idx), 'ref': ref, 'domain': int(dl)}
                jsn = encoder.encode(dct)
                jr.write(jsn + '\n')
    else:
        with open(args.save_path, 'w') as jp:
            for idx, hypo, un, weight in zip(ids, hypos, uncertainties, weights):
                hypo = [{'text': t, 'confidence': float(c)} for t, c in zip(hypo, weight)]
                dct = {'id': int(idx), 'uncertainty': float(un), 'hypos': hypo}
                jsn = encoder.encode(dct)
                jp.write(jsn + '\n')

if __name__ == '__main__':
    main()
