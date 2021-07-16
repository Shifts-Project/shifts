"""Utilities for plotting retention curves.

Script adapted from Uncertainty Baselines project,
    code due to Neil Band and Angelos Filos.
    (see https://github.com/google/uncertainty-baselines for full citation).
"""

import os
from itertools import product
from typing import Optional, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from absl import app
from absl import flags
from absl import logging

from sdc.assessment import calc_uncertainty_regection_curve, f_beta_metrics
from sdc.constants import BASELINE_TO_COLOR_HEX

# Data load / output flags.
flags.DEFINE_string(
    'model_name', None,
    'Should be specified if `results_dir` points to the subdirectory of a '
    'particular model name, as created by RIP evaluation.'
    'Otherwise, subdirectories corresponding to the model names '
    'will be located.')
flags.DEFINE_string(
    'results_dir',
    'metrics/',
    'The directory where the retention results are stored. See '
    'the `plot_retention_score_results` method for details on this argument.')
flags.DEFINE_string(
    'plot_dir', '/metrics',
    'The directory to which the plots are written.')
FLAGS = flags.FLAGS


def get_plotting_style_metric_name(metric_name):
    return metric_name.split('_')[0]


def get_sparsification_factor(arr_len):
    """Determine with which multiple we
    subsample the array (for easier plotting)."""
    sparsification_factor = None
    if arr_len > 100000:
        sparsification_factor = 1000
    elif arr_len > 10000:
        sparsification_factor = 100
    elif arr_len > 1000:
        sparsification_factor = 10

    return sparsification_factor


def construct_model_name_helper(model_prefix, full_name, auc_mean, auc_std):
    auc_mean = np.round_(auc_mean, 3).item()
    auc_std = np.round_(auc_std, 3).item()
    if model_prefix != 'Default':
        full_name = f'{model_prefix} | {full_name}'

    name_to_print = f'{full_name} [AUC: {auc_mean}'
    if np.isnan(auc_std):
        name_to_print += f']'
    else:
        name_to_print += f'± {auc_std}]'

    return name_to_print


def get_plotting_style_model_name(model_prefix, model_name, auc_mean, auc_std):
    try:
        (model_class, backbone_name, k_details, plan_algo,
            scene_algo, rand_or_opt) = model_name.split('-')
    except ValueError:
        (model_class, backbone_name, k_details, plan_algo,
            scene_algo) = model_name.split('-')
        rand_or_opt = None

    k = k_details.split('_')[-1]
    if model_class != 'rip':
        raise NotImplementedError
    plan_algo = plan_algo.split('_')[-1]
    scene_algo = scene_algo.split('_')[-1]

    if rand_or_opt is not None:
        model_name = (
            f'RIP ({backbone_name}, K={k}, Plan {plan_algo.upper()}, '
            f'Scene {scene_algo.upper()}) {rand_or_opt}')
    else:
        model_name = (
            f'RIP ({backbone_name}, K={k}, Plan {plan_algo.upper()}, '
            f'Scene {scene_algo.upper()})')

    return construct_model_name_helper(
        model_prefix, model_name, auc_mean, auc_std)


def get_results_from_model_dir(model_dir: str):
    """Get results from a subdir.

    Args:
      model_dir: `str`, subdirectory that contains a `results.tsv` file for the
        corresponding model name.

    Returns:
      Results pd.DataFrame, or None, from a subdirectory containing
      results from a particular model name.
    """
    model_results_path = os.path.join(model_dir, 'results.tsv')
    if not os.path.isfile(model_results_path):
        return None

    logging.info('Found results at %s.', model_results_path)

    with open(model_results_path, 'r') as f:
        return pd.read_csv(f, sep='\t')


def plot_retention_score_results(
    results_dir: str,
    plot_dir: str,
    model_name: Optional[str] = None
):
    """Load retention score results from the specified directory.

    Args:
      results_dir: `str`, directory from which deferred prediction results are
        loaded. If you aim to generate a plot for multiple models, this should
        point to a directory in which each subdirectory has a name corresponding
        to a model.
        Otherwise, if you aim to generate a plot for a particular
        model, `results_dir` should point directly to a model type's
        subdirectory, and the `model_name` argument should be provided.
      plot_dir: `str`, directory to which plots will be written
      model_name: `str`, should be provided if generating a plot for only one
        particular model.
    """
    if model_name is None:
        dir_path, child_dir_suffixes, _ = next(os.walk(results_dir))
        model_dirs = []
        results_dfs = []

        for child_dir_suffix in child_dir_suffixes:
            try:
                model_name = child_dir_suffix.split('/')[0]
            except:  # pylint: disable=bare-except
                continue

            model_dir = os.path.join(dir_path, model_name)
            logging.info(
                'Found retention results directory for model %s at %s.',
                model_name, model_dir)
            model_dirs.append(model_dir)

        for model_dir in model_dirs:
            results = get_results_from_model_dir(model_dir)
            if results is not None:
                results_dfs.append(results)

        results_df = pd.concat(results_dfs, axis=0)
    else:
        logging.info(
            'Plotting retention results for model %s.', model_name)
        model_results_path = os.path.join(results_dir, 'results.tsv')
        try:
            with open(model_results_path, 'r') as f:
                results_df = pd.read_csv(f, sep='\t')
        except FileNotFoundError:
            raise FileNotFoundError(
                f'No results found at path {model_results_path}.')

    plot_results_df(results_df, plot_dir)


def plot_results_df(results_df: pd.DataFrame, plot_dir: str):
    """Creates a retention plot for each metric in the results_df.

    For a particular model, aggregates metric values (e.g., minADE)
    when retain proportion is identical.

    If retain proportion and eval seed are identical for a
    particular model and metric, we consider only the most recent result.
    Args:
     results_df: `pd.DataFrame`, expects columns: {metric, retention_threshold,
       value, model_name, eval_seed, run_datetime}.
     plot_dir: `str`, directory to which plots will be written
    """
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    dataset_keys = list(sorted(set(results_df['dataset_key'])))
    metrics = set(results_df['metric'])

    outside_loop = product(dataset_keys, metrics)

    # We do a plot for each of these.
    for dataset_key, metric in outside_loop:
        metric_df = results_df[
            (results_df['dataset_key'] == dataset_key) &
            (results_df['metric'] == metric)].copy()

        fig, ax = plt.subplots()
        plotting_metric_name = get_plotting_style_metric_name(metric)

        model_prefixes = set(results_df['model_prefix'])
        baselines = list(sorted(set(metric_df['model_name'])))
        inner_loop = product(baselines, model_prefixes)
        for b, (baseline, model_prefix) in enumerate(inner_loop):
            baseline_metric_df = metric_df[
                (metric_df['model_name'] == baseline) &
                (metric_df['model_prefix'] == model_prefix)].copy()

            # Sort by datetime (newest first)
            baseline_metric_df.sort_values(
                'run_datetime', inplace=True, ascending=False)

            # For a particular baseline model and metric, drop duplicates
            # (keeping the newest entry)
            # if the retain proportion and eval seed are identical
            baseline_metric_df.drop_duplicates(
                subset=['retention_threshold', 'eval_seed'],
                keep='first',
                inplace=True)

            # Group by retention_threshold, and
            # compute the mean and standard deviation of the metric
            agg_baseline_metric_df = baseline_metric_df.groupby(
                'retention_threshold').value.agg(['mean', 'std']).reset_index()
            retained_data = agg_baseline_metric_df['retention_threshold']
            mean = agg_baseline_metric_df['mean']
            std = agg_baseline_metric_df['std']

            # Compute area under the curve
            # We assume evenly spaced retention thresholds
            auc_mean = np.mean(mean)
            auc_std = np.mean(std)

            # Sparsify the arrays for easier display if they are large
            sparsification_factor = get_sparsification_factor(
                len(retained_data))
            if sparsification_factor is not None:
                retained_data = retained_data[::sparsification_factor]
                mean = mean[::sparsification_factor]
                std = std[::sparsification_factor]

            # Visualize mean with standard error
            ax.plot(
                retained_data,
                mean,
                label=get_plotting_style_model_name(
                    model_prefix, baseline, auc_mean, auc_std),
                color=colors[b % len(colors)])
            ax.fill_between(
                retained_data,
                mean - std,
                mean + std,
                color=colors[b % len(colors)],
                alpha=0.25)
            ax.set(xlabel='Retention Fraction',
                   ylabel=plotting_metric_name)
            ax.legend()
            fig.tight_layout()

        if isinstance(plot_dir, str):
            os.makedirs(plot_dir, exist_ok=True)
            metric_plot_path = os.path.join(
                plot_dir,
                f'{dataset_key} - {plotting_metric_name}.pdf')
            fig.savefig(metric_plot_path, transparent=True, dpi=300,
                        format='pdf')
            logging.info('Saved plot for metric %s, baselines %s '
                         'to %s.', metric, baselines, metric_plot_path)
        else:
            print(f'{dataset_key} - {plotting_metric_name}')
            plt.show()


def plot_retention_curve_with_baselines(
    uncertainty_scores: np.ndarray,
    losses: np.ndarray,
    model_key: str,
    metric_name: str = 'weightedADE'
):
    """
    Plot a retention curve with Random and Optimal baselines.

    Assumes that `uncertainty_scores` convey uncertainty (not confidence)
    for a particular point.

    Args:
        uncertainty_scores: np.ndarray, per--prediction request uncertainty
            scores (e.g., obtained by averaging per-plan confidence scores
            and negating).
        losses: np.ndarray, array of loss values, e.g., weightedADE.
        model_key: str, used for displaying the model configurations in the
            plot legend.
        metric_name: str, retention metric, displayed on y-axis.
    """
    M = len(losses)

    methods = ['Random', 'Baseline', 'Optimal']
    aucs_retention_curves = []

    # Random results
    random_indices = np.arange(M)
    np.random.shuffle(random_indices)
    retention_curve = calc_uncertainty_regection_curve(
        errors=losses, uncertainty=random_indices)
    aucs_retention_curves.append((retention_curve.mean(), retention_curve))
    print('Computed Random curve.')

    # Get baseline results
    retention_curve = calc_uncertainty_regection_curve(
        errors=losses, uncertainty=uncertainty_scores)
    aucs_retention_curves.append((retention_curve.mean(), retention_curve))
    print('Computed Model curve.')

    # Optimal results
    retention_curve = calc_uncertainty_regection_curve(
        errors=losses, uncertainty=losses)
    aucs_retention_curves.append((retention_curve.mean(), retention_curve))
    print('Computed Optimal curve.')

    plt.rcParams.update({'font.size': 12})

    fig, ax = plt.subplots()
    for b, (baseline, (auc, retention_values)) in enumerate(
            zip(methods, aucs_retention_curves)):
        color = BASELINE_TO_COLOR_HEX[baseline]
        if baseline == 'Baseline':
            baseline = ''

        # Subsample the retention value,
        # as there are likely many of them
        # (on any of the dataset splits)
        sparsification_factor = get_sparsification_factor(
            retention_values.shape[0])
        retention_values = retention_values[::sparsification_factor][::-1]
        retention_thresholds = np.arange(
            len(retention_values)) / len(retention_values)

        ax.plot(
            retention_thresholds,
            retention_values,
            label=f'RIP ({model_key}) {baseline} [AUC: {auc:.3f}]',
            color=color)

    ax.set(xlabel='Retention Fraction', ylabel=metric_name)
    ax.legend()
    fig.tight_layout()
    sns.set_style('darkgrid')
    plt.show()
    return fig


def plot_retention_curves_from_dict(
    model_key_to_auc_and_curve: Dict[str, Tuple[float, np.ndarray]],
    metric_name: str = 'weightedADE'
):
    """
    Plot a set of retention curves.

    Args:
        model_key_to_auc_and_curve: Dict[str, Tuple[float, np.ndarray]],
            mapping from a model key to retention results.
        metric_name: str, retention metric, displayed on y-axis.
    """
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots()

    for b, (model_key, (auc, retention_values)) in enumerate(
            model_key_to_auc_and_curve.items()):
        # Subsample the retention value,
        # as there are likely many of them
        # (on any of the dataset splits)
        sparsification_factor = get_sparsification_factor(
            retention_values.shape[0])
        retention_values = retention_values[::sparsification_factor][::-1]
        retention_thresholds = np.arange(
            len(retention_values)) / len(retention_values)

        ax.plot(
            retention_thresholds,
            retention_values,
            label=f'RIP ({model_key}) [AUC: {auc:.3f}]',
            color=colors[b])

    ax.set(xlabel='Retention Fraction', ylabel=metric_name)
    ax.legend()
    fig.tight_layout()
    sns.set_style('darkgrid')
    plt.show()
    return fig


def plot_fbeta_retention_curve_with_baselines(
    results_dict: Dict[str, Tuple[float, float, np.array]],
    model_key: str,
    metric_name: str
):
    """Plot fbeta retention curve with Random and Optimal baselines.

    Args:
        results_dict: Dict[str, Tuple[float, float, np.array]],
            fbeta retention results as produced by
                `sdc.analyze_metadata.f1_retention_baseline_results`.
        model_key: str, used for displaying the model configurations in the
            plot legend.
        metric_name: str, retention metric, displayed on y-axis.
    """
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 12})

    for b, (baseline, retention_results) in enumerate(results_dict.items()):
        f_auc, f95, retention_arr = retention_results
        color = BASELINE_TO_COLOR_HEX[baseline]
        if baseline == 'Baseline':
            baseline = ''

        x = np.arange(len(retention_arr))
        x = x / len(retention_arr)
        ax.plot(
            x,
            retention_arr,
            label=f'RIP ({model_key}) {baseline} [AUC: {f_auc:.3f}]',
            color=color)

        ax.set(xlabel='Retention Fraction',
               ylabel=f'F1-{metric_name}')
        ax.legend()
        fig.tight_layout()

    sns.set_style('darkgrid')
    plt.show()
    return fig


def get_comparative_f1_retention_results(
    baseline_name_to_df,
    metric_name,
):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots()
    methods = []
    aucs_retention_curves = []

    for baseline_name, dataset_df in baseline_name_to_df.items():
        errors = dataset_df[metric_name].to_numpy()
        uncertainty = -(
            dataset_df['pred_request_confidence_scores'].to_numpy())
        threshold = 1
        beta = 1

        # Using our model's uncertainty values
        retention_results = f_beta_metrics(
            errors=errors, uncertainty=uncertainty, threshold=threshold,
            beta=beta)

        f_auc, f95, retention_arr = retention_results
        aucs_retention_curves.append((f_auc, retention_arr))
        methods.append(baseline_name)

    x = np.arange(len(retention_arr))
    x = x / len(retention_arr)

    for b, (baseline, (f_auc, retention_arr)) in enumerate(
            zip(methods, aucs_retention_curves)):
        ax.plot(
            x,
            retention_arr,
            label=f'{baseline} [AUC: {f_auc:.3f}]',
            color=colors[b % len(colors)]
        )

        ax.set(xlabel='Retention Fraction',
               ylabel=f'F1-{metric_name}')
        ax.legend()
        fig.tight_layout()

    plt.show()
    return fig


def main(argv):
    del argv  # unused arg
    logging.info(
        'Looking for Retention Score results at path %s.', FLAGS.results_dir)

    os.makedirs(FLAGS.plot_dir, exist_ok=True)
    logging.info('Saving Retention Score plots to %s.', FLAGS.plot_dir)

    plot_retention_score_results(
        results_dir=FLAGS.results_dir,
        plot_dir=FLAGS.plot_dir,
        model_name=FLAGS.model_name)


if __name__ == '__main__':
    app.run(main)
