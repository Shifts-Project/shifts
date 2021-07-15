"""
*** Helper Functions: Post-Hoc Analysis ***

Used with the --rip_cache_all_preds=True flag to aggregate/analyze
 predictions, ground-truth values, and per-plan confidence scores.
"""
import numpy as np
from sdc.assessment import f_beta_metrics, calc_uncertainty_regection_curve
from sklearn.metrics import roc_auc_score
from typing import Dict, Any, Optional, Tuple
from ysdc_dataset_api.evaluation.metrics import compute_all_aggregator_metrics
import pandas as pd


def filter_top_d_plans(model_preds, model_conf_scores, d=5):
    top_preds = []
    top_conf_scores = []

    for preds, conf_scores in zip(model_preds, model_conf_scores):
        top_score_indices = np.argsort(conf_scores)[::-1][:d]
        top_preds.append(preds[top_score_indices])
        top_conf_scores.append(conf_scores[top_score_indices])

    return np.stack(top_preds, axis=0), np.stack(top_conf_scores, axis=0)


def get_all_paper_results(data_df, retention_metric_name):
    """
    Get all paper results using a stored data_df (the request_df cached using
    `--debug_collect_dataset_stats=True`).
    """
    dataset_performance_metrics = {}

    metrics = ['minADE', 'weightedADE', 'minFDE', 'weightedFDE']
    for metric in metrics:
        dataset_performance_metrics[metric] = data_df[metric].mean()

    errors = data_df[retention_metric_name].to_numpy()
    uncertainty = -(data_df['pred_request_confidence_scores'].to_numpy())
    threshold = 1
    beta = 1

    # Using our model's uncertainty values
    f_auc, f95, f_ret_curve = f_beta_metrics(
        errors=errors, uncertainty=uncertainty, threshold=threshold, beta=beta)
    dataset_performance_metrics['f_auc'] = f_auc
    dataset_performance_metrics['f95'] = f95
    dataset_performance_metrics['f_ret_curve'] = f_ret_curve

    ret_curve = calc_uncertainty_regection_curve(
        errors=errors, uncertainty=uncertainty)
    r_auc = ret_curve.mean()
    dataset_performance_metrics['r_auc'] = r_auc
    dataset_performance_metrics['ret_curve'] = ret_curve
    return dataset_performance_metrics


def compute_dataset_results(
    k: int,
    d: int,
    plan_agg: str,
    pred_req_agg: str,
    dataset_key: str,
    dataset_key_to_arrs_dict: Dict[str, Dict],
    n_pred_per_model: int = 10,
    retention_column: str = 'weightedADE',
    return_preds_and_scores: bool = False,
    compute_ood_metrics: Optional[bool] = None
):
    """
    Computes results end-to-end using the predictions, per-plan conf scores,
        and ground truths.

    Will consider only the plans / conf scores of the k first models.

    Args:
        k: int, number of ensemble members to consider. Can be used to compare
            performance with varying model size.
        d: int, total number of predictions to sample from the RIP ensemble.
            For results in the paper, we fix this to 5 (at all k).
        plan_agg: str, plan/trajectory aggregation strategy.
        pred_req_agg: str, pred-req aggregation strategy.
        dataset_key: str, dataset for which we are computing results.
        dataset_key_to_arrs_dict: Dict[str, np.ndarray], mapping from
            dataset_key to a dict of predictions, per-plan conf scores, and
            ground truths for some cached RIP model.
            Is in the format loaded using method
                `sdc.cache_metadata.load_dataset_key_to_arrs`.
        n_pred_per_model: int, number of predictions that were sampled from
            each ensemble member during evaluation. This is set in the config
            as rip_samples_per_model (default: 10).
        retention_column: str, metric column on which we perform retention.
        return_preds_and_scores: bool, if True, do not compute losses and
            instead return preds and scores for use with creation of a
            submission protobuf.
        compute_ood_metrics: Optional[bool], determines if we should compute
            ood detection metrics using the per-
    """
    n_plans = int(k * n_pred_per_model)
    model_preds = dataset_key_to_arrs_dict[
                      dataset_key]['predictions'][:, :n_plans, :, :]
    model_conf_scores = dataset_key_to_arrs_dict[
                            dataset_key]['plan_conf_scores'][:, :n_plans, :k]
    print(f'Using {k} models in ensemble.')
    print('Preds shape: ', model_preds.shape)
    print('Conf scores shape: ', model_conf_scores.shape)

    # Aggregate per-plan confidence scores
    per_plan_conf_scores = numpy_run_rip_aggregation(
        algorithm=plan_agg, scores_to_aggregate=model_conf_scores, axis=-1)
    print(f'Aggregated plan confidence scores using aggregator '
          f'{plan_agg} to shape {per_plan_conf_scores.shape}')

    # Get topd plans / corresponding per-plan confidence scores
    top_model_preds, top_model_conf_scores = (
        filter_top_d_plans(
            model_preds=model_preds,
            model_conf_scores=per_plan_conf_scores,
            d=d))
    print(f'Filtered to top {d} plans.')
    print('Top model preds shape:', top_model_preds.shape)
    print('Top model per-plan conf scores shape:', top_model_conf_scores.shape)

    # Get per--prediction request confidence scores
    per_pred_req_conf_scores = numpy_run_rip_aggregation(
        algorithm=pred_req_agg,
        scores_to_aggregate=top_model_conf_scores, axis=-1)
    print(f'Aggregated pred-req confidence scores from scores of the top {d} '
          f'plans using aggregator {pred_req_agg} to shape '
          f'{per_pred_req_conf_scores.shape}.')

    # Compute OOD metrics
    if compute_ood_metrics is None:
        compute_ood_metrics = ('full' in dataset_key)

    if return_preds_and_scores:
        print('Returning predictions and scores.')
        if compute_ood_metrics:
            ood_arr = dataset_key_to_arrs_dict[dataset_key]['is_ood']
        else:
            ood_arr = None

        return (
            top_model_preds, top_model_conf_scores,
            per_pred_req_conf_scores,
            dataset_key_to_arrs_dict[dataset_key]['request_ids'],
            ood_arr)

    # Compute all base metrics
    metrics_dict = compute_all_aggregator_metrics(
        per_plan_confidences=top_model_conf_scores,
        predictions=top_model_preds,
        ground_truth=dataset_key_to_arrs_dict[dataset_key]['gt_trajectories']
    )
    print('Computed all base metrics.')

    metrics_dict['pred_request_confidence_scores'] = per_pred_req_conf_scores
    data_df = pd.DataFrame(data=metrics_dict)
    results = get_all_paper_results(data_df, retention_column)
    print('Computed R-AUC and F1-AUC metrics.')

    if compute_ood_metrics:
        results['ood_roc_auc'] = roc_auc_score(
            y_true=dataset_key_to_arrs_dict[dataset_key]['is_ood'],
            y_score=(-per_pred_req_conf_scores))
        print('Computed OOD detection ROC-AUC.')

    return results


def numpy_run_rip_aggregation(algorithm, scores_to_aggregate, axis=-1):
    if algorithm == 'WCM':
        return scores_to_aggregate.min(axis=axis)
    elif algorithm == 'BCM':
        return scores_to_aggregate.max(axis=axis)
    elif algorithm == 'MA':
        return scores_to_aggregate.mean(axis=axis)
    elif algorithm == 'UQ':
        return (scores_to_aggregate.mean(axis=axis) +
                scores_to_aggregate.std(axis=axis))
    elif algorithm == 'LQ':
        return (scores_to_aggregate.mean(axis=axis) -
                scores_to_aggregate.std(axis=axis))
    else:
        raise NotImplementedError


def f1_retention_baseline_results(
    errors: np.ndarray,
    uncertainty_scores: np.ndarray
) -> Dict[str, Tuple[float, float, np.array]]:
    """Obtain F1 retention results, including Random and Optimal baselines.

    NOTE: we expect per--prediction request `uncertainty_scores`, i.e.,
    the score should be higher for a point on which the model is more
    uncertain.
    These can be obtained by, for example, taking the mean of per-plan
    confidence scores and negating.

    Args:
        errors: np.ndarray, losses for a given dataset, such as weightedADE.
        uncertainty_scores: np.ndarray, accompanying uncertainty scores.
            See note above.
    Returns:
        Dict[str, Tuple[float, float, np.array]]
        Each of the Random, Baseline, and Optimal uncertainty strategies is
        associated with an f_auc, f95, and the F1 retention curve array.
    """
    threshold = 1
    beta = 1

    # Using our model's uncertainty values
    actual_retention_results = f_beta_metrics(
        errors=errors, uncertainty=uncertainty_scores,
        threshold=threshold, beta=beta)

    # Using randomly shuffled uncertainty values (uninformative uncertainty)
    shuffled_uncertainty = uncertainty_scores.copy()
    np.random.shuffle(shuffled_uncertainty)
    random_retention_results = f_beta_metrics(
        errors=errors, uncertainty=shuffled_uncertainty,
        threshold=threshold, beta=beta)

    # Using optimal uncertainty values (i.e., sorted by error)
    opt_retention_results = f_beta_metrics(
        errors=errors, uncertainty=errors, threshold=threshold, beta=beta)

    results_dict = {
        'Random': random_retention_results,
        'Baseline': actual_retention_results,
        'Optimal': opt_retention_results
    }

    return results_dict

