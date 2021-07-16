"""
Metrics utilities used in loss computation during training,
retention curve analysis (c.f. `plot_retention_curves.py`,
and metadata collection over the full dataset (c.f. `sdc/cache_metadata.py`).

Nomenclature (used in docstrings below):

B: number of prediction requests in a minibatch. Note that
    prediction_request != scene, because each scene generally has many
    prediction requests, i.e., one for each vehicle that is visible for
    all 25 future timesteps.
M: number of datapoints in an evaluation dataset (hence the number of total
    prediction requests).
D_i: number of predicted modes for the prediction request i.
    There can be a maximum of 25 modes predicted for each prediction request,
    and the number can vary for each prediction request.

See a description of the retention task in the README.
"""

import datetime
import os
from collections import defaultdict
from typing import Sequence, Union, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from sdc.assessment import f_beta_metrics, calc_uncertainty_regection_curve
from sdc.constants import (
    VALID_AGGREGATORS, VALID_BASE_METRICS)
from ysdc_dataset_api.evaluation.metrics import (
    average_displacement_error, final_displacement_error,
    aggregate_prediction_request_losses, _softmax_normalize)


class SDCLoss:
    def __init__(self, full_model_name, c):
        # Softmax used to normalize per-plan confidences for a particular
        # prediction request
        self.softmax = _softmax_normalize

        # Store ADE/FDE of all M * D_i preds
        self.base_metric_to_losses = defaultdict(list)
        
        # Will ultimately have shape (M, D_i), where
        #   M = number of datapoints in evaluation set
        #   D_i = number of predictions per scene, which can vary per scene i
        self.plan_confidence_scores = []
        self.pred_request_confidence_scores = []  # Shape (M,)

        # Path where retention (both standard and fbeta) results are stored
        metrics_dir = (
            f'{c.dir_data}/metrics' if not c.dir_metrics else c.dir_metrics)
        os.makedirs(metrics_dir, exist_ok=True)
        self.metrics_dir = os.path.join(metrics_dir, full_model_name)
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Name of the model,
        # e.g., rip-dim-k_3-plan_wcm-scene_bcm would correspond to a
        # RIP ensemble with:
        # * The DIM backbone model (see `sdc.oatomobile.torch.baselines`)
        # * K = 3 members
        # * Worst-case aggregation for plan confidence
        # * Best-case aggregation for scene confidence
        # See `sdc.oatomobile.torch.baselines.robust_imitative_planning.py`
        # for all aggregation strategies.
        self.model_name = full_model_name

        # Seeds used in evaluating the current model.
        self.np_seed = c.np_seed
        self.torch_seed = c.torch_seed

        # Model prefix -- used for convenience to differentiate runs, e.g.,
        # those trained on a subset or all of the training data
        self.model_prefix = c.model_prefix

        # The error threshold below which we consider the prediction acceptable
        self.fbeta_threshold = c.fbeta_threshold
        self.fbeta_beta = c.fbeta_beta

    def clear_per_dataset_attributes(self):
        self.base_metric_to_losses = defaultdict(list)
        self.plan_confidence_scores = []
        self.pred_request_confidence_scores = []

    def construct_dataset_losses_and_confidence_scores(self):
        """
        Aggregates losses and confidence scores that were accumulated
        over batches of an evaluation dataset.
        """
        assert self.plan_confidence_scores is not None
        assert self.pred_request_confidence_scores is not None
        plan_array_dims = len(self.plan_confidence_scores[0].shape)
        try:
            if plan_array_dims == 1:
                self.plan_confidence_scores = np.stack(
                    self.plan_confidence_scores, axis=0)
            elif plan_array_dims == 2:
                self.plan_confidence_scores = np.concatenate(
                    self.plan_confidence_scores, axis=0)
            else:
                raise ValueError(
                    f'Unexpected number of dims in `plan_confidence_scores`: '
                    f'{plan_array_dims}.')
        except RuntimeError:
            # Don't need to do a concatenation
            print('Detected differing number of predicted '
                  'plans for each scene.')
            pass

        self.pred_request_confidence_scores = np.concatenate(
            self.pred_request_confidence_scores, axis=0)

        base_metric_to_losses = {}
        for base_metric, losses_list in self.base_metric_to_losses.items():
            try:
                base_metric_to_losses[base_metric] = np.stack(
                    losses_list, axis=0)
            except RuntimeError:
                base_metric_to_losses[base_metric] = losses_list

        self.base_metric_to_losses = base_metric_to_losses

    def evaluate_dataset_losses(self, dataset_key: str):
        self.construct_dataset_losses_and_confidence_scores()

        assert len(self.base_metric_to_losses['ade']) == len(
            self.base_metric_to_losses['fde'])

        result_dicts = []

        # ** RIP evaluation **

        # * Default Method *
        # Evaluate retention (standard and fbeta) using
        # uncertainty estimates of model
        uncertainty_scores = (-self.pred_request_confidence_scores)
        default_fbeta_aucs, default_r_aucs, default_retention_arrs = (
            self.collect_retention_and_fbeta_metrics(
                uncertainty_scores=uncertainty_scores,
                model_name='Default'))
        result_dicts += [default_fbeta_aucs, default_r_aucs]

        # Store retention curve
        self.store_retention_metrics(
            model_name=self.model_name,
            retention_arrs=default_retention_arrs,
            dataset_key=dataset_key)

        # * Random Method *
        # Randomly sorting prediction request uncertainty scores provides a
        # baseline for the model with the same predictions
        # but uninformative uncertainty
        random_uncertainties = np.random.permutation(
            self.pred_request_confidence_scores.shape[0])
        random_fbeta_aucs, random_r_aucs, random_retention_arrs = (
            self.collect_retention_and_fbeta_metrics(
                uncertainty_scores=random_uncertainties,
                model_name='Random'))
        result_dicts += [random_fbeta_aucs, random_r_aucs]
        self.store_retention_metrics(
            model_name=f'{self.model_name}-Random',
            retention_arrs=random_retention_arrs,
            dataset_key=dataset_key)

        # * Optimal Method *
        # Use the loss as the uncertainty, giving us the performance
        # of the model if it had perfectly calibrated uncertainty
        optimal_fbeta_aucs, optimal_r_aucs, optimal_retention_arrs = (
            self.collect_retention_and_fbeta_metrics(
                uncertainty_scores=None,  # triggers using losses as uncert
                model_name='Optimal'))
        result_dicts += [optimal_fbeta_aucs, optimal_r_aucs]
        self.store_retention_metrics(
            model_name=f'{self.model_name}-Optimal',
            retention_arrs=optimal_retention_arrs,
            dataset_key=dataset_key)

        final_metrics = {}
        for result_dict in result_dicts:
            for key, value in result_dict.items():
                final_metrics[key] = value

        self.clear_per_dataset_attributes()
        return final_metrics

    def cache_batch_losses(
        self,
        predictions_list: Union[Sequence[np.ndarray], np.ndarray],
        ground_truth_batch: np.ndarray,
        plan_confidence_scores_list: Union[
            Sequence[np.ndarray], np.ndarray] = None,
        pred_request_confidence_scores: np.ndarray = None,
    ):
        """Precomputes and caches the batch ADE and FDE for each of the D_b
        predictions in each of the b in B scenes of the minibatch.
        
        Also stores plan_ and pred_request_confidence_scores if provided.

        Args:
            predictions_list:
                if List[np.ndarray]: length is B, the number of prediction
                    requests in a batch.
                    Each array has shape (D_b, T, 2) where D_b is the number
                    of trajectory predictions, which can vary for each
                    prediction request..
                if np.ndarray: we have a consistent number of predictions D
                    per prediction request. Shape (B, D, T, 2)
            ground_truth_batch: np.ndarray, shape (B, T, 2), there is only
                 one ground_truth trajectory for each prediction request.
            plan_confidence_scores_list: length is B, the number of scenes in a 
                batch. Similar to predictions_list, is a list if there are 
                a different number of predictions per prediction request, in
                which case each entry has shape (D_b,).
            pred_request_confidence_scores: shape (B,), model confidence scores
                on each prediction request.
        """
        # Do this instead of enumerate to handle np.ndarray
        for batch_index in range(len(predictions_list)):
            # predictions shape: (D_b, T, 2), where D_b can vary per
            # prediction request
            predictions = predictions_list[batch_index]
            ground_truth = np.expand_dims(
                ground_truth_batch[batch_index], axis=0)

            for base_metric in VALID_BASE_METRICS:
                if base_metric == 'ade':
                    base_metric_fn = average_displacement_error
                elif base_metric == 'fde':
                    base_metric_fn = final_displacement_error
                else:
                    raise NotImplementedError

                self.base_metric_to_losses[base_metric].append(
                    base_metric_fn(
                        ground_truth=ground_truth, predicted=predictions))

        if plan_confidence_scores_list is not None:
            if isinstance(plan_confidence_scores_list, list):
                self.plan_confidence_scores += plan_confidence_scores_list
            elif isinstance(plan_confidence_scores_list, np.ndarray):
                self.plan_confidence_scores.append(
                    plan_confidence_scores_list)
            else:
                raise ValueError(
                    f'Unexpected type for plan_confidence_scores_list: '
                    f'{type(plan_confidence_scores_list)}')

        if pred_request_confidence_scores is not None:
            self.pred_request_confidence_scores.append(
                pred_request_confidence_scores)

    def collect_retention_and_fbeta_metrics(
        self,
        uncertainty_scores: Optional[np.ndarray],
        model_name: str,
    ) -> Tuple[Dict, Dict, Dict]:
        """Computes:
                - retention curve
                - fbeta rejection curve (along with auc and fbeta at 95%)
            using general Shifts Challenge assessment API.

        Uses per--prediction request uncertainty scores and all
            pairwise combinations of aggregation (min, avg, weighted, top1)
            and ADE/FDE.

        NOTE: we expect per--prediction request `uncertainty_scores`, i.e.,
            the score should be higher for a point on which the model is more
            uncertain.
            These can be obtained by, for example, taking the mean of per-plan
            confidence scores and negating.

        Args:
            uncertainty_scores: np.ndarray, accompanying uncertainty scores.
                See note above. At each possible retention threshold, we
                retain the proportion of the dataset with lowest uncertainty
                and compute metrics as normal.
                If this is not provided, we assume that we are using the losses
                as the uncertainties, i.e., computing an optimal baseline.
            model_name: str, used for storing fbeta retention curves.
                e.g., we store for Random/Optimal/Default methods.
        Returns:
            Tuple[Dict, Dict, Dict]
            First Dict (fbeta auc, f95 values)
                key: str, e.g., minADE_fbeta_auc
                val: float, fbeta AUC or F95 for given metric
            Second Dict (retention auc values):
                key: str, e.g., minADE_r_auc
                val: float, retention AUC for given metric
            Third Dict (full retention curves)
                key: str, e.g., minADE
                val: np.ndarray, retention curve
        """
        if uncertainty_scores is None:
            compute_optimal_baseline = True
            print('Computing Optimal retention baseline.')
        else:
            compute_optimal_baseline = False

        M = self.pred_request_confidence_scores.shape[0]
        fbeta_aucs = {}
        r_aucs = {}
        retention_arrs = {}

        # Create directory to store retention arrays
        retention_dir = 'fbeta_retention'
        retention_dir = os.path.join(self.metrics_dir, retention_dir)
        os.makedirs(retention_dir, exist_ok=True)

        for base_metric in VALID_BASE_METRICS:
            assert base_metric in {'ade', 'fde'}

            for aggregator in VALID_AGGREGATORS:
                metric_key_prefix = f'{aggregator}{base_metric.upper()}'
                per_plan_losses = self.base_metric_to_losses[base_metric]
                per_pred_req_losses = []

                # Do this instead of enumerate to handle np.ndarray
                for datapoint_index in range(M):
                    datapoint_losses = per_plan_losses[datapoint_index]
                    datapoint_weights = self.softmax(
                        self.plan_confidence_scores[datapoint_index])
                    agg_prediction_loss = (
                        aggregate_prediction_request_losses(
                            aggregator=aggregator,
                            per_plan_losses=datapoint_losses,
                            per_plan_weights=datapoint_weights))
                    per_pred_req_losses.append(agg_prediction_loss)

                per_pred_req_losses = np.array(per_pred_req_losses)

                if compute_optimal_baseline:
                    # Use the loss as uncertainty, i.e., perfectly calibrated
                    # uncertainty estimates
                    uncertainty_scores = per_pred_req_losses

                # Compute fbeta results and store the arrays.
                f_auc, f95, retention = f_beta_metrics(
                    errors=per_pred_req_losses,
                    uncertainty=uncertainty_scores,
                    threshold=self.fbeta_threshold,
                    beta=self.fbeta_beta)
                fbeta_aucs[f'{model_name}__{metric_key_prefix}__f_auc'] = f_auc
                fbeta_aucs[f'{model_name}__{metric_key_prefix}__f95'] = f95
                fbeta_retention_path = (
                    f'{retention_dir}/{model_name}__{metric_key_prefix}')
                print(f'Stored fbeta retention results to '
                      f'{fbeta_retention_path}.')
                np.save(fbeta_retention_path, arr=retention)

                # Compute retention/rejection arrays.
                rejection_curve = calc_uncertainty_regection_curve(
                    errors=per_pred_req_losses,
                    uncertainty=uncertainty_scores)[::-1]
                r_auc = rejection_curve.mean()
                r_aucs[f'{model_name}__{metric_key_prefix}__r_auc'] = r_auc
                retention_arrs[metric_key_prefix] = rejection_curve

        return fbeta_aucs, r_aucs, retention_arrs

    def store_retention_metrics(
        self,
        model_name: str,
        retention_arrs: Dict[str, np.ndarray],
        dataset_key: str
    ):
        """
        Code based on Deferred Prediction utilities due to
            Neil Band and Angelos Filos
            https://github.com/google/uncertainty-baselines

        Parses a dict of retention arrays.
        Stores results to a DataFrame at the specified path
            (or updates a DataFrame that already exists at the path).

        Args:
            model_name: `str`, name of the model for which we are storing
                metrics.
            retention_arrs: `Dict`, metrics computed at all possible retention
                thresholds.
            dataset_key: `str`, key denoting the name of the evaluation dataset

        Returns:
          `Optional[Dict]`
        """
        # Get the length of a retention array, confirm all are same length
        first_key = next(iter(retention_arrs))
        ret_arr_len = len(retention_arrs[first_key])
        for _, arr in retention_arrs.items():
            assert len(arr) == ret_arr_len

        # Tile for the number of retention metrics
        retention_thresholds = np.arange(ret_arr_len) / float(ret_arr_len)
        retention_thresholds_expanded = (
            retention_thresholds.tolist() * len(retention_arrs))

        retention_values = []
        metric_names = []
        for metric_key, arr in retention_arrs.items():
            metric_names += [metric_key] * len(arr)
            retention_values.append(arr)

        data = {
            'metric': metric_names,
            'retention_threshold': retention_thresholds_expanded,
            'value': np.concatenate(retention_values, axis=0)}
        new_results_df = pd.DataFrame(data)

        # Add metadata
        new_results_df['dataset_key'] = dataset_key
        new_results_df['model_prefix'] = self.model_prefix
        new_results_df['model_name'] = model_name
        new_results_df['eval_seed'] = (
            f'np_{self.np_seed}__torch_{self.torch_seed}')
        new_results_df['run_datetime'] = datetime.datetime.now()
        new_results_df['run_datetime'] = pd.to_datetime(
            new_results_df['run_datetime'])

        model_results_path = os.path.join(self.metrics_dir, 'results.tsv')

        # Update or initialize results DataFrame
        try:
            with open(model_results_path, 'r') as f:
                previous_results_df = pd.read_csv(f, sep='\t')
            results_df = pd.concat([previous_results_df, new_results_df])
            action_str = 'updated'
        except FileNotFoundError:
            print(f'No previous results found at path {model_results_path}. '
                  f'Storing a new metrics dataframe.')
            results_df = new_results_df
            action_str = 'stored initial'

        # Store to file
        with open(model_results_path, 'w') as f:
            results_df.to_csv(path_or_buf=f, sep='\t', index=False)

        print(
            f'Successfully {action_str} results dataframe '
            f'at {model_results_path}.')
