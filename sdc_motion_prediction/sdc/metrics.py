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
from functools import partial
from typing import Sequence, Union, Dict, Tuple

import numpy as np
import pandas as pd

from sdc.constants import (
    VALID_AGGREGATORS, VALID_BASE_METRICS)
from ysdc_dataset_api.evaluation.metrics import (
    average_displacement_error, final_displacement_error,
    aggregate_prediction_request_losses, _softmax_normalize)
from tabular_weather_prediction.assessment import

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

        # We sweep over these retention thresholds, at each threshold retaining
        # the specified proportion of prediction requests with the highest
        # confidence scores.
        self.retention_thresholds = c.metrics_retention_thresholds

        # Path where area under retention curve results should be stored
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

        # If self.use_oracle == True, we assume that all datapoints not
        # retained have 0 loss (i.e., perfect accuracy)
        # Otherwise we disclude them, and report performance of predictions
        # made on retained points only.
        self.use_oracle = c.metrics_retention_use_oracle

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

        # ** RIP evaluation **

        # Sort prediction request confidences by decreasing confidence
        sorted_pred_req_confs = np.argsort(
            self.pred_request_confidence_scores)[::-1]

        # Evaluate metrics for each prediction request retention proportion
        pred_req_retention_metrics = (
            self.evaluate_pred_request_retention_thresholds(
                sorted_pred_req_confs))

        # Store metrics that were computed with RIP
        self.store_retention_metrics(
            model_name=self.model_name,
            metrics_dict=pred_req_retention_metrics, dataset_key=dataset_key,
            return_parsed_dict=False)

        # ** Random Retention evaluation **
        # Randomly sorting prediction request confidences provides a baseline
        # for the model with the same predictions but poor uncertainty
        random_pred_req_confs = np.random.permutation(
            self.pred_request_confidence_scores.shape[0])
        random_retention_metrics = (
            self.evaluate_pred_request_retention_thresholds(
                random_pred_req_confs))
        self.store_retention_metrics(
            model_name=f'{self.model_name}-Random',
            metrics_dict=random_retention_metrics, dataset_key=dataset_key,
            return_parsed_dict=False)

        # ** Optimal Retention evaluation **
        # Sort prediction requests by the loss, giving us the performance
        # of the model if it had perfectly calibrated uncertainty
        optimal_retention_metrics = (
            self.get_pred_request_retention_thresholds_optimal_perf())
        self.store_retention_metrics(
            model_name=f'{self.model_name}-Optimal',
            metrics_dict=optimal_retention_metrics, dataset_key=dataset_key,
            return_parsed_dict=False)

        pred_req_retention_metrics = {
            f'{metric_key}_{retention_threshold}': loss
            for (retention_threshold, metric_key), loss in
            pred_req_retention_metrics.items()
        }

        # Additional evaluation metrics:
        #


        self.clear_per_dataset_attributes()
        return pred_req_retention_metrics

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

    def collect_fbeta_metrics(self):
        """Use general Shifts Challenge assessment API to compute fbeta_auc,
        fbeta_95, and fbeta rejection curve.

        Returns:
            Dict, with
                key: str, e.g., minADE_fbeta_auc
                val: np.array, error metric
        """
        uncertainties = -np.array(self.pred_request_confidence_scores)
        M = uncertainties.shape[0]
        fbeta_metrics = {}

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
                f_auc, f95, retention =
                fbeta_metrics[f'{metric_key_prefix}']




            metric_key = (f'{aggregator}{base_metric.upper()}_'
                          f'retain_scene')

    def evaluate_pred_request_retention_thresholds(
        self,
        sorted_pred_request_confidences: np.ndarray,
    ) -> Dict[Tuple[float, str], np.ndarray]:
        """For various retention thresholds, evaluate all pairwise
            combinations of aggregation (e.g., min, avg, ...) and ADE/FDE.

            Code based on Deferred Prediction utilities due to
                Neil Band and Angelos Filos
                https://github.com/google/uncertainty-baselines

        Args:
            sorted_pred_request_confidences: np.ndarray, per--prediction
                request confidence scores, sorted by decreasing confidence.
                At each retention threshold, we retain the top (X * 100)%
                of the dataset and compute metrics as normal.
        Returns:
          Dict with format key: str = f'metric_retain_scene_{fraction}',
          and value: metric value at given data retained fraction.
        """
        M = len(self.base_metric_to_losses['ade'])
        metrics_dict = {}
        retention_threshold_to_n_points = {}

        # Compute metrics for range of retention thresholds
        for retention_threshold in self.retention_thresholds:
            retain_bool_mask = np.zeros_like(
                sorted_pred_request_confidences)
            n_retained_points = int(M * retention_threshold)
            indices_to_retain = sorted_pred_request_confidences[
                :n_retained_points]

            # These are used for normalization
            # We can reflect our use of the oracle on all non-retained points
            # (importantly, assuming we are using ADE/FDE) by just normalizing
            # over all datapoints (since the losses at non-retained points
            # would be 0).
            if self.use_oracle:
                n_retained_points = M
            retention_threshold_to_n_points[
                retention_threshold] = n_retained_points
            retain_bool_mask[indices_to_retain] = 1
            retain_bool_mask = retain_bool_mask.astype(np.bool_)

            counter = 0

            # Do this instead of enumerate to handle np.ndarray
            for datapoint_index in range(M):
                if not retain_bool_mask[datapoint_index]:
                    continue
                else:
                    counter += 1

                for base_metric in VALID_BASE_METRICS:
                    assert base_metric in {'ade', 'fde'}, (
                        'This method must be updated to handle oracle use + '
                        'any metric for which perfect performance is not 0 '
                        '(e.g., an accuracy).')

                    per_plan_losses = self.base_metric_to_losses[
                        base_metric][datapoint_index]
                    per_plan_confidences = self.plan_confidence_scores[
                        datapoint_index]

                    # Normalize the per-plan confidence scores for a
                    # particular prediction request (i.e., within this scene).
                    per_plan_weights = self.softmax(per_plan_confidences)

                    for aggregator in VALID_AGGREGATORS:
                        metric_key = (f'{aggregator}{base_metric.upper()}_'
                                      f'retain_scene')
                        agg_prediction_loss = (
                            aggregate_prediction_request_losses(
                                aggregator=aggregator,
                                per_plan_losses=per_plan_losses,
                                per_plan_weights=per_plan_weights))
                        if (retention_threshold, metric_key) not in (
                                metrics_dict.keys()):
                            metrics_dict[
                                (retention_threshold, metric_key)] = (
                                agg_prediction_loss)
                        else:
                            metrics_dict[
                                (retention_threshold, metric_key)] += (
                                agg_prediction_loss)

        normed_metrics_dict = {}
        for (retention_threshold, metric_key), loss in metrics_dict.items():
            normed_metrics_dict[(retention_threshold, metric_key)] = (
                loss / retention_threshold_to_n_points[retention_threshold])

        return normed_metrics_dict

    def get_pred_request_retention_thresholds_optimal_perf(
            self) -> Dict[Tuple[float, str], np.ndarray]:
        """Determine the optimal performance on the retention task if we
            used a model with perfectly calibrated confidence (i.e., its
            confidence ordering has perfect Spearman correlation with the
            true ordering of losses for each pairwise combination of
            aggregation (e.g., min, top1, ...) and ADE/FDE.

            Code based on Deferred Prediction utilities due to
                Neil Band and Angelos Filos
                https://github.com/google/uncertainty-baselines

        Returns:
          Dict with format key: str = f'metric_retain_scene_{fraction}',
          and value: metric value at given data retained fraction.
        """
        M = len(self.base_metric_to_losses['ade'])
        metrics_dict = {}
        retention_threshold_to_n_points = {}

        # Here we need to put the metrics and aggregators
        # (i.e., minADE, maxFDE, and other ways to aggregate losses within
        # a scene across proposed plans) in the outer loop.
        # This allows us to sort "perfectly" based on loss -- i.e., what a
        # model with optimally calibrated uncertainty would do, to obtain
        # an upper bound on retention performance with our given
        # model's plan predictions.
        for base_metric in VALID_BASE_METRICS:
            assert base_metric in {'ade', 'fde'}, (
                'This method must be updated to handle oracle use + '
                'any metric for which perfect performance is not 0 '
                '(e.g., an accuracy).')
            per_plan_losses = self.base_metric_to_losses[base_metric]
            for aggregator in VALID_AGGREGATORS:
                metric_key = (f'{aggregator}{base_metric.upper()}_'
                              f'retain_scene')
                metric_agg_losses = np.zeros(M)
                for datapoint_index in range(M):
                    datapoint_losses = per_plan_losses[datapoint_index]
                    datapoint_weights = self.softmax(
                        self.plan_confidence_scores[datapoint_index])
                    metric_agg_losses[datapoint_index] = (
                        aggregate_prediction_request_losses(
                            aggregator=aggregator,
                            per_plan_losses=datapoint_losses,
                            per_plan_weights=datapoint_weights))

                # Now we argsort the losses to get the ideal order
                # i.e., order losses in ascending order (breaks for accuracy)
                # TODO: update for any miss rate - like metric
                sorted_metric_agg_losses = np.argsort(metric_agg_losses)

                # Compute metrics for range of retention thresholds
                for retention_threshold in self.retention_thresholds:
                    retain_bool_mask = np.zeros_like(
                        sorted_metric_agg_losses)
                    n_retained_points = int(M * retention_threshold)
                    indices_to_retain = sorted_metric_agg_losses[
                        :n_retained_points]

                    # These are used for normalization
                    # We can reflect our use of the oracle on all
                    # non-retained points (importantly, assuming we are
                    # using ADE/FDE) by just normalizing over all datapoints
                    # (since the losses at non-retained points would be 0).
                    if self.use_oracle:
                        n_retained_points = M
                    retention_threshold_to_n_points[
                        retention_threshold] = n_retained_points
                    retain_bool_mask[indices_to_retain] = 1
                    retain_bool_mask = retain_bool_mask.astype(np.bool_)
                    counter = 0

                    # Do this instead of enumerate to handle np.ndarray
                    for datapoint_index in range(M):
                        if not retain_bool_mask[datapoint_index]:
                            continue
                        else:
                            counter += 1

                        datapoint_losses = per_plan_losses[datapoint_index]
                        datapoint_weights = self.softmax(
                            self.plan_confidence_scores[datapoint_index])
                        agg_prediction_loss = (
                            aggregate_prediction_request_losses(
                                aggregator=aggregator,
                                per_plan_losses=datapoint_losses,
                                per_plan_weights=datapoint_weights))

                        if (retention_threshold, metric_key) not in (
                                metrics_dict.keys()):
                            metrics_dict[
                                (retention_threshold, metric_key)] = (
                                agg_prediction_loss)
                        else:
                            metrics_dict[
                                (retention_threshold, metric_key)] += (
                                agg_prediction_loss)

        normed_metrics_dict = {}
        for (retention_threshold, metric_key), loss in metrics_dict.items():
            normed_metrics_dict[(retention_threshold, metric_key)] = (
                loss / retention_threshold_to_n_points[retention_threshold])

        return normed_metrics_dict

    def store_retention_metrics(
        self,
        model_name: str,
        metrics_dict: Dict[Tuple[float, str], np.ndarray],
        dataset_key: str,
        return_parsed_dict: bool = True
    ):
        """
        Code based on Deferred Prediction utilities due to
            Neil Band and Angelos Filos
            https://github.com/google/uncertainty-baselines

        Parses a dict of metrics for values obtained at various
            retain proportions.

        Stores results to a DataFrame at the specified path
            (or updates a DataFrame that already exists at the path).

        Optionally, returns a dict with retain proportions separated,
        which allows for more natural logging of tf.Summary values and downstream
        TensorBoard visualization.

        Args:
            model_name: `str`, name of the model for which we are storing
                metrics.
            metrics_dict: `Dict`, metrics computed at various retention
                thresholds.
            dataset_key: `str`, key denoting the name of the evaluation dataset
            return_parsed_dict: `bool`, will return a dict with retain
                proportions separated, which is easier to use for logging
                downstream.

        Returns:
          `Optional[Dict]`
        """
        results = []
        parsed_dict = defaultdict(list) if return_parsed_dict else None

        for (retention_threshold, metric_key), metric_value in (
                metrics_dict.items()):
            results.append((metric_key, retention_threshold, metric_value))

            if parsed_dict is not None:
                parsed_dict[metric_key].append(
                    (retention_threshold, metric_value))

        new_results_df = pd.DataFrame(
            results, columns=['metric', 'retention_threshold', 'value'])

        # Add metadata
        new_results_df['dataset_key'] = dataset_key
        new_results_df['model_prefix'] = self.model_prefix
        new_results_df['model_name'] = model_name
        new_results_df['use_oracle'] = self.use_oracle
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

        if parsed_dict is not None:
            for metric_name in parsed_dict.keys():
                # Sort by ascending retain proportion
                parsed_dict[metric_name] = sorted(parsed_dict[metric_name])
            return parsed_dict
