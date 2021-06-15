import datetime
import os
from collections import defaultdict
from typing import Sequence, Callable, Union, Dict, Text, Mapping

import pandas as pd
import torch
import torch.nn as nn


class SDCLoss:
    def __init__(self, full_model_name, c):
        # self.l1_criterion = nn.L1Loss(reduction="none")

        self.valid_base_metrics = {'ade', 'fde'}
        self.valid_aggregators = {'min', 'mean', 'max', 'confidence-weight'}
        self.softmax = torch.nn.Softmax(dim=0)

        # Store ADE/FDE of all B * D_b preds
        self.base_metric_to_losses = defaultdict(list)
        
        # Will ultimately have shape (M, D_i), where
        #   M = number of datapoints in evaluation set
        #   D_i = number of predictions per scene, which can vary per scene i
        self.plan_confidence_scores = []

        # Will ultimately have shape (M,)
        self.pred_request_confidence_scores = []

        # We sweep over these retention thresholds, at each threshold retaining
        # the specified proportion of prediction requests (or plans, of which
        # there may be many for a  given prediction request) with the highest
        # confidence scores.
        self.retention_thresholds = c.metrics_retention_thresholds

        # Path where area under retention curve results should be stored
        metrics_dir = f'{c.dir_data}/metrics' if not c.dir_metrics else c.dir_metrics
        os.makedirs(metrics_dir, exist_ok=True)
        self.metrics_dir = os.path.join(metrics_dir, full_model_name)
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Name of the model,
        # e.g., rip-dim-k_3-plan_wcm-scene_bcm would correspond to
        # RIP ensemble with DIM backbone, K = 3 members, worst-case
        # aggregation for plan confidence, best-case aggregation
        # for scene confidence.
        # See sdc.oatomobile.torch.baselines.robust_imitative_planning.py
        self.model_name = full_model_name

        # seed used in evaluating the current model.
        self.eval_seed = c.torch_seed

        # if use_oracle, we assume that all points not retained have
        # perfect accuracy / 0 loss
        # otherwise we disclude them, and consider only predictions on
        # retained points
        self.use_oracle = c.metrics_retention_use_oracle

        # Model prefix -- used for convenience to differentiate e.g.
        # runs trained on a subset or all of the training data
        self.model_prefix = c.model_prefix

        # ** Caching Full Results **
        # Done if we wish to perform post-hoc analyses of model predictions
        # e.g., how does confidence correlate with various GT trajectory types?
        #
        # predictions: torch.Tensor,
        # batch: Mapping[str, torch.Tensor],
        # plan_confidence_scores: torch.Tensor = None,
        # pred_request_confidence_scores: torch.Tensor = None,


    def average_displacement_error(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes average displacement error
            ADE(y) = (1/T) \sum_{t=1}^T || s_t - s^*_t ||_2
        where y = (s_1, ..., s_T)

        Does not aggregate over the first (batch) dimension.

        Args:
            predictions: shape (*, T, 2)
            ground_truth: shape (*, T, 2)
        """
        loss = (predictions - ground_truth) ** 2

        # Sum displacements within timesteps (i.e., across x-y coords)
        loss = torch.sum(loss, dim=-1)

        # Calculate root of loss (= L2 norm)
        loss = loss ** 0.5

        # Average displacements over timesteps
        return torch.mean(loss, dim=-1)

    @staticmethod
    def final_displacement_error(
        predictions: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes final displacement error
            FDE(y) = (1/T) || s_T - s^*_T ||_2
        where y = (s_1, ..., s_T)

        Does not aggregate over the first (batch) dimension.

        Args:
            predictions: shape (*, T, 2)
            ground_truth: shape (*, T, 2)
        """
        assert predictions.size(1) == ground_truth.size(1)
        final_pred_step = predictions[:, predictions.size(1) - 1, :]
        final_ground_truth_step = ground_truth[:, ground_truth.size(1) - 1, :]
        loss = (final_pred_step - final_ground_truth_step) ** 2

        # Sum displacement of final step over the position dimension (x, y)
        loss = torch.sum(loss, dim=-1)

        # Calculate root of loss (= L2 norm)
        loss = loss ** 0.5

        return loss

    def clear_per_dataset_attributes(self):
        self.base_metric_to_losses = defaultdict(list)
        self.plan_confidence_scores = []
        self.pred_request_confidence_scores = []

    def construct_dataset_losses_and_confidence_scores(self):
        assert self.plan_confidence_scores is not None
        assert self.pred_request_confidence_scores is not None
        if True:
            a = 1

        plan_tensor_dims = len(self.plan_confidence_scores[0].size())
        try:
            if plan_tensor_dims == 1:
                self.plan_confidence_scores = torch.stack(
                    self.plan_confidence_scores, dim=0)
            elif plan_tensor_dims == 2:
                self.plan_confidence_scores = torch.cat(
                    self.plan_confidence_scores, dim=0)
            else:
                raise ValueError(
                    'Unexpected number of dims in plan_confidence_scores.')
        except RuntimeError:
            print('Detected differing number of predicted plans for '
                  'each scene.')
            pass

        self.pred_request_confidence_scores = torch.cat(
            self.pred_request_confidence_scores, dim=0)

        base_metric_to_losses = {}
        for base_metric, losses_list in self.base_metric_to_losses.items():
            try:
                base_metric_to_losses[base_metric] = torch.stack(
                    losses_list, dim=0)
            except RuntimeError:
                base_metric_to_losses[base_metric] = losses_list

        self.base_metric_to_losses = base_metric_to_losses

    def evaluate_dataset_losses(self, dataset_key: str):
        self.construct_dataset_losses_and_confidence_scores()

        assert len(self.base_metric_to_losses['ade']) == len(
            self.base_metric_to_losses['fde'])

        if True:
            a = 1

        # ** RIP evaluation **

        # Sort prediction request confidences by decreasing confidence
        sorted_pred_req_confs = torch.argsort(
            self.pred_request_confidence_scores, descending=True)

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
        # Randomly sort prediction requests
        random_pred_req_confs = torch.randperm(
            self.pred_request_confidence_scores.size(0))
        random_retention_metrics = (
            self.evaluate_pred_request_retention_thresholds(
                random_pred_req_confs))
        self.store_retention_metrics(
            model_name='Random',
            metrics_dict=random_retention_metrics, dataset_key=dataset_key,
            return_parsed_dict=False)

        # ** Optimal Retention evaluation **
        # Sort prediction requests by the loss, giving us the performance
        # of a model with perfectly calibrated uncertainty
        optimal_retention_metrics = (
            self.get_pred_request_retention_thresholds_optimal_perf())
        self.store_retention_metrics(
            model_name='Optimal',
            metrics_dict=optimal_retention_metrics, dataset_key=dataset_key,
            return_parsed_dict=False)

        # TODO:
        # Evaluate metrics for each plan retention proportion (i.e.,
        # the uncertainty-aware model could choose to retain a subset of
        # predictions for a given prediction request)

        pred_req_retention_metrics = {
            f'{metric_key}_{retention_threshold}': loss
            for (retention_threshold, metric_key), loss in
            pred_req_retention_metrics.items()
        }
        self.clear_per_dataset_attributes()
        return pred_req_retention_metrics

    # def cache_full_results(
    #     self,
    #     predictions: torch.Tensor,
    #     batch: Mapping[str, torch.Tensor],
    #     plan_confidence_scores: torch.Tensor = None,
    #     pred_request_confidence_scores: torch.Tensor = None,
    # ):
    #     # TODO: support for varying # plans per prediction request
    #     for obj in [predictions, plan_confidence_scores,
    #                 pred_request_confidence_scores]:
    #         if not isinstance(obj, torch.Tensor):
    #             raise NotImplementedError
    #
    #     ground_truth = batch['ground_truth_trajectory']
    #
    #     # cache predictions, ground truth, uncertainty estimates,
    #     # scene ID, and trajectory tags




    def cache_batch_losses(
        self,
        predictions_list: Union[Sequence[torch.Tensor], torch.Tensor],
        ground_truth_batch: torch.Tensor,
        plan_confidence_scores_list: Union[
            Sequence[torch.Tensor], torch.Tensor] = None,
        pred_request_confidence_scores: torch.Tensor = None,
    ):
        """Precomputes and caches the batch ADE and FDE for each of the D_b
        predictions in each of the b \in B scenes.
        
        Also stores plan_ and pred_request_confidence_scores if provided.

        Args:
            predictions_list:
                if List[torch.Tensor]: length is B, the number of prediction
                    requests in a batch.
                    Each tensor has shape (D_b, T, 2) where D_b is the number
                    of trajectory predictions, which can vary for each scene.
                if torch.Tensor: we have a consistent number of predictions D
                    per scene. Shape (B, D, T, 2)
            ground_truth_batch: torch.Tensor, shape (B, T, 2), there is only
                 one ground_truth trajectory for each prediction request.
            plan_confidence_scores_list: length is B, the number of scenes in a 
                batch. Similar to predictions_list, is a list if there are 
                a different number of predictions per scene, in which case
                each entry has shape (D_b,).
            pred_request_confidence_scores: shape (B,), model confidence scores
                on each scene.
        """
        # Do this instead of enumerate to handle torch.Tensor
        for batch_index in range(len(predictions_list)):
            # if True:
            #     a = 1

            # predictions shape: (D_b, T, 2), where D_b can vary per
            # prediction request
            predictions = predictions_list[batch_index]
            D_b = predictions.size(0)
            ground_truth = torch.unsqueeze(
                ground_truth_batch[batch_index], dim=0)

            # Tile over the first dimension
            ground_truth = torch.tile(ground_truth, (D_b, 1, 1)).to(
                device=predictions.device)
            for base_metric in self.valid_base_metrics:
                if base_metric == 'ade':
                    base_metric_fn = self.average_displacement_error
                elif base_metric == 'fde':
                    base_metric_fn = self.final_displacement_error
                else:
                    raise NotImplementedError

                self.base_metric_to_losses[base_metric].append(
                    base_metric_fn(
                        predictions=predictions, ground_truth=ground_truth))

        # if True:
        #     a = 1

        if plan_confidence_scores_list is not None:
            if isinstance(plan_confidence_scores_list, list):
                self.plan_confidence_scores += plan_confidence_scores_list
            elif isinstance(plan_confidence_scores_list, torch.Tensor):
                self.plan_confidence_scores.append(
                    plan_confidence_scores_list)
            else:
                raise ValueError(
                    f'Unexpected type for plan_confidence_scores_list: '
                    f'{type(plan_confidence_scores_list)}')

        if pred_request_confidence_scores is not None:
            self.pred_request_confidence_scores.append(
                pred_request_confidence_scores)

    @staticmethod
    def batch_mean_metric(
        base_metric: Callable,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
    ) -> torch.Tensor:
        """During training, we may wish to produce a single prediction for each
        prediction request (i.e., just sample once from the predictive
        posterior; similar to training of an MC Dropout model).

        Args:
            base_metric: Callable, function such as
                `average_displacement_error`
            predictions: torch.Tensor, shape (B, T, 2) where B is the number of
                prediction requests in the batch.
            ground_truth: torch.Tensor, shape (B, T, 2), there is only one
                ground_truth trajectory for each prediction request.
        """
        return torch.mean(
            base_metric(predictions=predictions, ground_truth=ground_truth))

    def evaluate_pred_request_retention_thresholds(
        self,
        sorted_pred_request_confidences: torch.Tensor,
    ) -> Dict[Text, torch.Tensor]:
        """For various retention thresholds, evaluate all pairwise
            combinations of aggregation (e.g., min, max, ...) and ADE/FDE.

            Code based on Deferred Prediction utilities due to
                Neil Band and Angelos Filos
                https://github.com/google/uncertainty-baselines

        Args:
            sorted_pred_request_confidences: torch.Tensor, per--prediction
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
            retain_bool_mask = torch.zeros_like(
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
            retain_bool_mask = retain_bool_mask.bool()
            counter = 0

            # Do this instead of enumerate to handle torch.Tensor
            for datapoint_index in range(M):
                if not retain_bool_mask[datapoint_index]:
                    continue
                else:
                    counter += 1

                for base_metric in self.valid_base_metrics:
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
                    per_plan_confidences = self.softmax(per_plan_confidences)

                    for aggregator in self.valid_aggregators:
                        metric_key = (f'{aggregator}{base_metric.upper()}_'
                                      f'retain_scene')
                        agg_prediction_loss = (
                            self.aggregate_prediction_request_losses(
                                aggregator=aggregator,
                                per_plan_losses=per_plan_losses,
                                per_plan_confidences=per_plan_confidences))
                        if (retention_threshold, metric_key) not in (
                                metrics_dict.keys()):
                            metrics_dict[
                                (retention_threshold, metric_key)] = (
                                agg_prediction_loss)
                        else:
                            metrics_dict[
                                (retention_threshold, metric_key)] += (
                                agg_prediction_loss)

            print('counter:', counter, 'retention thresh', retention_threshold)
        normed_metrics_dict = {}
        for (retention_threshold, metric_key), loss in metrics_dict.items():
            print(retention_threshold_to_n_points[retention_threshold])
            normed_metrics_dict[(retention_threshold, metric_key)] = (
                loss / retention_threshold_to_n_points[retention_threshold])

        return normed_metrics_dict

    @staticmethod
    def aggregate_prediction_request_losses(
        aggregator,
        per_plan_losses,
        per_plan_confidences
    ) -> torch.Tensor:
        if aggregator == 'min':
            agg_prediction_loss = torch.min(per_plan_losses)
        elif aggregator == 'max':
            agg_prediction_loss = torch.max(per_plan_losses)
        elif aggregator == 'mean':
            agg_prediction_loss = torch.mean(per_plan_losses)
        elif aggregator == 'confidence-weight':
            # Linear combination of the losses for the generated
            # predictions of a given request, using normalized
            # per-plan confidence scores as coefficients.
            agg_prediction_loss = torch.sum(
                per_plan_confidences * per_plan_losses)
        else:
            raise NotImplementedError

        return agg_prediction_loss

    def get_pred_request_retention_thresholds_optimal_perf(
            self) -> Dict[Text, torch.Tensor]:
        """Determine the optimal performance on the retention task if we
            used a model with perfectly calibrated confidence (i.e., its
            confidence ordering has perfect Spearman correlation with the
            True ordering of losses for each pairwise combination of
            aggregation (e.g., min, max, ...) and ADE/FDE.

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
        # a performance upper bound.
        for base_metric in self.valid_base_metrics:
            assert base_metric in {'ade', 'fde'}, (
                'This method must be updated to handle oracle use + '
                'any metric for which perfect performance is not 0 '
                '(e.g., an accuracy).')
            per_plan_losses = self.base_metric_to_losses[base_metric]
            for aggregator in self.valid_aggregators:
                metric_key = (f'{aggregator}{base_metric.upper()}_'
                              f'retain_scene')
                metric_agg_losses = torch.zeros(M)
                for datapoint_index in range(M):
                    datapoint_losses = per_plan_losses[datapoint_index]
                    datapoint_confidences = self.softmax(
                        self.plan_confidence_scores[datapoint_index])
                    metric_agg_losses[datapoint_index] = (
                        self.aggregate_prediction_request_losses(
                            aggregator=aggregator,
                            per_plan_losses=datapoint_losses,
                            per_plan_confidences=datapoint_confidences))

                # Now we argsort the losses to get the ideal order
                # i.e., order losses in ascending order (breaks for accuracy)
                # TODO: update for any miss rate - like metric
                sorted_metric_agg_losses = torch.argsort(
                    metric_agg_losses, descending=False)

                # Compute metrics for range of retention thresholds
                for retention_threshold in self.retention_thresholds:
                    retain_bool_mask = torch.zeros_like(
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
                    retain_bool_mask = retain_bool_mask.bool()
                    counter = 0

                    # Do this instead of enumerate to handle torch.Tensor
                    for datapoint_index in range(M):
                        if not retain_bool_mask[datapoint_index]:
                            continue
                        else:
                            counter += 1

                        datapoint_losses = per_plan_losses[datapoint_index]
                        datapoint_confidences = self.softmax(
                            self.plan_confidence_scores[datapoint_index])
                        agg_prediction_loss = (
                            self.aggregate_prediction_request_losses(
                                aggregator=aggregator,
                                per_plan_losses=datapoint_losses,
                                per_plan_confidences=datapoint_confidences))

                        if (retention_threshold, metric_key) not in (
                                metrics_dict.keys()):
                            metrics_dict[
                                (retention_threshold, metric_key)] = (
                                agg_prediction_loss)
                        else:
                            metrics_dict[
                                (retention_threshold, metric_key)] += (
                                agg_prediction_loss)

                    print(
                        'counter:', counter,
                        'retention thresh', retention_threshold)
        normed_metrics_dict = {}
        for (retention_threshold, metric_key), loss in metrics_dict.items():
            print(retention_threshold_to_n_points[retention_threshold])
            normed_metrics_dict[(retention_threshold, metric_key)] = (
                loss / retention_threshold_to_n_points[retention_threshold])

        return normed_metrics_dict


    def evaluate_per_plan_retention_thresholds(
        self,
        sorted_pred_request_confidences: torch.Tensor,
    ) -> Dict[Text, torch.Tensor]:
        """For a given retention fraction -- a proportion of the evaluation
            set to "retain" (i.e. not defer) -- evaluate ADE and FDE among
            all plans (of which there may be many for a particular prediction
            request).

            Code based on Deferred Prediction utilities due to
                Angelos Filos and Neil Band
                https://github.com/google/uncertainty-baselines

        Args:
            sorted_pred_request_confidences: torch.Tensor, per--prediction
                request confidence scores, sorted by decreasing confidence.
                At each retention threshold, we retain the top (X * 100)%
                of the dataset and compute metrics as normal.
        Returns:
          Dict with format key: str = f'metric_retain_{fraction}', and
          value: metric value at given data retained fraction.
        """
        raise NotImplementedError
        # M = len(self.base_metric_to_losses['ade'])
        # metrics_dict = {}
        # retention_threshold_to_n_points = {}
        #
        # # Compute metrics for range of retention thresholds
        # for retention_threshold in self.retention_thresholds:
        #     retain_bool_mask = torch.zeros_like(
        #         sorted_pred_request_confidences)
        #     n_retained_points = int(M * retention_threshold)
        #     retention_threshold_to_n_points[
        #         retention_threshold] = n_retained_points
        #     indices_to_retain = sorted_pred_request_confidences[
        #                         :n_retained_points]
        #     retain_bool_mask[indices_to_retain] = 1
        #     retain_bool_mask = retain_bool_mask.bool()
        #
        #     # Do this instead of enumerate to handle torch.Tensor
        #     for datapoint_index in range(M):
        #         if retain_bool_mask[datapoint_index] is False:
        #             continue
        #
        #         for base_metric in self.valid_base_metrics:
        #             per_plan_losses = self.base_metric_to_losses[
        #                 base_metric][datapoint_index]
        #
        #             per_plan_confidences = self.plan_confidence_scores[
        #                 datapoint_index]
        #
        #             # Normalize the per-plan confidence scores for a
        #             # particular prediction request (i.e., within this scene).
        #             per_plan_confidences = self.softmax(per_plan_confidences)
        #
        #             for aggregator in self.valid_aggregators:
        #                 metric_key = (f'{aggregator}{base_metric.upper()}')
        #                 if aggregator == 'min':
        #                     agg_prediction_loss = torch.min(per_plan_losses)
        #                 elif aggregator == 'max':
        #                     agg_prediction_loss = torch.max(per_plan_losses)
        #                 elif aggregator == 'mean':
        #                     agg_prediction_loss = torch.mean(per_plan_losses)
        #                 elif aggregator == 'confidence-weight':
        #                     # Linear combination of the losses for the
        #                     # generated predictions of a given request, using
        #                     # normalized per-plan confidence scores
        #                     # as coefficients.
        #                     agg_prediction_loss = torch.sum(
        #                         per_plan_confidences * per_plan_losses)
        #
        #                 full_metric_key = (retention_threshold, metric_key)
        #                 if full_metric_key not in metrics_dict.keys():
        #                     metrics_dict[full_metric_key] = (
        #                         agg_prediction_loss)
        #                 else:
        #                     metrics_dict[full_metric_key] += (
        #                         agg_prediction_loss)
        #
        # if True:
        #     a = 1
        #
        # normed_metrics_dict = {}
        # for (retention_threshold, metric_key), metric_value in (
        #         metrics_dict.items()):
        #     normed_metrics_dict[(retention_threshold, metric_key)] = (
        #         metric_value /
        #         retention_threshold_to_n_points[retention_threshold])
        #
        # return normed_metrics_dict

    def store_retention_metrics(
        self,
        model_name: str,
        metrics_dict: Dict[Text, torch.Tensor],
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
        if return_parsed_dict:
            parsed_dict = defaultdict(list)

        # if True:
        #     a = 1

        for (retention_threshold, metric_key), metric_value in (
                metrics_dict.items()):
            metric_value = metric_value.detach().cpu().numpy()
            results.append((metric_key, retention_threshold, metric_value))

            if return_parsed_dict:
                parsed_dict[metric_key].append(
                    (retention_threshold, metric_value))

        new_results_df = pd.DataFrame(
            results, columns=['metric', 'retention_threshold', 'value'])

        # Add metadata
        new_results_df['dataset_key'] = dataset_key
        new_results_df['model_prefix'] = self.model_prefix
        new_results_df['model_name'] = model_name
        new_results_df['use_oracle'] = self.use_oracle
        new_results_df['eval_seed'] = self.eval_seed
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

        if return_parsed_dict:
            for metric_name in parsed_dict.keys():
                # Sort by ascending retain proportion
                parsed_dict[metric_name] = sorted(parsed_dict[metric_name])
            return parsed_dict
