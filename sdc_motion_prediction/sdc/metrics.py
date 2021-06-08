from typing import Sequence, Callable

import torch
import torch.nn as nn


class SDCLoss:
    def __init__(self):
        self.l1_criterion = nn.L1Loss(reduction="none")
        self.valid_aggregators = {'min', 'mean', 'max', 'confidence-weight'}

    def average_displacement_error(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes average displacement error
            ADE(y) = (1/T) \sum_{t=1}^T || s_t - s^*_t ||
        where y = (s_1, ..., s_T)

        Does not aggregate over the first (batch) dimension.

        Args:
            predictions: shape (*, T, 2)
            ground_truth: shape (*, T, 2)
        """
        loss = self.l1_criterion(predictions, ground_truth)

        # Sum displacements within timesteps
        loss = torch.sum(loss, dim=-1)

        # Average displacements over timesteps
        return torch.mean(loss, dim=-1)

    def final_displacement_error(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes final displacement error
            FDE(y) = (1/T) || s_T - s^*_T ||
        where y = (s_1, ..., s_T)

        Does not aggregate over the first (batch) dimension.

        Args:
            predictions: shape (*, T, 2)
            ground_truth: shape (*, T, 2)
        """
        assert predictions.size(1) == ground_truth.size(1)
        final_pred_step = predictions[:, predictions.size(1) - 1, :]
        final_ground_truth_step = ground_truth[:, ground_truth.size(1) - 1, :]
        loss = self.l1_criterion(final_pred_step, final_ground_truth_step)

        # Sum displacement of final step over the position dimension (x, y)
        loss = torch.sum(loss, dim=-1)

        return loss

    def aggregate_metric(
        self,
        base_metric: Callable,
        predictions_list: Sequence[torch.Tensor],
        ground_truth: torch.Tensor,
        confidence_scores_list: Sequence[torch.Tensor] = None,
        aggregator: str = 'mean'
    ) -> torch.Tensor:
        """Wraps a metric independent over the batch dimension with an
        aggregation.

        Args:
            base_metric: Callable, function such as
                `average_displacement_error`
            predictions_list: Sequence[torch.Tensor], length is B, the
                number of prediction requests in a batch.
                Each tensor has shape (P_b, T, 2) where P_b is the number
                of model trajectory predictions, which can vary for each
                prediction request.
            ground_truth: torch.Tensor, shape (B, T, 2), there is only one
                ground_truth trajectory for each prediction request.
            confidence_scores_list: Sequence[torch.Tensor], length is B, the
                number of prediction requests in a batch.
                Each tensor has shape (P_b), and is expected to sum to 1,
                i.e., confidence scores among the predictions for a particular
                input are normalized.
            aggregator: specifies the aggregation that is applied among the
                P_b predictions for each prediction_request.
                e.g., aggregator='min' with
                base_metric=average_displacement_error computes
                    minADE, i.e.,
                    minADE_k(\{y^i\}_{i=1}^k) = min_{\{y^i\}_{i=1}^k} ADE(y^i)
                aggregator='min' with base_metric=final_displacement_error
                    computes minFDE, i.e.,
                    minFDE_k(\{y^i\}_{i=1}^k) = min_{\{y^i\}_{i=1}^k} FDE(y^i)
                with k = P_b.
        """
        assert aggregator in self.valid_aggregators, (
            'Invalid aggregator over predictions (dimension P_b) specified.')
        if aggregator == 'confidence-weight':
            assert confidence_scores_list is not None

        prediction_losses = []
        for batch_index, predictions in enumerate(predictions_list):
            # predictions shape: (P_b, T, 2), where P_b can vary per
            # prediction request
            P_b = predictions.size(0)
            ground_truth = torch.unsqueeze(ground_truth[batch_index], dim=0)

            # Tile over the first dimension
            ground_truth = torch.tile(ground_truth, (P_b, 1, 1)).to(
                device=predictions.device)
            prediction_loss = base_metric(
                    predictions=predictions, ground_truth=ground_truth)

            # Aggregate the prediction loss over the P_b dimension
            if aggregator == 'min':
                prediction_loss = torch.min(prediction_loss)
            elif aggregator == 'max':
                prediction_loss = torch.max(prediction_loss)
            elif aggregator == 'mean':
                prediction_loss = torch.mean(prediction_loss)
            elif aggregator == 'confidence-weight':
                # Linear combination of the losses for the generated
                # predictions of a given request, using confidence scores
                # as coefficients.
                confidence_scores = confidence_scores_list[batch_index]
                prediction_loss = torch.sum(
                    confidence_scores * prediction_loss)
            else:
                raise NotImplementedError

            prediction_losses.append(prediction_loss)

        # Take the mean over the per--prediction request losses
        return torch.mean(torch.stack(prediction_losses))

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
