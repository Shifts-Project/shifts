import torch
import torch.nn as nn
from typing import Sequence


class SDCLoss:
    def __init__(self):
        self.l1_criterion = nn.L1Loss(reduction="none")

    def average_displacement_error(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes average displacement error
            ADE(y) = (1/T) \sum_{t=1}^T || s_t - s^*_t ||
        where y = (s_1, ..., s_T)

        Args:
            predictions: shape (B, T, 2)
            ground_truth: shape (B, T, 2)
        """
        loss = self.l1_criterion(predictions, ground_truth)

        # Sum displacements within timesteps
        loss = torch.sum(loss, dim=-1)

        # Average displacements over batch and timesteps
        return torch.mean(loss)

    def min_average_displacement_error(
        self,
        predictions_list: Sequence[torch.Tensor],
        ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes
            minADE_k(\{y^i\}_{i=1}^k) = min_{\{y^i\}_{i=1}^k} ADE(y^i)
        """
        return torch.min(
            torch.stack([self.average_displacement_error(
                predictions=predictions, ground_truth=ground_truth)
                for predictions in predictions_list]))

    def final_displacement_error(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes final displacement error
            FDE(y) = (1/T) || s_T - s^*_T ||
        where y = (s_1, ..., s_T)

        Args:
            predictions: shape (B, T, 2)
            ground_truth: shape (B, T, 2)
        """
        assert predictions.size(1) == ground_truth.size(1)
        final_pred_step = predictions[:, predictions.size(1) - 1, :]
        final_ground_truth_step = ground_truth[:, ground_truth.size(1) - 1, :]
        loss = self.l1_criterion(final_pred_step, final_ground_truth_step)

        # Sum displacement of final step over the position dimension (x, y)
        loss = torch.sum(loss, dim=-1)

        # Average displacement of final step over batch
        loss = torch.mean(loss)

        return loss

    def min_final_displacement_error(
        self,
        predictions_list: Sequence[torch.Tensor],
        ground_truth: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes
            minFDE_k(\{y^i\}_{i=1}^k) = min_{\{y^i\}_{i=1}^k} FDE(y^i)
        """
        return torch.min(
            torch.stack([self.final_displacement_error(
                predictions=predictions, ground_truth=ground_truth)
                for predictions in predictions_list]))
