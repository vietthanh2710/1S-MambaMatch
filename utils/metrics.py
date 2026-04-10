from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SegmentationMetrics:
    """
    Computes pixel accuracy / dice / precision / recall for multiclass segmentation.

    This is a lightly cleaned version of the notebook's `SegmentationMetrics`.
    """

    def __init__(
        self,
        eps: float = 1e-5,
        average: bool = True,
        ignore_background: bool = True,
        activation: str = "0-1",
        ignore_index: int = 255,
    ):
        self.eps = eps
        self.average = average
        self.ignore_background = ignore_background
        self.activation = activation
        self.ignore_index = ignore_index

    @staticmethod
    def _one_hot(gt: torch.Tensor, class_num: int) -> torch.Tensor:
        # gt: (N,H,W) long
        n, *spatial = gt.shape
        one_hot = torch.zeros((n, class_num, *spatial), device=gt.device, dtype=torch.float32)
        return one_hot.scatter_(1, gt.unsqueeze(1).clamp_min(0), 1.0)

    def _get_class_data(self, gt_onehot: torch.Tensor, pred_onehot: torch.Tensor, class_num: int) -> np.ndarray:
        matrix = np.zeros((3, class_num), dtype=np.float64)
        for i in range(class_num):
            class_pred = pred_onehot[:, i, :, :]
            class_gt = gt_onehot[:, i, :, :]
            pred_flat = class_pred.contiguous().view(-1)
            gt_flat = class_gt.contiguous().view(-1)
            tp = torch.sum(gt_flat * pred_flat)
            fp = torch.sum(pred_flat) - tp
            fn = torch.sum(gt_flat) - tp
            matrix[:, i] = tp.item(), fp.item(), fn.item()
        return matrix

    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """
        y_true: (N, H, W) long
        y_pred: (N, C, H, W) logits
        """
        class_num = int(y_pred.size(1))

        # mask out ignore_index in the ground truth
        valid = y_true != self.ignore_index
        y_true_safe = y_true.clone()
        y_true_safe[~valid] = 0

        if self.activation in [None, "none"]:
            pred = y_pred
        elif self.activation == "sigmoid":
            pred = torch.sigmoid(y_pred)
        elif self.activation == "softmax":
            pred = torch.softmax(y_pred, dim=1)
        elif self.activation == "0-1":
            pred = torch.argmax(y_pred, dim=1)
        else:
            raise NotImplementedError(f"Unsupported activation={self.activation}")

        if pred.dim() == 3:
            pred_onehot = self._one_hot(pred.long(), class_num)
        else:
            pred_onehot = (pred > 0.5).float() if class_num == 1 else (pred > 0.5).float()
            if pred_onehot.shape[1] != class_num:
                pred_onehot = pred_onehot[:, :class_num]

        gt_onehot = self._one_hot(y_true_safe.long(), class_num)

        # zero-out ignored pixels in both
        valid_f = valid.float().unsqueeze(1)
        gt_onehot = gt_onehot * valid_f
        pred_onehot = pred_onehot * valid_f

        matrix = self._get_class_data(gt_onehot, pred_onehot, class_num)
        if self.ignore_background and class_num > 1:
            matrix = matrix[:, 1:]

        pixel_acc = (np.sum(matrix[0, :]) + self.eps) / (np.sum(matrix[0, :]) + np.sum(matrix[1, :]) + self.eps)
        dice = (2 * matrix[0] + self.eps) / (2 * matrix[0] + matrix[1] + matrix[2] + self.eps)
        precision = (matrix[0] + self.eps) / (matrix[0] + matrix[1] + self.eps)
        recall = (matrix[0] + self.eps) / (matrix[0] + matrix[2] + self.eps)

        if self.average:
            dice = float(np.average(dice)) if dice.size else 0.0
            precision = float(np.average(precision)) if precision.size else 0.0
            recall = float(np.average(recall)) if recall.size else 0.0

        return float(pixel_acc), dice, precision, recall

