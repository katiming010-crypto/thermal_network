"""Evaluation helpers for the thermal PINN."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import torch


def r2_score_924(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def mae_924(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred))) if y_true.size else float("nan")


def rmse_924(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if y_true.size else float("nan")


def compute_metric_dict_924(
    targets: np.ndarray,
    predictions: np.ndarray,
    observed_nodes: Iterable[str],
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for index, node in enumerate(observed_nodes):
        node_target = targets[:, index]
        node_pred = predictions[:, index]
        metrics[node] = {
            "r2": r2_score_924(node_target, node_pred),
            "mae": mae_924(node_target, node_pred),
            "rmse": rmse_924(node_target, node_pred),
        }
    return metrics


def stack_tensor_batches_924(tensors: Iterable[torch.Tensor]) -> np.ndarray:
    tensor_list = list(tensors)
    if not tensor_list:
        return np.empty((0, 0), dtype=float)
    return torch.cat([tensor.detach().cpu() for tensor in tensor_list], dim=0).numpy()

