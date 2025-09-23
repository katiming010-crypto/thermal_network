"""Training entry-point for the improved thermal PINN."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_utils_924 import PreparedData924, load_measurements_924, prepare_datasets_924
from evaluation_utils_924 import (
    compute_metric_dict_924,
    stack_tensor_batches_924,
)
from physics_utils_924 import (
    build_target_mean_vectors,
    build_target_scaling_vectors,
    compute_energy_balance_residuals,
)
from pinn_model_924 import ThermalPINN924
from thermal_graph_924 import ThermalNetworkParameters924
from visualization_924 import plot_temperature_profiles_924, plot_topology_924


DEFAULT_CONFIG_924: Dict[str, Any] = {
    "hidden_layers": [256, 256, 128, 128],
    "activation": "silu",
    "dropout": 0.05,
    "batch_size": 512,
    "epochs": 800,
    "learning_rate": 3e-4,
    "weight_decay": 1e-5,
    "physics_weight": 0.7,
    "parameter_reg_weight": 1e-4,
    "pm_rotor_coupling_weight": 0.02,
    "gradient_clip": 1.5,
    "patience": 120,
    "data_loss_weights": {
        "coolant_channel": 0.6,
        "stator_yoke": 1.0,
        "stator_tooth": 1.2,
        "stator_winding": 1.2,
        "rotor_core": 1.5,
        "permanent_magnet": 3.4,
        "bearing": 0.9,
    },
    "physics_residual_weights": {
        "coolant_channel": 0.8,
        "stator_yoke": 1.0,
        "stator_tooth": 1.1,
        "stator_winding": 1.15,
        "rotor_core": 1.35,
        "permanent_magnet": 2.4,
        "bearing": 0.85,
    },
    "pm_focus_target_r2": 0.9,
    "pm_focus_relax_r2": 0.96,
    "pm_focus_step": 0.45,
    "pm_focus_max_multiplier": 5.5,
    "pm_focus_cooldown": 10,
}


def set_seed_924(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - device specific
        torch.cuda.manual_seed_all(seed)


def create_dataloaders_924(
    prepared: PreparedData924,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return (
        DataLoader(prepared.train_dataset, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(prepared.val_dataset, batch_size=batch_size, shuffle=False, drop_last=False),
        DataLoader(prepared.test_dataset, batch_size=batch_size, shuffle=False, drop_last=False),
    )


def _forward_batch_924(
    model: ThermalPINN924,
    graph: ThermalNetworkParameters924,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    observed_indices: List[int],
    observed_weight_tensor: torch.Tensor,
    residual_weight_tensor: torch.Tensor,
    target_std_tensor: torch.Tensor,
    target_mean_tensor: torch.Tensor,
    time_index: int,
    time_std: float,
    config: Dict[str, Any],
    compute_predictions: bool = False,
) -> Dict[str, Any]:
    features = batch["features"].to(device)
    time_scaled = features[:, time_index: time_index + 1].detach()
    features = features.clone().detach().requires_grad_(True)
    target_norm = batch["target_norm"].to(device)
    powers = batch["powers"].to(device)
    sink_temperature = batch["sink_temperature"].to(device)

    predictions_norm = model(features)
    if observed_indices:
        observed_pred = predictions_norm[:, observed_indices]
        data_diff = observed_pred - target_norm
        data_loss = (data_diff.pow(2) * observed_weight_tensor).mean()
    else:
        data_loss = torch.tensor(0.0, device=device)

    grads: List[torch.Tensor] = []
    for output_index in range(predictions_norm.shape[1]):
        grad = torch.autograd.grad(
            outputs=predictions_norm[:, output_index : output_index + 1],
            inputs=features,
            grad_outputs=torch.ones_like(predictions_norm[:, output_index : output_index + 1]),
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )[0]
        if grad is None:
            grads.append(torch.zeros_like(time_scaled))
        else:
            grads.append(grad[:, time_index : time_index + 1])
    grad_tensor = torch.cat(grads, dim=1)

    std_vector = target_std_tensor.unsqueeze(0)
    mean_vector = target_mean_tensor.unsqueeze(0)
    dtemp_dt = grad_tensor * (std_vector / time_std)
    temperatures = predictions_norm * std_vector + mean_vector

    residuals = compute_energy_balance_residuals(
        temperatures=temperatures,
        dtemperatures_dt=dtemp_dt,
        powers=powers,
        sink_temperatures=sink_temperature,
        graph=graph,
    )
    if residuals:
        residual_losses = torch.stack([res.pow(2).mean() for res in residuals])
        if residual_weight_tensor.numel() == residual_losses.numel():
            weighted = residual_losses * residual_weight_tensor
        else:
            weighted = residual_losses
        physics_loss = weighted.mean()
    else:
        physics_loss = torch.tensor(0.0, device=device)

    pm_idx = None
    rotor_idx = None
    try:
        pm_idx = graph.node_order.index("permanent_magnet")
        rotor_idx = graph.node_order.index("rotor_core")
    except ValueError:  # pragma: no cover - defensive
        pass

    coupling_loss = torch.tensor(0.0, device=device)
    if pm_idx is not None and rotor_idx is not None:
        coupling_loss = (temperatures[:, pm_idx] - temperatures[:, rotor_idx]).pow(2).mean()

    param_reg = graph.parameter_regularisation()

    loss = (
        data_loss
        + config["physics_weight"] * physics_loss
        + config["parameter_reg_weight"] * param_reg
        + config["pm_rotor_coupling_weight"] * coupling_loss
    )

    result: Dict[str, Any] = {
        "loss": loss,
        "data_loss": data_loss.detach(),
        "physics_loss": physics_loss.detach(),
        "coupling_loss": coupling_loss.detach(),
        "temperatures": temperatures if compute_predictions else None,
        "predictions_norm": predictions_norm if compute_predictions else None,
        "time_scaled": time_scaled if compute_predictions else None,
    }
    return result


def train_epoch_924(
    model: ThermalPINN924,
    graph: ThermalNetworkParameters924,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    observed_indices: List[int],
    observed_weight_tensor: torch.Tensor,
    residual_weight_tensor: torch.Tensor,
    target_std_tensor: torch.Tensor,
    target_mean_tensor: torch.Tensor,
    time_index: int,
    time_std: float,
    config: Dict[str, Any],
) -> Dict[str, float]:
    model.train()
    graph.train()
    total_loss = 0.0
    total_data = 0.0
    total_physics = 0.0
    total_coupling = 0.0
    count = 0

    for batch in dataloader:
        optimizer.zero_grad()
        result = _forward_batch_924(
            model,
            graph,
            batch,
            device,
            observed_indices,
            observed_weight_tensor,
            residual_weight_tensor,
            target_std_tensor,
            target_mean_tensor,
            time_index,
            time_std,
            config,
        )
        result["loss"].backward()
        if config["gradient_clip"] > 0:
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(graph.parameters()), config["gradient_clip"]
            )
        optimizer.step()

        batch_size = batch["features"].shape[0]
        total_loss += float(result["loss"].detach()) * batch_size
        total_data += float(result["data_loss"]) * batch_size
        total_physics += float(result["physics_loss"]) * batch_size
        total_coupling += float(result["coupling_loss"]) * batch_size
        count += batch_size

    return {
        "loss": total_loss / max(count, 1),
        "data_loss": total_data / max(count, 1),
        "physics_loss": total_physics / max(count, 1),
        "coupling_loss": total_coupling / max(count, 1),
    }


def evaluate_epoch_924(
    model: ThermalPINN924,
    graph: ThermalNetworkParameters924,
    dataloader: DataLoader,
    device: torch.device,
    observed_indices: List[int],
    observed_nodes: Iterable[str],
    observed_weight_tensor: torch.Tensor,
    residual_weight_tensor: torch.Tensor,
    target_std_tensor: torch.Tensor,
    target_mean_tensor: torch.Tensor,
    time_index: int,
    time_std: float,
    time_mean: float,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    model.eval()
    graph.eval()

    total_loss = 0.0
    total_data = 0.0
    total_physics = 0.0
    total_coupling = 0.0
    count = 0

    prediction_batches: List[torch.Tensor] = []
    target_batches: List[torch.Tensor] = []
    time_batches: List[np.ndarray] = []

    for batch in dataloader:
        with torch.enable_grad():
            result = _forward_batch_924(
                model,
                graph,
                batch,
                device,
                observed_indices,
                observed_weight_tensor,
                residual_weight_tensor,
                target_std_tensor,
                target_mean_tensor,
                time_index,
                time_std,
                config,
                compute_predictions=True,
            )

        batch_size = batch["features"].shape[0]
        total_loss += float(result["loss"].detach()) * batch_size
        total_data += float(result["data_loss"]) * batch_size
        total_physics += float(result["physics_loss"]) * batch_size
        total_coupling += float(result["coupling_loss"]) * batch_size
        count += batch_size

        if observed_indices:
            prediction_batches.append(result["temperatures"][:, observed_indices].detach().cpu())
            target_batches.append(batch["target_actual"].detach().cpu())
            time_scaled = result["time_scaled"].detach().cpu().numpy().ravel()
            time_batches.append(time_scaled)

    if observed_indices:
        predictions = stack_tensor_batches_924(prediction_batches)
        targets = stack_tensor_batches_924(target_batches)
        time_axis_scaled = np.concatenate(time_batches) if time_batches else np.array([])
        time_axis = time_axis_scaled * time_std + time_mean
        metrics = compute_metric_dict_924(targets, predictions, observed_nodes)
    else:
        predictions = np.empty((0, 0), dtype=float)
        targets = np.empty((0, 0), dtype=float)
        time_axis = np.array([])
        metrics = {}

    return {
        "loss": total_loss / max(count, 1),
        "data_loss": total_data / max(count, 1),
        "physics_loss": total_physics / max(count, 1),
        "coupling_loss": total_coupling / max(count, 1),
        "metrics": metrics,
        "predictions": predictions,
        "targets": targets,
        "time_axis": time_axis,
    }


def parse_args_924() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the thermal PINN with physics guidance.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/input/measures_v2.csv"),
        help="Path to the measurement CSV file.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_924"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=924)
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG_924["epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG_924["batch_size"])
    return parser.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _build_observed_weight_vector(observed_nodes: Iterable[str], config: Dict[str, Any]) -> List[float]:
    return [float(config["data_loss_weights"].get(node, 1.0)) for node in observed_nodes]


def _weights_to_tensor(weights: Sequence[float], device: torch.device, *, expand: bool) -> torch.Tensor:
    if not weights:
        shape = (1, 0) if expand else (0,)
        return torch.zeros(shape, dtype=torch.float32, device=device)
    tensor = torch.as_tensor(list(weights), dtype=torch.float32, device=device)
    return tensor.unsqueeze(0) if expand else tensor


def _prepare_weights_tensor(observed_nodes: Iterable[str], config: Dict[str, Any], device: torch.device) -> torch.Tensor:
    weights = _build_observed_weight_vector(observed_nodes, config)
    return _weights_to_tensor(weights, device, expand=True)


def _prepare_residual_weights_tensor(
    node_order: Iterable[str], config: Dict[str, Any], device: torch.device
) -> torch.Tensor:
    weights = [float(config["physics_residual_weights"].get(node, 1.0)) for node in node_order]
    return _weights_to_tensor(weights, device, expand=False)


def main() -> None:
    args = parse_args_924()
    config = copy.deepcopy(DEFAULT_CONFIG_924)
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size

    set_seed_924(args.seed)
    device = _resolve_device(args.device)

    frame = load_measurements_924(str(args.data))
    prepared = prepare_datasets_924(frame)
    train_loader, val_loader, test_loader = create_dataloaders_924(prepared, config["batch_size"])

    input_dim = prepared.train_dataset.features.shape[1]
    output_dim = len(prepared.node_order)
    model = ThermalPINN924(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=config["hidden_layers"],
        activation=config["activation"],
        dropout=config["dropout"],
    ).to(device)
    graph = ThermalNetworkParameters924().to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(graph.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.6, patience=20, verbose=False
    )

    target_std_tensor = build_target_scaling_vectors(prepared.target_stats, prepared.node_order, device)
    target_mean_tensor = build_target_mean_vectors(prepared.target_stats, prepared.node_order, device)

    residual_weight_base = _prepare_residual_weights_tensor(prepared.node_order, config, device)
    base_observed_weights = _build_observed_weight_vector(prepared.observed_nodes, config)
    observed_weights = base_observed_weights.copy()

    pm_obs_index = (
        prepared.observed_nodes.index("permanent_magnet")
        if "permanent_magnet" in prepared.observed_nodes
        else None
    )
    rotor_obs_index = (
        prepared.observed_nodes.index("rotor_core")
        if "rotor_core" in prepared.observed_nodes
        else None
    )
    pm_node_index = (
        prepared.node_order.index("permanent_magnet")
        if "permanent_magnet" in prepared.node_order
        else None
    )
    rotor_node_index = (
        prepared.node_order.index("rotor_core")
        if "rotor_core" in prepared.node_order
        else None
    )

    base_coupling_weight = config["pm_rotor_coupling_weight"]
    config["pm_rotor_coupling_weight_initial"] = base_coupling_weight
    pm_multiplier = 1.0
    focus_cooldown = 0
    focus_history: List[Dict[str, Any]] = []

    observed_weight_tensor = _weights_to_tensor(observed_weights, device, expand=True)

    best_state = None
    best_metric = math.inf
    best_epoch = -1
    patience_counter = 0

    for epoch in range(config["epochs"]):
        observed_weight_tensor = _weights_to_tensor(observed_weights, device, expand=True)
        residual_weight_tensor = residual_weight_base.clone()
        if pm_node_index is not None:
            residual_weight_tensor[pm_node_index] *= 1.0 + 0.6 * max(pm_multiplier - 1.0, 0.0)
        if rotor_node_index is not None:
            residual_weight_tensor[rotor_node_index] *= 1.0 + 0.35 * max(pm_multiplier - 1.0, 0.0)

        config["pm_rotor_coupling_weight"] = base_coupling_weight * pm_multiplier

        train_stats = train_epoch_924(
            model,
            graph,
            train_loader,
            optimizer,
            device,
            prepared.observed_indices,
            observed_weight_tensor,
            residual_weight_tensor,
            target_std_tensor,
            target_mean_tensor,
            prepared.time_index,
            prepared.time_scaler["std"],
            config,
        )

        val_stats = evaluate_epoch_924(
            model,
            graph,
            val_loader,
            device,
            prepared.observed_indices,
            prepared.observed_nodes,
            observed_weight_tensor,
            residual_weight_tensor,
            target_std_tensor,
            target_mean_tensor,
            prepared.time_index,
            prepared.time_scaler["std"],
            prepared.time_scaler["mean"],
            config,
        )
        scheduler.step(val_stats["loss"])

        if focus_cooldown > 0:
            focus_cooldown -= 1

        pm_r2 = val_stats["metrics"].get("permanent_magnet", {}).get("r2")
        if pm_obs_index is not None and pm_r2 is not None and not math.isnan(pm_r2):
            if pm_r2 < config["pm_focus_target_r2"] and focus_cooldown == 0:
                pm_multiplier = min(
                    config["pm_focus_max_multiplier"], pm_multiplier + config["pm_focus_step"]
                )
                observed_weights[pm_obs_index] = base_observed_weights[pm_obs_index] * pm_multiplier
                if rotor_obs_index is not None:
                    rotor_base = base_observed_weights[rotor_obs_index]
                    rotor_boost = 1.0 + 0.35 * max(pm_multiplier - 1.0, 0.0)
                    observed_weights[rotor_obs_index] = max(rotor_base, rotor_base * rotor_boost)
                focus_cooldown = config["pm_focus_cooldown"]
            elif pm_r2 >= config["pm_focus_relax_r2"] and pm_multiplier > 1.0:
                pm_multiplier = max(1.0, pm_multiplier * 0.92)
                observed_weights[pm_obs_index] = base_observed_weights[pm_obs_index] * pm_multiplier
                if rotor_obs_index is not None:
                    rotor_base = base_observed_weights[rotor_obs_index]
                    rotor_boost = 1.0 + 0.35 * max(pm_multiplier - 1.0, 0.0)
                    observed_weights[rotor_obs_index] = max(rotor_base, rotor_base * rotor_boost)

        focus_history.append(
            {
                "epoch": int(epoch),
                "pm_multiplier": float(pm_multiplier),
                "val_pm_r2": None
                if pm_r2 is None or math.isnan(pm_r2)
                else float(pm_r2),
                "data_weight_pm": float(observed_weights[pm_obs_index])
                if pm_obs_index is not None
                else None,
                "coupling_weight": float(config["pm_rotor_coupling_weight"]),
            }
        )

        metric_key = val_stats["loss"] if pm_r2 is None or math.isnan(pm_r2) else (1.0 - pm_r2)

        improved = metric_key < best_metric
        if improved:
            best_metric = metric_key
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "graph": copy.deepcopy(graph.state_dict()),
                "epoch": epoch,
                "train": train_stats,
                "val": val_stats,
            }
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state["model"])
    graph.load_state_dict(best_state["graph"])

    observed_weight_tensor = _weights_to_tensor(observed_weights, device, expand=True)
    residual_weight_tensor = residual_weight_base.clone()
    if pm_node_index is not None:
        residual_weight_tensor[pm_node_index] *= 1.0 + 0.6 * max(pm_multiplier - 1.0, 0.0)
    if rotor_node_index is not None:
        residual_weight_tensor[rotor_node_index] *= 1.0 + 0.35 * max(pm_multiplier - 1.0, 0.0)
    test_stats = evaluate_epoch_924(
        model,
        graph,
        test_loader,
        device,
        prepared.observed_indices,
        prepared.observed_nodes,
        observed_weight_tensor,
        residual_weight_tensor,
        target_std_tensor,
        target_mean_tensor,
        prepared.time_index,
        prepared.time_scaler["std"],
        prepared.time_scaler["mean"],
        config,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config["pm_rotor_coupling_weight_final"] = config["pm_rotor_coupling_weight"]
    config["pm_focus_final_multiplier"] = pm_multiplier

    adaptive_weight_report = {
        node: float(observed_weights[idx]) for idx, node in enumerate(prepared.observed_nodes)
    }

    metrics_payload = {
        "config": config,
        "best_epoch": best_epoch,
        "train": best_state["train"],
        "validation": best_state["val"],
        "test": {
            "loss": test_stats["loss"],
            "data_loss": test_stats["data_loss"],
            "physics_loss": test_stats["physics_loss"],
            "coupling_loss": test_stats["coupling_loss"],
            "metrics": test_stats["metrics"],
        },
        "thermal_parameters": graph.to_dict(),
        "pm_focus_history": focus_history,
        "adaptive_data_loss_weights": adaptive_weight_report,
    }

    metrics_path = output_dir / "pinn_metrics_924.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, indent=2)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "graph_state_dict": graph.state_dict(),
            "feature_columns": prepared.feature_columns,
            "feature_scaler": {
                "mean": prepared.feature_scaler.mean_.tolist(),
                "scale": prepared.feature_scaler.scale_.tolist(),
                "var": prepared.feature_scaler.var_.tolist(),
            },
            "target_stats": prepared.target_stats,
            "target_columns": prepared.target_columns,
            "time_scaler": prepared.time_scaler,
            "node_order": prepared.node_order,
        },
        output_dir / "tnn_state_dict_924.pt",
    )

    with (output_dir / "thermal_parameters_924.json").open("w", encoding="utf-8") as fp:
        json.dump(graph.to_dict(), fp, indent=2)

    plot_topology_924(graph, output_dir / "tnn_topology_924.png")

    if prepared.observed_indices:
        time_axis = test_stats["time_axis"]
        predictions = test_stats["predictions"]
        targets = test_stats["targets"]
        ordering = np.argsort(time_axis)
        plot_temperature_profiles_924(
            time_axis=time_axis[ordering],
            targets=targets[ordering],
            predictions=predictions[ordering],
            observed_nodes=prepared.observed_nodes,
            save_path=output_dir / "pinn_test_profiles_924.png",
        )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

