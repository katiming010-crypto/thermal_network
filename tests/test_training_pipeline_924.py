from __future__ import annotations

import copy

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")

from data_utils_924 import prepare_datasets_924
from physics_utils_924 import build_target_mean_vectors, build_target_scaling_vectors
from train_pinn_924 import (
    DEFAULT_CONFIG_924,
    _forward_batch_924,
    _prepare_weights_tensor,
    _prepare_residual_weights_tensor,
    create_dataloaders_924,
    train_epoch_924,
)
from thermal_graph_924 import ThermalNetworkParameters924
from pinn_model_924 import ThermalPINN924


def _synthetic_frame_924(num_samples: int = 180) -> pd.DataFrame:
    time = np.linspace(0.0, 60.0, num_samples)
    frame = pd.DataFrame(
        {
            "time": time,
            "coolant_temperature": 30.0 + 2.0 * np.sin(0.05 * time),
            "stator_yoke": 70.0 + 5.0 * np.sin(0.1 * time),
            "stator_tooth": 75.0 + 4.5 * np.sin(0.1 * time + 0.2),
            "stator_winding": 85.0 + 4.0 * np.sin(0.1 * time + 0.4),
            "rotor_core": 65.0 + 3.5 * np.sin(0.09 * time + 0.3),
            "pm_temp": 55.0 + 3.0 * np.sin(0.08 * time + 0.5),
            "bearing_temperature": 50.0 + 2.5 * np.sin(0.12 * time + 0.1),
            "coolant_loss": 10.0 + 0.5 * np.cos(0.05 * time),
            "loss_stator_yoke": 90.0 + 2.0 * np.cos(0.07 * time),
            "loss_stator_tooth": 95.0 + 1.8 * np.cos(0.06 * time + 0.1),
            "loss_stator_winding": 120.0 + 2.5 * np.cos(0.05 * time + 0.2),
            "loss_rotor": 80.0 + 1.5 * np.cos(0.04 * time + 0.3),
            "loss_pm": 60.0 + 1.2 * np.cos(0.04 * time + 0.4),
            "loss_bearing": 40.0 + 0.9 * np.cos(0.05 * time + 0.2),
            "speed_rpm": 1500.0 + 100.0 * np.sin(0.03 * time),
            "torque_nm": 50.0 + 5.0 * np.cos(0.02 * time),
        }
    )
    frame["profile_id"] = (time // (time.max() / 6)).astype(int)
    frame["coolant_inlet"] = 25.0 + 0.3 * np.sin(0.05 * time + 0.3)
    return frame


def test_prepare_datasets_aliases_924() -> None:
    frame = _synthetic_frame_924()
    prepared = prepare_datasets_924(frame, test_size=0.2, val_size=0.2, random_state=1)

    assert "permanent_magnet" in prepared.observed_nodes
    assert prepared.target_columns["permanent_magnet"] is not None
    assert prepared.train_dataset.features.shape[1] == len(prepared.feature_columns)


def test_forward_batch_and_training_loop_924() -> None:
    frame = _synthetic_frame_924()
    prepared = prepare_datasets_924(frame, test_size=0.2, val_size=0.2, random_state=2)
    train_loader, _, _ = create_dataloaders_924(prepared, batch_size=32)

    model = ThermalPINN924(
        input_dim=prepared.train_dataset.features.shape[1],
        output_dim=len(prepared.node_order),
        hidden_layers=[64, 64],
        activation="silu",
        dropout=0.0,
    )
    graph = ThermalNetworkParameters924()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(graph.parameters()), lr=1e-3
    )

    config = copy.deepcopy(DEFAULT_CONFIG_924)
    config.update(
        {
            "physics_weight": 0.2,
            "parameter_reg_weight": 1e-4,
            "pm_rotor_coupling_weight": 0.01,
            "gradient_clip": 0.0,
        }
    )

    device = torch.device("cpu")
    target_std_tensor = build_target_scaling_vectors(prepared.target_stats, prepared.node_order, device)
    target_mean_tensor = build_target_mean_vectors(prepared.target_stats, prepared.node_order, device)
    observed_weight_tensor = _prepare_weights_tensor(prepared.observed_nodes, config, device)
    residual_weight_tensor = _prepare_residual_weights_tensor(prepared.node_order, config, device)

    batch = next(iter(train_loader))
    result = _forward_batch_924(
        model,
        graph,
        batch,
        device,
        prepared.observed_indices,
        observed_weight_tensor,
        residual_weight_tensor,
        target_std_tensor,
        target_mean_tensor,
        prepared.time_index,
        prepared.time_scaler["std"],
        config,
        compute_predictions=True,
    )

    assert torch.isfinite(result["loss"]).all()
    assert result["temperatures"].shape[1] == len(prepared.node_order)

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

    assert np.isfinite(train_stats["loss"])
    assert np.isfinite(train_stats["physics_loss"])
