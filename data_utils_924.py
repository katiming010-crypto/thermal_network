"""Utility helpers for preparing thermal network datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


# Canonical dynamic node ordering.
DYNAMIC_NODES_924: List[str] = [
    "coolant_channel",
    "stator_yoke",
    "stator_tooth",
    "stator_winding",
    "rotor_core",
    "permanent_magnet",
    "bearing",
]


NODE_ALIASES_924: Dict[str, Sequence[str]] = {
    "coolant_channel": ("coolant", "tc", "coolant_channel", "coolant_temperature"),
    "stator_yoke": ("stator_yoke", "tsy", "yoke", "ty"),
    "stator_tooth": ("stator_tooth", "tst", "tooth"),
    "stator_winding": (
        "stator_winding",
        "winding",
        "tw",
        "stator_coil",
        "phase_winding",
    ),
    "rotor_core": ("rotor_core", "tr", "rotor", "core"),
    "permanent_magnet": ("permanent_magnet", "pm", "tm", "magnet"),
    "bearing": ("bearing", "tb", "shaft", "bearing_temperature"),
}


POWER_ALIASES_924: Dict[str, Sequence[str]] = {
    "coolant_channel": ("loss_coolant", "coolant_loss", "pc"),
    "stator_yoke": ("loss_stator_yoke", "stator_yoke_loss", "psy"),
    "stator_tooth": ("loss_stator_tooth", "stator_tooth_loss", "pst"),
    "stator_winding": ("loss_stator_winding", "stator_winding_loss", "pw"),
    "rotor_core": ("loss_rotor", "rotor_loss", "pr"),
    "permanent_magnet": (
        "loss_pm",
        "pm_loss",
        "permanent_magnet_loss",
        "pm_heat_loss",
        "ppm",
    ),
    "bearing": ("loss_bearing", "bearing_loss", "pb"),
}


SINK_ALIASES_924: Sequence[str] = (
    "coolant_sink",
    "tk",
    "coolant_inlet",
    "coolant_outlet",
    "coolant_temp_in",
    "coolant_temp_out",
    "ambient",
    "ambient_temperature",
)

TIME_ALIASES_924: Sequence[str] = ("time", "timestamp", "t", "seconds", "sample")
PROFILE_ALIASES_924: Sequence[str] = ("profile", "profile_id", "cycle", "test_id")


def _normalise_column_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _find_first_match(
    columns: Iterable[str],
    aliases: Sequence[str],
    *,
    exclude: Optional[Iterable[str]] = None,
) -> Optional[str]:
    normalised = {_normalise_column_name(col): col for col in columns}
    excluded_normalised = {
        _normalise_column_name(column)
        for column in (exclude or [])
        if column is not None
    }
    for alias in aliases:
        alias_norm = _normalise_column_name(alias)
        if alias_norm in normalised:
            candidate = normalised[alias_norm]
            if _normalise_column_name(candidate) not in excluded_normalised:
                return candidate

    for alias in aliases:
        alias_norm = _normalise_column_name(alias)
        for norm_col, original in normalised.items():
            if alias_norm in norm_col and norm_col not in excluded_normalised:
                return original
    return None


class ThermalDataset924(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        target_norm: np.ndarray,
        target_actual: np.ndarray,
        powers: np.ndarray,
        sink_temperature: np.ndarray,
    ) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.target_norm = torch.as_tensor(target_norm, dtype=torch.float32)
        self.target_actual = torch.as_tensor(target_actual, dtype=torch.float32)
        self.powers = torch.as_tensor(powers, dtype=torch.float32)
        self.sink_temperature = torch.as_tensor(sink_temperature, dtype=torch.float32)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.features.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.features[index],
            "target_norm": self.target_norm[index],
            "target_actual": self.target_actual[index],
            "powers": self.powers[index],
            "sink_temperature": self.sink_temperature[index],
        }


@dataclass
class PreparedData924:
    train_dataset: ThermalDataset924
    val_dataset: ThermalDataset924
    test_dataset: ThermalDataset924
    feature_scaler: StandardScaler
    target_stats: Dict[str, Dict[str, float]]
    target_columns: Dict[str, Optional[str]]
    time_index: int
    node_order: List[str]
    observed_nodes: List[str]
    observed_indices: List[int]
    feature_columns: List[str]
    power_columns: Dict[str, Optional[str]]
    time_scaler: Dict[str, float]


def load_measurements_924(csv_path: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.interpolate(method="linear").ffill().bfill()
    return frame


def _prepare_targets(
    frame: pd.DataFrame,
    node_order: Sequence[str],
) -> Tuple[
    Dict[str, Dict[str, float]],
    Dict[str, Optional[str]],
    Dict[str, Optional[str]],
    np.ndarray,
    np.ndarray,
    List[str],
    List[int],
]:
    target_stats: Dict[str, Dict[str, float]] = {}
    target_columns: Dict[str, Optional[str]] = {}
    observed_nodes: List[str] = []
    observed_indices: List[int] = []
    target_norm_columns: List[np.ndarray] = []
    target_actual_columns: List[np.ndarray] = []

    for node_index, node in enumerate(node_order):
        column = _find_first_match(frame.columns, NODE_ALIASES_924[node])
        target_columns[node] = column
        if column is None:
            target_stats[node] = {"mean": 0.0, "std": 1.0}
            continue

        values = frame[column].to_numpy(dtype=float)
        mean_value = float(np.nanmean(values))
        std_value = float(np.nanstd(values))
        if np.isclose(std_value, 0.0):
            std_value = 1.0

        target_stats[node] = {"mean": mean_value, "std": std_value}
        observed_nodes.append(node)
        observed_indices.append(node_index)
        target_norm_columns.append((values - mean_value) / std_value)
        target_actual_columns.append(values)

    if not observed_nodes:
        raise ValueError(
            "No temperature columns could be matched. Please check the dataset "
            "or extend NODE_ALIASES_924."
        )

    target_norm = np.column_stack(target_norm_columns)
    target_actual = np.column_stack(target_actual_columns)

    power_columns: Dict[str, Optional[str]] = {}
    for node in node_order:
        power_columns[node] = _find_first_match(
            frame.columns,
            POWER_ALIASES_924.get(node, ()),
            exclude={target_columns.get(node)},
        )

    return (
        target_stats,
        power_columns,
        target_columns,
        target_norm,
        target_actual,
        observed_nodes,
        observed_indices,
    )


def _extract_sink_temperature(frame: pd.DataFrame) -> np.ndarray:
    column = _find_first_match(frame.columns, SINK_ALIASES_924)
    if column is None:
        return np.full((len(frame), 1), 25.0, dtype=float)
    return frame[column].to_numpy(dtype=float).reshape(-1, 1)


def _resolve_time_column(frame: pd.DataFrame) -> Tuple[str, np.ndarray]:
    column = _find_first_match(frame.columns, TIME_ALIASES_924)
    if column is not None:
        return column, frame[column].to_numpy(dtype=float)

    synthetic_time = np.arange(len(frame), dtype=float)
    return "__synthetic_time__", synthetic_time


def _resolve_profile_column(frame: pd.DataFrame) -> Optional[str]:
    return _find_first_match(frame.columns, PROFILE_ALIASES_924)


def _build_power_tensor(
    frame: pd.DataFrame,
    node_order: Sequence[str],
    power_columns: Dict[str, Optional[str]],
) -> np.ndarray:
    powers = np.zeros((len(frame), len(node_order)), dtype=float)
    for index, node in enumerate(node_order):
        column = power_columns.get(node)
        if column is None:
            continue
        powers[:, index] = frame[column].to_numpy(dtype=float)
    return powers


def _prepare_features(
    frame: pd.DataFrame,
    feature_columns: List[str],
) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    features = scaler.fit_transform(frame[feature_columns].to_numpy(dtype=float))
    return features, scaler


def prepare_datasets_924(
    frame: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 924,
) -> PreparedData924:
    node_order = list(DYNAMIC_NODES_924)

    (
        target_stats,
        power_columns,
        target_columns,
        target_norm,
        target_actual,
        observed_nodes,
        observed_indices,
    ) = _prepare_targets(frame, node_order)

    sink_temperature = _extract_sink_temperature(frame)
    time_column, time_values = _resolve_time_column(frame)
    profile_column = _resolve_profile_column(frame)

    numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    excluded = set()
    for column in target_columns.values():
        if column:
            excluded.add(column)
    for column in power_columns.values():
        if column:
            excluded.add(column)
    if time_column in numeric_columns:
        excluded.add(time_column)
    if profile_column in numeric_columns:
        excluded.add(profile_column)

    feature_columns = [col for col in numeric_columns if col not in excluded]
    feature_frame = frame[feature_columns].copy()
    feature_frame[time_column] = time_values
    feature_columns.append(time_column)

    features, feature_scaler = _prepare_features(feature_frame, feature_columns)
    time_index = feature_columns.index(time_column)

    time_mean = float(feature_scaler.mean_[time_index])
    time_std = float(feature_scaler.scale_[time_index])
    if np.isclose(time_std, 0.0):
        time_std = 1.0

    powers = _build_power_tensor(frame, node_order, power_columns)

    indices = np.arange(len(frame))
    if profile_column is not None:
        groups = frame[profile_column].to_numpy()
        unique_profiles = np.unique(groups)
        train_profiles, test_profiles = train_test_split(
            unique_profiles,
            test_size=test_size,
            random_state=random_state,
        )
        train_profiles, val_profiles = train_test_split(
            train_profiles,
            test_size=val_size,
            random_state=random_state,
        )

        train_mask = np.isin(groups, train_profiles)
        val_mask = np.isin(groups, val_profiles)
        test_mask = np.isin(groups, test_profiles)
        train_indices = indices[train_mask]
        val_indices = indices[val_mask]
        test_indices = indices[test_mask]
    else:
        train_indices, test_indices = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
        train_indices, val_indices = train_test_split(
            train_indices,
            test_size=val_size,
            random_state=random_state,
            shuffle=True,
        )

    def _slice(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return arr[idx] if arr.size else np.empty((len(idx), 0), dtype=float)

    train_dataset = ThermalDataset924(
        features[train_indices],
        _slice(target_norm, train_indices),
        _slice(target_actual, train_indices),
        powers[train_indices],
        sink_temperature[train_indices],
    )
    val_dataset = ThermalDataset924(
        features[val_indices],
        _slice(target_norm, val_indices),
        _slice(target_actual, val_indices),
        powers[val_indices],
        sink_temperature[val_indices],
    )
    test_dataset = ThermalDataset924(
        features[test_indices],
        _slice(target_norm, test_indices),
        _slice(target_actual, test_indices),
        powers[test_indices],
        sink_temperature[test_indices],
    )

    return PreparedData924(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        feature_scaler=feature_scaler,
        target_stats=target_stats,
        target_columns=target_columns,
        time_index=time_index,
        node_order=node_order,
        observed_nodes=observed_nodes,
        observed_indices=observed_indices,
        feature_columns=feature_columns,
        power_columns=power_columns,
        time_scaler={"mean": time_mean, "std": time_std},
    )

