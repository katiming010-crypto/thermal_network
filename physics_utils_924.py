"""Physics helpers for the thermal PINN."""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch


def build_target_scaling_vectors(
    target_stats: Dict[str, Dict[str, float]],
    node_order: Iterable[str],
    device: torch.device,
) -> torch.Tensor:
    stds = [float(target_stats[node]["std"]) for node in node_order]
    return torch.as_tensor(stds, dtype=torch.float32, device=device)


def build_target_mean_vectors(
    target_stats: Dict[str, Dict[str, float]],
    node_order: Iterable[str],
    device: torch.device,
) -> torch.Tensor:
    means = [float(target_stats[node]["mean"]) for node in node_order]
    return torch.as_tensor(means, dtype=torch.float32, device=device)


def compute_energy_balance_residuals(
    temperatures: torch.Tensor,
    dtemperatures_dt: torch.Tensor,
    powers: torch.Tensor,
    sink_temperatures: torch.Tensor,
    graph,
) -> List[torch.Tensor]:
    """Return residual tensors for each dynamic node in the graph."""

    residuals: List[torch.Tensor] = []
    node_to_index = {node: idx for idx, node in enumerate(graph.node_order)}

    for node_index, node in enumerate(graph.node_order):
        capacity = graph.capacitance(node)
        power = powers[:, node_index]
        derivative = capacity * dtemperatures_dt[:, node_index]
        conduction = torch.zeros_like(power)

        for neighbour in graph.neighbours(node):
            resistance = graph.resistance(node, neighbour)
            if neighbour == graph.sink_node:
                neighbour_temp = sink_temperatures.squeeze(-1)
            else:
                neighbour_temp = temperatures[:, node_to_index[neighbour]]
            conduction = conduction + (neighbour_temp - temperatures[:, node_index]) / resistance

        residuals.append(derivative - (power + conduction))

    return residuals

