"""Thermal network parameterisation and helper utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class PositiveParameter924(nn.Module):
    """Ensures the stored value remains strictly positive."""

    def __init__(self, initial_value: float, minimum: float = 1e-4) -> None:
        super().__init__()
        if initial_value <= 0:
            raise ValueError("Initial value for PositiveParameter924 must be positive")
        init = torch.as_tensor(initial_value, dtype=torch.float32)
        self.raw = nn.Parameter(torch.log(torch.expm1(init)))
        self.minimum = float(minimum)

    def forward(self) -> torch.Tensor:  # pragma: no cover - simple wrapper
        return F.softplus(self.raw) + self.minimum

    def value(self) -> torch.Tensor:
        return self.forward()


@dataclass
class ThermalEdge924:
    node_a: str
    node_b: str
    name: str


class ThermalNetworkParameters924(nn.Module):
    """Stores the thermal resistances and capacitances as trainable parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.node_order: List[str] = [
            "coolant_channel",
            "stator_yoke",
            "stator_tooth",
            "stator_winding",
            "rotor_core",
            "permanent_magnet",
            "bearing",
        ]
        self.sink_node: str = "coolant_sink"

        self.edges: List[ThermalEdge924] = [
            ThermalEdge924("bearing", "coolant_channel", "Rb_c"),
            ThermalEdge924("bearing", "permanent_magnet", "Rb_m"),
            ThermalEdge924("permanent_magnet", "rotor_core", "Rm_r"),
            ThermalEdge924("rotor_core", "stator_tooth", "Rr_st"),
            ThermalEdge924("rotor_core", "stator_winding", "Rr_w"),
            ThermalEdge924("stator_tooth", "stator_winding", "Rst_w"),
            ThermalEdge924("stator_tooth", "stator_yoke", "Rst_sy"),
            ThermalEdge924("stator_yoke", "coolant_channel", "Rsy_c"),
            ThermalEdge924("stator_winding", "stator_yoke", "Rw_sy"),
            ThermalEdge924("coolant_channel", self.sink_node, "Rv"),
        ]

        initial_resistances = {
            "Rb_c": 0.18,
            "Rb_m": 0.25,
            "Rm_r": 0.12,
            "Rr_st": 0.08,
            "Rr_w": 0.14,
            "Rst_w": 0.05,
            "Rst_sy": 0.07,
            "Rsy_c": 0.09,
            "Rw_sy": 0.06,
            "Rv": 0.03,
        }

        initial_capacitances = {
            "coolant_channel": 210.0,
            "stator_yoke": 540.0,
            "stator_tooth": 320.0,
            "stator_winding": 400.0,
            "rotor_core": 250.0,
            "permanent_magnet": 180.0,
            "bearing": 160.0,
        }

        self.resistances = nn.ModuleDict(
            {
                edge.name: PositiveParameter924(initial_resistances[edge.name])
                for edge in self.edges
            }
        )

        self.capacitances = nn.ModuleDict(
            {
                node: PositiveParameter924(initial_capacitances[node])
                for node in self.node_order
            }
        )

        self._neighbour_cache: Dict[str, List[str]] = {}

    def _edge_key(self, node_a: str, node_b: str) -> str:
        for edge in self.edges:
            if {edge.node_a, edge.node_b} == {node_a, node_b}:
                return edge.name
        raise KeyError(f"No thermal connection between {node_a} and {node_b}")

    def resistance(self, node_a: str, node_b: str) -> torch.Tensor:
        key = self._edge_key(node_a, node_b)
        return self.resistances[key].value()

    def capacitance(self, node: str) -> torch.Tensor:
        return self.capacitances[node].value()

    def neighbours(self, node: str) -> List[str]:
        if node in self._neighbour_cache:
            return self._neighbour_cache[node]

        connected: List[str] = []
        for edge in self.edges:
            if edge.node_a == node:
                connected.append(edge.node_b)
            elif edge.node_b == node:
                connected.append(edge.node_a)
        self._neighbour_cache[node] = connected
        return connected

    def resistance_dict(self) -> Dict[str, float]:
        return {name: float(module.value().item()) for name, module in self.resistances.items()}

    def capacitance_dict(self) -> Dict[str, float]:
        return {name: float(module.value().item()) for name, module in self.capacitances.items()}

    def parameter_regularisation(self) -> torch.Tensor:
        reg_terms: List[torch.Tensor] = []
        for module in self.resistances.values():
            value = module.value()
            reg_terms.append((value.log() ** 2))
        for module in self.capacitances.values():
            value = module.value()
            reg_terms.append((value.log() ** 2))
        return torch.stack(reg_terms).mean()

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            "resistances": self.resistance_dict(),
            "capacitances": self.capacitance_dict(),
        }


def stack_node_parameters(
    graph: ThermalNetworkParameters924,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    capacitances = torch.stack([graph.capacitance(node) for node in graph.node_order]).to(device)
    resistances = torch.stack([module.value() for module in graph.resistances.values()]).to(device)
    return capacitances, resistances

