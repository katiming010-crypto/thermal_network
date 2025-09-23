"""Neural network architectures used by the PINN trainer."""

from __future__ import annotations

import math
from typing import Iterable, List

import torch
from torch import nn


def _activation_from_name(name: str) -> nn.Module:
    name = name.lower()
    if name in {"relu"}:
        return nn.ReLU()
    if name in {"gelu"}:
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name in {"tanh"}:
        return nn.Tanh()
    raise ValueError(f"Unsupported activation '{name}'")


class ThermalPINN924(nn.Module):
    """Feed-forward regressor that outputs all dynamic node temperatures."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: Iterable[int],
        activation: str = "silu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        previous_dim = input_dim
        act_module = _activation_from_name(activation)
        for width in hidden_layers:
            layers.append(nn.Linear(previous_dim, width))
            layers.append(act_module.__class__())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_dim = width
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

