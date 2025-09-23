"""Plotting helpers for the thermal PINN workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_topology_924(graph, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    g = nx.Graph()
    for node in graph.node_order + [graph.sink_node]:
        g.add_node(node)
    for edge in graph.edges:
        g.add_edge(edge.node_a, edge.node_b, label=edge.name)

    layout: Dict[str, np.ndarray] = {}
    x_positions = {
        "coolant_channel": 0,
        "stator_yoke": 1,
        "stator_tooth": 2,
        "stator_winding": 3,
        "rotor_core": 4,
        "permanent_magnet": 5,
        "bearing": 6,
        graph.sink_node: -1,
    }
    for node, x in x_positions.items():
        layout[node] = np.array([x, 0 if node != graph.sink_node else -1])

    plt.figure(figsize=(10, 4))
    nx.draw_networkx(
        g,
        pos=layout,
        with_labels=True,
        node_color="#1f78b4",
        font_weight="bold",
        node_size=1200,
    )
    edge_labels = {(edge.node_a, edge.node_b): edge.name for edge in graph.edges}
    nx.draw_networkx_edge_labels(g, pos=layout, edge_labels=edge_labels)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_temperature_profiles_924(
    time_axis: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    observed_nodes: Iterable[str],
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    num_nodes = len(list(observed_nodes))
    cols = min(2, num_nodes)
    rows = int(np.ceil(num_nodes / cols))
    plt.figure(figsize=(6 * cols, 3 * rows))
    observed_nodes = list(observed_nodes)

    for idx, node in enumerate(observed_nodes):
        ax = plt.subplot(rows, cols, idx + 1)
        ax.plot(time_axis, targets[:, idx], label="Measured", linewidth=2)
        ax.plot(time_axis, predictions[:, idx], label="Predicted", linestyle="--", linewidth=2)
        ax.set_title(node)
        ax.set_xlabel("Time")
        ax.set_ylabel("Temperature [°C]")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

