from __future__ import annotations

from datetime import datetime

import numpy as np

from astrotsp.models.problem import CelestialBody, TSPInstance


def build_euclidean_cost_matrix(nodes: list[CelestialBody]) -> np.ndarray:
    n = len(nodes)
    if n < 2:
        raise ValueError("At least 2 nodes are required.")

    matrix = np.zeros((n, n), dtype=float)
    for i, src in enumerate(nodes):
        for j, dst in enumerate(nodes):
            if i == j:
                continue
            matrix[i, j] = np.sqrt(
                (dst.x - src.x) ** 2 + (dst.y - src.y) ** 2 + (dst.z - src.z) ** 2
            )
    return matrix


def build_instance(nodes: list[CelestialBody], epoch: datetime) -> TSPInstance:
    costs = build_euclidean_cost_matrix(nodes)
    instance = TSPInstance(epoch=epoch, nodes=nodes, costs=costs, directed=False)
    instance.validate()
    return instance
