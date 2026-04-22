from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class CelestialBody:
    name: str
    horizons_id: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class TSPInstance:
    epoch: datetime
    nodes: list[CelestialBody]
    costs: np.ndarray
    directed: bool = False

    def validate(self) -> None:
        size = len(self.nodes)
        if self.costs.shape != (size, size):
            raise ValueError("Cost matrix shape does not match node count.")
        if np.any(np.diag(self.costs) != 0.0):
            raise ValueError("Diagonal of cost matrix must be zero.")
        if not self.directed and not np.allclose(self.costs, self.costs.T):
            raise ValueError("Undirected instance requires symmetric costs.")


@dataclass(frozen=True)
class SolverResult:
    solver_name: str
    route: list[int]
    total_cost: float
    elapsed_seconds: float
    status: str
    metadata: dict[str, float | int | str] = field(default_factory=dict)
