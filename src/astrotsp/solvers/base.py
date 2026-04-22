from __future__ import annotations

from abc import ABC, abstractmethod

from astrotsp.models.problem import SolverResult, TSPInstance


class TSPSolver(ABC):
    name: str

    @abstractmethod
    def solve(self, instance: TSPInstance, seed: int | None = None) -> SolverResult:
        raise NotImplementedError
