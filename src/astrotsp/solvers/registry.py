from __future__ import annotations

from astrotsp.solvers.aco import ACOSolver
from astrotsp.solvers.base import TSPSolver
from astrotsp.solvers.branch_and_bound import BranchAndBoundSolver
from astrotsp.solvers.ilp import ILPSolver


def build_solver(name: str) -> TSPSolver:
    normalized = name.strip().lower()
    if normalized == "branch_and_bound":
        return BranchAndBoundSolver()
    if normalized == "ilp":
        return ILPSolver()
    if normalized == "aco":
        return ACOSolver()
    raise ValueError(f"Unsupported solver '{name}'.")
