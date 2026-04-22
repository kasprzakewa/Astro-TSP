from __future__ import annotations

import time
from itertools import permutations

from astrotsp.models.problem import SolverResult, TSPInstance
from astrotsp.solvers.base import TSPSolver


class BranchAndBoundSolver(TSPSolver):
    name = "branch_and_bound"

    def solve(self, instance: TSPInstance, seed: int | None = None) -> SolverResult:
        del seed
        start_time = time.perf_counter()
        n = len(instance.nodes)
        if n < 2:
            raise ValueError("Instance must contain at least 2 nodes.")

        best_route: list[int] | None = None
        best_cost = float("inf")
        fixed_start = 0

        for perm in permutations(range(1, n)):
            route = [fixed_start, *perm]
            cost = 0.0
            for idx in range(n):
                src = route[idx]
                dst = route[(idx + 1) % n]
                cost += float(instance.costs[src, dst])
                if cost >= best_cost:
                    break
            if cost < best_cost:
                best_cost = cost
                best_route = list(route)

        elapsed = time.perf_counter() - start_time
        if best_route is None:
            raise RuntimeError("No route found.")

        return SolverResult(
            solver_name=self.name,
            route=best_route,
            total_cost=best_cost,
            elapsed_seconds=elapsed,
            status="optimal",
            metadata={"method": "exact_enumeration_pruned"},
        )
