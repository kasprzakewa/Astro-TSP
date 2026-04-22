from __future__ import annotations

import time

import pulp

from astrotsp.models.problem import SolverResult, TSPInstance
from astrotsp.solvers.base import TSPSolver


class ILPSolver(TSPSolver):
    name = "ilp"

    def solve(self, instance: TSPInstance, seed: int | None = None) -> SolverResult:
        del seed
        start_time = time.perf_counter()
        n = len(instance.nodes)
        nodes = range(n)

        model = pulp.LpProblem("tsp", pulp.LpMinimize)
        x = pulp.LpVariable.dicts("x", (nodes, nodes), lowBound=0, upBound=1, cat=pulp.LpBinary)
        u = pulp.LpVariable.dicts("u", nodes, lowBound=0, upBound=n - 1, cat=pulp.LpContinuous)

        model += pulp.lpSum(instance.costs[i, j] * x[i][j] for i in nodes for j in nodes if i != j)

        for i in nodes:
            model += pulp.lpSum(x[i][j] for j in nodes if i != j) == 1
            model += pulp.lpSum(x[j][i] for j in nodes if i != j) == 1
            model += x[i][i] == 0

        for i in nodes:
            for j in nodes:
                if i != j and i != 0 and j != 0:
                    model += u[i] - u[j] + n * x[i][j] <= n - 1

        solver = pulp.PULP_CBC_CMD(msg=False)
        model.solve(solver)
        status = pulp.LpStatus[model.status].lower()
        if status not in {"optimal", "feasible"}:
            raise RuntimeError(f"ILP solver ended with status: {status}")

        successor: dict[int, int] = {}
        for i in nodes:
            for j in nodes:
                if i != j and pulp.value(x[i][j]) > 0.5:
                    successor[i] = j

        route = [0]
        while len(route) < n:
            route.append(successor[route[-1]])

        elapsed = time.perf_counter() - start_time
        total_cost = sum(instance.costs[route[k], route[(k + 1) % n]] for k in range(n))
        return SolverResult(
            solver_name=self.name,
            route=route,
            total_cost=float(total_cost),
            elapsed_seconds=elapsed,
            status=status,
            metadata={"backend": "cbc"},
        )
