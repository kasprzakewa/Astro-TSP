from __future__ import annotations

import random
import time

import numpy as np

from astrotsp.models.problem import SolverResult, TSPInstance
from astrotsp.solvers.base import TSPSolver
from astrotsp.solvers.utils import route_cost


class ACOSolver(TSPSolver):
    name = "aco"

    def __init__(
        self,
        iterations: int = 150,
        ants: int = 30,
        alpha: float = 1.0,
        beta: float = 2.5,
        evaporation: float = 0.4,
        q: float = 100.0,
    ) -> None:
        self.iterations = iterations
        self.ants = ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.q = q

    def solve(self, instance: TSPInstance, seed: int | None = None) -> SolverResult:
        start_time = time.perf_counter()
        rng = random.Random(seed)
        n = len(instance.nodes)

        pheromone = np.ones((n, n), dtype=float)
        heuristic = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i != j:
                    heuristic[i, j] = 1.0 / max(instance.costs[i, j], 1e-12)

        best_route: list[int] | None = None
        best_cost = float("inf")

        for _ in range(self.iterations):
            iteration_routes: list[list[int]] = []
            iteration_costs: list[float] = []

            for _ in range(self.ants):
                route = self._construct_route(rng, pheromone, heuristic, n)
                cost = route_cost(instance, route)
                iteration_routes.append(route)
                iteration_costs.append(cost)
                if cost < best_cost:
                    best_cost = cost
                    best_route = route

            pheromone *= (1.0 - self.evaporation)
            for route, cost in zip(iteration_routes, iteration_costs):
                deposit = self.q / max(cost, 1e-12)
                for k in range(n):
                    i = route[k]
                    j = route[(k + 1) % n]
                    pheromone[i, j] += deposit
                    pheromone[j, i] += deposit

        elapsed = time.perf_counter() - start_time
        if best_route is None:
            raise RuntimeError("ACO did not return any route.")

        return SolverResult(
            solver_name=self.name,
            route=best_route,
            total_cost=float(best_cost),
            elapsed_seconds=elapsed,
            status="feasible",
            metadata={"iterations": self.iterations, "ants": self.ants},
        )

    def _construct_route(
        self,
        rng: random.Random,
        pheromone: np.ndarray,
        heuristic: np.ndarray,
        n: int,
    ) -> list[int]:
        route = [0]
        unvisited = set(range(1, n))
        while unvisited:
            current = route[-1]
            candidates = list(unvisited)
            weights = []
            for nxt in candidates:
                tau = pheromone[current, nxt] ** self.alpha
                eta = heuristic[current, nxt] ** self.beta
                weights.append(tau * eta)

            total = sum(weights)
            if total <= 0:
                chosen = rng.choice(candidates)
            else:
                threshold = rng.random() * total
                cumulative = 0.0
                chosen = candidates[-1]
                for node, w in zip(candidates, weights):
                    cumulative += w
                    if cumulative >= threshold:
                        chosen = node
                        break

            route.append(chosen)
            unvisited.remove(chosen)
        return route
