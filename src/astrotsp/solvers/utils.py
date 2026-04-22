from __future__ import annotations

from astrotsp.models.problem import TSPInstance


def route_cost(instance: TSPInstance, route: list[int]) -> float:
    if len(route) != len(instance.nodes):
        raise ValueError("Route size must equal number of nodes.")
    total = 0.0
    for idx in range(len(route)):
        src = route[idx]
        dst = route[(idx + 1) % len(route)]
        total += float(instance.costs[src, dst])
    return total
