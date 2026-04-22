from datetime import datetime

from astrotsp.models.costs import build_instance
from astrotsp.models.problem import CelestialBody, TSPInstance
from astrotsp.solvers.aco import ACOSolver
from astrotsp.solvers.branch_and_bound import BranchAndBoundSolver
from astrotsp.solvers.ilp import ILPSolver


def _square_instance() -> TSPInstance:
    nodes = [
        CelestialBody("A", "A", 0.0, 0.0, 0.0),
        CelestialBody("B", "B", 1.0, 0.0, 0.0),
        CelestialBody("C", "C", 1.0, 1.0, 0.0),
        CelestialBody("D", "D", 0.0, 1.0, 0.0),
    ]
    return build_instance(nodes=nodes, epoch=datetime(2026, 4, 21))


def test_exact_solvers_agree_on_square() -> None:
    instance = _square_instance()
    bnb = BranchAndBoundSolver().solve(instance)
    ilp = ILPSolver().solve(instance)

    assert bnb.status in {"optimal", "feasible"}
    assert ilp.status in {"optimal", "feasible"}
    assert abs(bnb.total_cost - ilp.total_cost) < 1e-6


def test_aco_returns_full_route() -> None:
    instance = _square_instance()
    result = ACOSolver(iterations=20, ants=10).solve(instance, seed=123)
    assert len(result.route) == len(instance.nodes)
    assert sorted(result.route) == list(range(len(instance.nodes)))
