from datetime import datetime

from astrotsp.models.costs import build_euclidean_cost_matrix, build_instance
from astrotsp.models.problem import CelestialBody


def test_euclidean_matrix_is_symmetric_and_zero_diagonal() -> None:
    nodes = [
        CelestialBody("A", "A", 0.0, 0.0, 0.0),
        CelestialBody("B", "B", 3.0, 4.0, 0.0),
        CelestialBody("C", "C", 0.0, 0.0, 12.0),
    ]
    matrix = build_euclidean_cost_matrix(nodes)

    assert matrix.shape == (3, 3)
    assert matrix[0, 0] == 0.0
    assert matrix[1, 1] == 0.0
    assert matrix[2, 2] == 0.0
    assert matrix[0, 1] == matrix[1, 0]
    assert matrix[1, 2] == matrix[2, 1]


def test_instance_validation_passes_for_valid_input() -> None:
    nodes = [
        CelestialBody("A", "A", 0.0, 0.0, 0.0),
        CelestialBody("B", "B", 1.0, 0.0, 0.0),
    ]
    instance = build_instance(nodes=nodes, epoch=datetime(2026, 4, 21))
    assert len(instance.nodes) == 2
