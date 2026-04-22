from datetime import datetime

from astrotsp.experiments.instance_generator import generate_instance_specs
from astrotsp.models.config import AsteroidPoolConfig, BenchmarkConfig


def test_generator_creates_unique_instances_per_n() -> None:
    cfg = BenchmarkConfig(
        epoch=datetime(2026, 4, 21),
        n_values=[6, 8],
        instances_per_n=10,
        aco_repetitions=5,
        timeout_seconds=60,
        global_seed=42,
        asteroid_pools=[AsteroidPoolConfig(name="pool", start_id=1, end_id=500)],
        anchor_body_ids=["399"],
        enabled_solvers=["branch_and_bound", "ilp", "aco"],
        aco_seeds=[11, 22, 33, 44, 55],
        output_dir="results",
    )

    specs = generate_instance_specs(cfg)
    assert len(specs) == 20

    for n in [6, 8]:
        chunk = [spec for spec in specs if spec.n_nodes == n]
        assert len(chunk) == 10
        bodies = {tuple(spec.body_ids) for spec in chunk}
        assert len(bodies) == 10
