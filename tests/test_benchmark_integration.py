from datetime import datetime
from pathlib import Path

from astrotsp.experiments.runner import BenchmarkRunner
from astrotsp.models.config import AsteroidPoolConfig, BenchmarkConfig
from astrotsp.models.costs import build_instance
from astrotsp.models.problem import CelestialBody, TSPInstance


class FakeInstanceBuilder:
    def from_horizons(self, body_ids: list[str], epoch: datetime) -> TSPInstance:
        nodes = [
            CelestialBody(name=bid, horizons_id=bid, x=float(i), y=float(i % 2), z=0.0)
            for i, bid in enumerate(body_ids)
        ]
        return build_instance(nodes=nodes, epoch=epoch)


def test_full_pipeline_creates_csv_and_summary(tmp_path: Path) -> None:
    cfg = BenchmarkConfig(
        epoch=datetime(2026, 4, 21),
        n_values=[4],
        instances_per_n=2,
        aco_repetitions=2,
        timeout_seconds=5,
        global_seed=7,
        asteroid_pools=[AsteroidPoolConfig(name="tiny", start_id=1, end_id=50)],
        anchor_body_ids=["399"],
        enabled_solvers=["branch_and_bound", "ilp", "aco"],
        aco_seeds=[1, 2],
        output_dir=str(tmp_path),
    )
    runner = BenchmarkRunner(instance_builder=FakeInstanceBuilder())
    raw_df, summary_df = runner.run(cfg)

    assert not raw_df.empty
    assert not summary_df.empty
    assert "instance_id" in raw_df.columns
    assert "repetition_id" in raw_df.columns
    assert "selected_body_ids" in raw_df.columns
    assert "gap_pct" in raw_df.columns
    assert "exact_match" in raw_df.columns
    assert (tmp_path / "benchmark_raw.csv").exists()
    assert (tmp_path / "benchmark_summary.csv").exists()
    assert (tmp_path / "instances_catalog.csv").exists()
    assert (tmp_path / "summary.txt").exists()
    assert (tmp_path / "gap_stability_vs_nodes.png").exists()
    assert (tmp_path / "time_vs_nodes_mean.png").exists()
    assert (tmp_path / "time_vs_nodes_max.png").exists()
    assert (tmp_path / "memory_vs_nodes_mean.png").exists()
    assert (tmp_path / "memory_vs_nodes_max.png").exists()
    assert (tmp_path / "exact_validation_delta_vs_nodes.png").exists()
