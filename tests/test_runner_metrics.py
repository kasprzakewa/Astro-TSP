import pandas as pd

from astrotsp.experiments.runner import BenchmarkRunner


def test_gap_and_exact_match_metrics() -> None:
    runner = BenchmarkRunner()
    raw = pd.DataFrame(
        [
            {"instance_id": "n6_i01", "solver": "ilp", "status": "optimal", "total_cost": 100.0},
            {"instance_id": "n6_i01", "solver": "branch_and_bound", "status": "optimal", "total_cost": 100.0},
            {"instance_id": "n6_i01", "solver": "aco", "status": "feasible", "total_cost": 110.0},
            {"instance_id": "n6_i01", "solver": "aco", "status": "feasible", "total_cost": 105.0},
            {"instance_id": "n8_i01", "solver": "ilp", "status": "optimal", "total_cost": 200.0},
            {"instance_id": "n8_i01", "solver": "branch_and_bound", "status": "optimal", "total_cost": 201.0},
            {"instance_id": "n8_i01", "solver": "aco", "status": "feasible", "total_cost": 220.0},
        ]
    )
    for col in ["n_nodes", "elapsed_seconds", "memory_usage_mb", "repetition_id", "epoch", "seed", "route"]:
        if col not in raw.columns:
            raw[col] = 0

    out = runner._apply_exact_and_gap_metrics(raw)
    first = out[out["instance_id"] == "n6_i01"]
    assert first["exact_match"].dropna().unique().tolist() == [True]
    assert first[first["solver"] == "aco"]["gap_pct"].round(2).tolist() == [10.0, 5.0]

    second = out[out["instance_id"] == "n8_i01"]
    assert second["consistency_error"].dropna().unique().tolist() == [True]
