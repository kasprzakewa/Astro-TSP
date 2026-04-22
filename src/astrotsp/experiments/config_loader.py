from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from astrotsp.models.config import AsteroidPoolConfig, BenchmarkConfig


def load_benchmark_config(path: str) -> BenchmarkConfig:
    config_path = Path(path)
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    pools = [
        AsteroidPoolConfig(
            name=p["name"],
            start_id=int(p["start_id"]),
            end_id=int(p["end_id"]),
        )
        for p in raw["asteroid_pools"]
    ]

    theme = str(raw.get("trajectory_plot_theme", "dark"))
    if theme not in ("dark", "light"):
        theme = "dark"

    return BenchmarkConfig(
        epoch=datetime.fromisoformat(raw["epoch"]),
        n_values=[int(n) for n in raw["n_values"]],
        instances_per_n=int(raw.get("instances_per_n", 10)),
        aco_repetitions=int(raw.get("aco_repetitions", 5)),
        timeout_seconds=int(raw.get("timeout_seconds", 60)),
        global_seed=int(raw.get("global_seed", 42)),
        asteroid_pools=pools,
        anchor_body_ids=[str(body_id) for body_id in raw.get("anchor_body_ids", ["399"])],
        enabled_solvers=raw["enabled_solvers"],
        aco_seeds=[int(seed) for seed in raw.get("aco_seeds", [11, 22, 33, 44, 55])],
        output_dir=raw.get("output_dir", "results"),
        trajectory_plot_theme=theme,
        trajectory_plot_show_title=bool(raw.get("trajectory_plot_show_title", True)),
    )
