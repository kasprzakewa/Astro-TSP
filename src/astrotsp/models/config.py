from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class AsteroidPoolConfig:
    name: str
    start_id: int
    end_id: int


@dataclass(frozen=True)
class BenchmarkConfig:
    epoch: datetime
    n_values: list[int]
    instances_per_n: int
    aco_repetitions: int
    timeout_seconds: int
    global_seed: int
    asteroid_pools: list[AsteroidPoolConfig]
    anchor_body_ids: list[str]
    enabled_solvers: list[str]
    aco_seeds: list[int]
    output_dir: str = "results"
    trajectory_plot_theme: str = "dark"
    trajectory_plot_show_title: bool = True
