from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import asdict, dataclass
from pathlib import Path
import time
import tracemalloc
from typing import Literal

import numpy as np
import pandas as pd

from astrotsp.data.service import InstanceBuilder
from astrotsp.experiments.instance_generator import InstanceSpec, generate_instance_specs
from astrotsp.models.config import BenchmarkConfig
from astrotsp.models.problem import TSPInstance
from astrotsp.reporting.plots import TrajectoryPlotStyle, save_plots, save_trajectory_plots
from astrotsp.reporting.summary import build_summary, save_text_summary
from astrotsp.solvers.registry import build_solver


@dataclass
class RunRecord:
    n_nodes: int
    instance_id: str
    repetition_id: int
    solver: str
    status: str
    total_cost: float | None
    elapsed_seconds: float
    memory_usage_mb: float | None
    gap_pct: float | None
    exact_match: bool | None
    consistency_error: bool | None
    epoch: str
    seed: int | None
    selected_body_ids: str
    route: str


RUN_COLUMNS = list(RunRecord.__annotations__.keys())


def _trajectory_style_from_config(config: BenchmarkConfig) -> TrajectoryPlotStyle:
    theme: Literal["dark", "light"] = "light" if config.trajectory_plot_theme == "light" else "dark"
    return TrajectoryPlotStyle(theme=theme, show_title=config.trajectory_plot_show_title)


def _solve_worker(solver_name: str, instance: TSPInstance, seed: int | None) -> dict[str, object]:
    solver = build_solver(solver_name)
    tracemalloc.start()
    result = solver.solve(instance=instance, seed=seed)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "status": str(result.status).lower(),
        "total_cost": float(result.total_cost),
        "route": "-".join(str(idx) for idx in result.route),
        "memory_usage_mb": float(peak_bytes / (1024 * 1024)),
    }


class BenchmarkRunner:
    def __init__(self, instance_builder: InstanceBuilder | None = None) -> None:
        self.instance_builder = instance_builder or InstanceBuilder()

    def run(self, config: BenchmarkConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
        specs = generate_instance_specs(config)
        return self._run_specs(config=config, specs=specs)

    def run_from_catalog(
        self,
        config: BenchmarkConfig,
        catalog_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        required_cols = {"instance_id", "n_nodes", "epoch", "selected_body_ids"}
        missing = required_cols.difference(catalog_df.columns)
        if missing:
            raise ValueError(f"Catalog is missing columns: {sorted(missing)}")

        specs: list[InstanceSpec] = []
        for row in catalog_df.itertuples(index=False):
            specs.append(
                InstanceSpec(
                    n_nodes=int(getattr(row, "n_nodes")),
                    instance_id=str(getattr(row, "instance_id")),
                    body_ids=str(getattr(row, "selected_body_ids")).split("|"),
                )
            )
        return self._run_specs(config=config, specs=specs)

    def _run_specs(
        self,
        config: BenchmarkConfig,
        specs: list[InstanceSpec],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows: list[RunRecord] = []
        all_instances_by_id: dict[str, TSPInstance] = {}
        instance_body_ids: dict[str, list[str]] = {}
        specs_by_n: dict[int, list] = {}
        for spec in specs:
            specs_by_n.setdefault(spec.n_nodes, []).append(spec)

        for n_nodes in sorted(specs_by_n):
            for spec in specs_by_n[n_nodes]:
                instance = self.instance_builder.from_horizons(
                    body_ids=spec.body_ids,
                    epoch=config.epoch,
                )
                all_instances_by_id[spec.instance_id] = instance
                instance_body_ids[spec.instance_id] = spec.body_ids
                for solver_name in config.enabled_solvers:
                    repetitions = config.aco_repetitions if solver_name == "aco" else 1
                    for repetition in range(1, repetitions + 1):
                        seed = (
                            config.aco_seeds[(repetition - 1) % len(config.aco_seeds)]
                            if solver_name == "aco"
                            else None
                        )
                        if solver_name == "branch_and_bound" and spec.n_nodes > 12:
                            continue
                        print(f"Running {solver_name} for {spec.instance_id} (n={spec.n_nodes}) rep={repetition}")
                        exec_result = self._run_with_timeout(
                            solver_name=solver_name,
                            instance=instance,
                            seed=seed,
                            timeout_seconds=config.timeout_seconds,
                        )
                        rows.append(
                            RunRecord(
                                n_nodes=spec.n_nodes,
                                instance_id=spec.instance_id,
                                repetition_id=repetition,
                                solver=solver_name,
                                status=str(exec_result["status"]),
                                total_cost=exec_result["total_cost"],
                                elapsed_seconds=float(exec_result["elapsed_seconds"]),
                                memory_usage_mb=exec_result["memory_usage_mb"],
                                gap_pct=None,
                                exact_match=None,
                                consistency_error=None,
                                epoch=config.epoch.isoformat(),
                                seed=seed,
                                selected_body_ids="|".join(spec.body_ids),
                                route=str(exec_result["route"]),
                            )
                        )

            # Checkpoint after finishing each n bucket.
            raw_df_n = self._rows_to_dataframe(rows)
            raw_df_n = self._apply_exact_and_gap_metrics(raw_df_n)
            summary_df_n = build_summary(raw_df_n)
            self._save_outputs(
                config.output_dir,
                raw_df_n,
                summary_df_n,
                all_instances_by_id,
                instance_body_ids,
                config.epoch.isoformat(),
                trajectory_style=_trajectory_style_from_config(config),
            )
            print(f"Checkpoint saved after n={n_nodes}")

        raw_df = self._rows_to_dataframe(rows)
        raw_df = self._apply_exact_and_gap_metrics(raw_df)
        summary_df = build_summary(raw_df)
        self._save_outputs(
            config.output_dir,
            raw_df,
            summary_df,
            all_instances_by_id,
            instance_body_ids,
            config.epoch.isoformat(),
            trajectory_style=_trajectory_style_from_config(config),
        )
        return raw_df, summary_df

    def _rows_to_dataframe(self, rows: list[RunRecord]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=RUN_COLUMNS)
        return pd.DataFrame(asdict(r) for r in rows)

    def _run_with_timeout(
        self,
        solver_name: str,
        instance: TSPInstance,
        seed: int | None,
        timeout_seconds: int,
    ) -> dict[str, object]:
        started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_solve_worker, solver_name, instance, seed)
            try:
                result = future.result(timeout=timeout_seconds)
                elapsed = time.perf_counter() - started
                status = str(result.get("status", "error"))
                if status not in {"optimal", "feasible", "timeout", "error"}:
                    status = "feasible"
                return {
                    "status": status,
                    "total_cost": result.get("total_cost"),
                    "elapsed_seconds": elapsed,
                    "memory_usage_mb": result.get("memory_usage_mb"),
                    "route": result.get("route", ""),
                }
            except FuturesTimeoutError:
                future.cancel()
                return {
                    "status": "timeout",
                    "total_cost": None,
                    "elapsed_seconds": float(timeout_seconds),
                    "memory_usage_mb": None,
                    "route": "",
                }
            except Exception:
                return {
                    "status": "error",
                    "total_cost": None,
                    "elapsed_seconds": time.perf_counter() - started,
                    "memory_usage_mb": None,
                    "route": "",
                }

    def _apply_exact_and_gap_metrics(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if raw_df.empty:
            return raw_df

        raw_df = raw_df.copy()
        raw_df["gap_pct"] = np.nan
        raw_df["exact_match"] = pd.NA
        raw_df["consistency_error"] = pd.NA

        for instance_id, chunk in raw_df.groupby("instance_id"):
            ilp_valid = chunk[(chunk["solver"] == "ilp") & (chunk["status"].isin(["optimal", "feasible"])) & chunk["total_cost"].notna()]
            bb_valid = chunk[(chunk["solver"] == "branch_and_bound") & (chunk["status"].isin(["optimal", "feasible"])) & chunk["total_cost"].notna()]

            ilp_cost = float(ilp_valid.iloc[0]["total_cost"]) if not ilp_valid.empty else None
            bb_cost = float(bb_valid.iloc[0]["total_cost"]) if not bb_valid.empty else None

            if ilp_cost is not None and bb_cost is not None:
                match = abs(ilp_cost - bb_cost) <= 1e-6
                raw_df.loc[raw_df["instance_id"] == instance_id, "exact_match"] = match
                raw_df.loc[raw_df["instance_id"] == instance_id, "consistency_error"] = (not match)

            if ilp_cost is not None and ilp_cost != 0:
                mask_aco = (
                    (raw_df["instance_id"] == instance_id)
                    & (raw_df["solver"] == "aco")
                    & raw_df["total_cost"].notna()
                )
                raw_df.loc[mask_aco, "gap_pct"] = (
                    100.0 * (raw_df.loc[mask_aco, "total_cost"] - ilp_cost) / ilp_cost
                )

        return raw_df

    def _save_outputs(
        self,
        output_dir: str,
        raw_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        instances_by_id: dict[str, TSPInstance],
        instance_body_ids: dict[str, list[str]],
        epoch_iso: str,
        trajectory_style: TrajectoryPlotStyle,
    ) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        raw_path = out / "benchmark_raw.csv"
        summary_path = out / "benchmark_summary.csv"

        raw_df.to_csv(raw_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        self._save_instance_catalog(
            output_dir=out,
            instance_body_ids=instance_body_ids,
            epoch_iso=epoch_iso,
        )
        save_plots(raw_df=raw_df, summary_df=summary_df, output_dir=out)
        save_trajectory_plots(
            raw_df=raw_df,
            instances_by_instance_id=instances_by_id,
            output_dir=out,
            style=trajectory_style,
        )
        save_text_summary(summary_df=summary_df, raw_df=raw_df, output_dir=out)

    def _save_instance_catalog(
        self,
        output_dir: Path,
        instance_body_ids: dict[str, list[str]],
        epoch_iso: str,
    ) -> None:
        rows = []
        for instance_id, body_ids in sorted(instance_body_ids.items()):
            rows.append(
                {
                    "instance_id": instance_id,
                    "n_nodes": len(body_ids),
                    "epoch": epoch_iso,
                    "selected_body_ids": "|".join(body_ids),
                }
            )
        pd.DataFrame(rows).to_csv(output_dir / "instances_catalog.csv", index=False)
