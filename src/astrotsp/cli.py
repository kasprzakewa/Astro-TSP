from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from astrotsp.data.service import InstanceBuilder
from astrotsp.experiments.config_loader import load_benchmark_config
from astrotsp.experiments.runner import BenchmarkRunner
from astrotsp.models.problem import TSPInstance
from astrotsp.reporting.plots import TrajectoryPlotStyle, save_plots, save_trajectory_plots
from astrotsp.reporting.summary import build_summary


def _load_raw_csv_resilient(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        print(f"[warn] ParserError in {path}. Falling back to tolerant parser.")
        bad_line_count = 0

        def _on_bad_lines(bad_line: list[str]) -> None:
            nonlocal bad_line_count
            bad_line_count += 1
            preview = "|".join(bad_line[:8])
            print(f"[warn] Skipping malformed CSV line #{bad_line_count}: {preview}")
            return None

        df = pd.read_csv(path, engine="python", on_bad_lines=_on_bad_lines)
        print(f"[warn] Skipped malformed lines in {path}: {bad_line_count}")
        return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Astro-TSP CLI")
    parser.add_argument(
        "--mode",
        choices=["run-benchmark", "run-benchmark-from-catalog", "generate-plots"],
        default="run-benchmark",
        help="Execution mode.",
    )
    parser.add_argument(
        "--config",
        default="config/benchmark.json",
        help="Path to benchmark config JSON file (run-benchmark mode).",
    )
    parser.add_argument(
        "--raw-csv",
        default="results/benchmark_raw.csv",
        help="Path to benchmark raw CSV (generate-plots mode).",
    )
    parser.add_argument(
        "--summary-csv",
        default="results/benchmark_summary.csv",
        help="Optional path to benchmark summary CSV (generate-plots mode).",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory where plots will be written (generate-plots mode).",
    )
    parser.add_argument(
        "--catalog-csv",
        default="results/instances_catalog.csv",
        help="Path to instances catalog CSV (run-benchmark-from-catalog mode).",
    )
    parser.add_argument(
        "--with-trajectories",
        action="store_true",
        help="Also regenerate trajectory PNGs from raw CSV (generate-plots mode).",
    )
    parser.add_argument(
        "--trajectory-theme",
        choices=["dark", "light"],
        default="dark",
        help="Figure theme for trajectory plots (generate-plots with --with-trajectories).",
    )
    parser.add_argument(
        "--trajectory-hide-title",
        action="store_true",
        help="Omit the trajectory plot title line (generate-plots with --with-trajectories).",
    )
    parser.add_argument(
        "--trajectory-instances",
        default="",
        help="Comma-separated instance_id values to include (empty = all).",
    )
    parser.add_argument(
        "--trajectory-solvers",
        default="",
        help="Comma-separated solver names to include (empty = all).",
    )

    args = parser.parse_args()
    command = args.mode

    if command == "run-benchmark":
        config = load_benchmark_config(args.config)
        runner = BenchmarkRunner()
        raw_df, summary_df = runner.run(config)
        print(f"Benchmark completed: {len(raw_df)} runs")
        print(f"Summary rows: {len(summary_df)}")
        return

    if command == "run-benchmark-from-catalog":
        config = load_benchmark_config(args.config)
        catalog_df = _load_raw_csv_resilient(args.catalog_csv)
        runner = BenchmarkRunner()
        raw_df, summary_df = runner.run_from_catalog(config=config, catalog_df=catalog_df)
        print(f"Benchmark (catalog) completed: {len(raw_df)} runs")
        print(f"Summary rows: {len(summary_df)}")
        return

    if command == "generate-plots":
        raw_df = _load_raw_csv_resilient(args.raw_csv)
        summary_path = Path(args.summary_csv)
        if summary_path.exists():
            try:
                summary_df = pd.read_csv(summary_path)
            except pd.errors.ParserError:
                print(f"[warn] ParserError in {summary_path}. Falling back to tolerant parser.")
                bad_line_count = 0

                def _on_bad_lines_summary(bad_line: list[str]) -> None:
                    nonlocal bad_line_count
                    bad_line_count += 1
                    preview = "|".join(bad_line[:8])
                    print(f"[warn] Skipping malformed summary line #{bad_line_count}: {preview}")
                    return None

                summary_df = pd.read_csv(
                    summary_path,
                    engine="python",
                    on_bad_lines=_on_bad_lines_summary,
                )
                print(f"[warn] Skipped malformed lines in {summary_path}: {bad_line_count}")
        else:
            summary_df = build_summary(raw_df)
        out_dir = Path(args.output_dir)
        save_plots(raw_df=raw_df, summary_df=summary_df, output_dir=out_dir)
        if args.with_trajectories:
            raw_traj = raw_df
            inst_filter = {s.strip() for s in args.trajectory_instances.split(",") if s.strip()}
            solver_filter = {s.strip() for s in args.trajectory_solvers.split(",") if s.strip()}
            if inst_filter:
                raw_traj = raw_traj[raw_traj["instance_id"].isin(inst_filter)]
            if solver_filter:
                raw_traj = raw_traj[raw_traj["solver"].isin(solver_filter)]
            if raw_traj.empty:
                print("[warn] No benchmark rows left for trajectory plots after filters.")
            else:
                instances_by_id: dict[str, TSPInstance] = {}
                builder = InstanceBuilder()
                for instance_id, group in raw_traj.groupby("instance_id"):
                    first = group.iloc[0]
                    epoch = datetime.fromisoformat(str(first["epoch"]))
                    body_ids = [b for b in str(first["selected_body_ids"]).split("|") if b]
                    instances_by_id[str(instance_id)] = builder.from_horizons(
                        body_ids=body_ids,
                        epoch=epoch,
                    )
                traj_style = TrajectoryPlotStyle(
                    theme=args.trajectory_theme,
                    show_title=not args.trajectory_hide_title,
                )
                save_trajectory_plots(
                    raw_df=raw_traj,
                    instances_by_instance_id=instances_by_id,
                    output_dir=out_dir,
                    style=traj_style,
                )
                print(f"Trajectory plots written under: {out_dir} (instances={inst_filter or 'all'}, solvers={solver_filter or 'all'})")
        print(f"Plots generated in: {args.output_dir}")
        print(f"Raw rows used: {len(raw_df)}")
        print(f"Summary rows used: {len(summary_df)}")
        return

    parser.error(f"Unknown mode: {command}")


if __name__ == "__main__":
    main()
