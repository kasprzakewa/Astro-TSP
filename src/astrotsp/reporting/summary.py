from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    exact_numeric = raw_df["exact_match"].map(
        lambda v: 1.0 if v is True else (0.0 if v is False else None)
    )
    consistency_numeric = raw_df["consistency_error"].map(
        lambda v: 1.0 if v is True else (0.0 if v is False else None)
    )
    tmp = raw_df.assign(
        exact_match_numeric=exact_numeric,
        consistency_error_numeric=consistency_numeric,
    )

    grouped = (
        tmp.groupby(["solver", "n_nodes"], dropna=False)
        .agg(
            runs=("solver", "count"),
            mean_cost=("total_cost", "mean"),
            max_cost=("total_cost", "max"),
            std_cost=("total_cost", "std"),
            best_cost=("total_cost", "min"),
            worst_cost=("total_cost", "max"),
            mean_time_s=("elapsed_seconds", "mean"),
            max_time_s=("elapsed_seconds", "max"),
            std_time_s=("elapsed_seconds", "std"),
            mean_memory_mb=("memory_usage_mb", "mean"),
            max_memory_mb=("memory_usage_mb", "max"),
            mean_gap_pct=("gap_pct", "mean"),
            max_gap_pct=("gap_pct", "max"),
            exact_match_rate=("exact_match_numeric", "mean"),
            consistency_error_rate=("consistency_error_numeric", "mean"),
        )
        .reset_index()
    )
    return grouped.fillna(0.0)


def save_text_summary(summary_df: pd.DataFrame, raw_df: pd.DataFrame, output_dir: Path) -> Path:
    lines = ["Astro-TSP benchmark summary", ""]
    for n_nodes in sorted(raw_df["n_nodes"].unique()):
        lines.append(f"n_nodes: {int(n_nodes)}")
        chunk = summary_df[summary_df["n_nodes"] == n_nodes].sort_values("mean_cost")
        if chunk.empty:
            lines.append("- no data")
            lines.append("")
            continue
        best_solver = chunk.iloc[0]
        fastest_solver = chunk.sort_values("mean_time_s").iloc[0]
        validation = _validation_status_for_n(raw_df, int(n_nodes))
        lines.append(f"- best mean cost: {best_solver['solver']} ({best_solver['mean_cost']:.6f})")
        lines.append(f"- fastest mean time: {fastest_solver['solver']} ({fastest_solver['mean_time_s']:.6f}s)")
        lines.append(f"- ILP vs B&B validation: {validation}")
        lines.append("")

    path = output_dir / "summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _validation_status_for_n(raw_df: pd.DataFrame, n_nodes: int) -> str:
    chunk = raw_df[raw_df["n_nodes"] == n_nodes]
    if chunk.empty:
        return "no data"

    resolved = chunk.dropna(subset=["exact_match"])
    if resolved.empty:
        return "unresolved (timeout/error)"
    if (resolved["exact_match"] == False).any():
        return "mismatch detected"
    return "all matched"


def _canonical_cycle(route: list[int]) -> str:
    if not route:
        return ""
    min_node = min(route)
    min_index = route.index(min_node)
    rotated = route[min_index:] + route[:min_index]
    reversed_rotated = [rotated[0], *list(reversed(rotated[1:]))]
    fwd = "-".join(str(x) for x in rotated)
    rev = "-".join(str(x) for x in reversed_rotated)
    return min(fwd, rev)
