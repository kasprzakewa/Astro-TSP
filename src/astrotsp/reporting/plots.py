from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from astrotsp.models.problem import TSPInstance

# Light-theme aggregate plots: high-contrast solver colours (lines = saturated; ACO cloud can be softer).
_BENCH_SURFACE: dict[str, str] = {
    "fig": "#FAFAFA",
    "ax": "#FFFFFF",
    "text": "#262626",
    "muted": "#6E6E6E",
    "grid": "#E2E2E6",
    "spine": "#C4C4CA",
    "zero_line": "#5C5C62",
}

_BENCH_SOLVER_COLOR: dict[str, str] = {
    "aco": "#FF1493",
    "ilp": "#00C853",
    "branch_and_bound": "#0066FF",
}

_BENCH_SOLVER_FALLBACK_COLORS: list[str] = [
    "#FF6D00",
    "#7C4DFF",
    "#00B8D4",
]


def _solver_color(solver: str) -> str:
    if solver in _BENCH_SOLVER_COLOR:
        return _BENCH_SOLVER_COLOR[solver]
    digest = hashlib.md5(solver.encode(), usedforsecurity=False).digest()
    idx = int.from_bytes(digest[:4], "big") % len(_BENCH_SOLVER_FALLBACK_COLORS)
    return _BENCH_SOLVER_FALLBACK_COLORS[idx]


def _solver_legend_label(solver: str) -> str:
    return {
        "aco": "Algorytm mrówkowy",
        "ilp": "Programowanie liniowe",
        "branch_and_bound": "Branch and bound",
    }.get(solver, solver.replace("_", " ").title())


def _style_benchmark_axes(ax: Axes) -> None:
    s = _BENCH_SURFACE
    ax.set_facecolor(s["ax"])
    ax.tick_params(
        axis="both",
        which="both",
        colors=s["text"],
        labelcolor=s["text"],
        length=4,
        width=0.85,
    )
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_color(s["text"])
    for spine in ax.spines.values():
        spine.set_color(s["spine"])
        spine.set_linewidth(0.9)
    ax.grid(True, color=s["grid"], linestyle="-", linewidth=0.75, alpha=1.0)
    ax.set_axisbelow(True)
    ax.xaxis.offsetText.set_color(s["muted"])
    ax.yaxis.offsetText.set_color(s["muted"])


def _style_benchmark_legend(ax: Axes) -> None:
    s = _BENCH_SURFACE
    leg = ax.get_legend()
    if leg is None:
        return
    frame = leg.get_frame()
    frame.set_facecolor("#FFFFFF")
    frame.set_edgecolor(s["spine"])
    frame.set_linewidth(0.85)
    frame.set_alpha(0.96)
    for t in leg.get_texts():
        t.set_color(s["text"])


def _save_benchmark_figure(fig: plt.Figure, out_path: Path) -> None:
    s = _BENCH_SURFACE
    fig.patch.set_facecolor(s["fig"])
    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=150,
        facecolor=s["fig"],
        edgecolor="none",
        bbox_inches="tight",
        pad_inches=0.18,
    )
    plt.close(fig)


@dataclass(frozen=True)
class TrajectoryPlotStyle:
    theme: Literal["dark", "light"] = "dark"
    show_title: bool = True


def _trajectory_palette(style: TrajectoryPlotStyle) -> dict[str, object]:
    if style.theme == "light":
        return {
            "fig_face": "#f7f7f7",
            "ax_face": "#ffffff",
            "route": "#1a1a1a",
            "scatter_edge": "#222222",
            "label": "#111111",
            "tick": "#222222",
            "title": "#111111",
            "spine": "#333333",
            "grid": ("#cccccc", 0.45),
            "legend_face": "#ffffff",
            "legend_edge": "#bbbbbb",
            "legend_text": "#111111",
            "pane_face": (1.0, 1.0, 1.0, 1.0),
            "pane_edge": "#dddddd",
        }
    return {
        "fig_face": "#000000",
        "ax_face": "#000000",
        "route": "#ffffff",
        "scatter_edge": "#ffffff",
        "label": "#ffffff",
        "tick": "#ffffff",
        "title": "#ffffff",
        "spine": "#ffffff",
        "grid": ("#808080", 0.3),
        "legend_face": "#000000",
        "legend_edge": "#ffffff",
        "legend_text": "#ffffff",
        "pane_face": (0.0, 0.0, 0.0, 1.0),
        "pane_edge": "#555555",
    }


def save_plots(raw_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> None:
    _plot_gap_stability(raw_df, output_dir / "gap_stability_vs_nodes.png")
    _plot_metric_vs_nodes(
        summary_df,
        output_dir / "time_vs_nodes_mean.png",
        metric="mean_time_s",
        y_label="Czas [s]",
        title="Czas w funkcji liczby węzłów (średnia, logarytmiczna skala)",
        log_scale=True,
    )
    _plot_metric_vs_nodes(
        summary_df,
        output_dir / "time_vs_nodes_max.png",
        metric="max_time_s",
        y_label="Czas [s]",
        title="Czas w funkcji liczby węzłów (maksimum, logarytmiczna skala)",
        log_scale=True,
    )
    _plot_metric_vs_nodes(
        summary_df,
        output_dir / "memory_vs_nodes_mean.png",
        metric="mean_memory_mb",
        y_label="Pamięć [MB]",
        title="Pamięć w funkcji liczby węzłów (średnia)",
        log_scale=False,
    )
    _plot_metric_vs_nodes(
        summary_df,
        output_dir / "memory_vs_nodes_max.png",
        metric="max_memory_mb",
        y_label="Pamięć [MB]",
        title="Pamięć w funkcji liczby węzłów (maksimum)",
        log_scale=False,
    )
    _plot_exact_validation(raw_df, output_dir / "exact_validation_delta_vs_nodes.png")


def _plot_gap_stability(raw_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    aco = raw_df[(raw_df["solver"] == "aco") & raw_df["gap_pct"].notna()].copy()
    if aco.empty:
        return

    s = _BENCH_SURFACE
    aco_vivid = _BENCH_SOLVER_COLOR["aco"]
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=s["fig"])
    _style_benchmark_axes(ax)

    rng = np.random.default_rng(123)
    aco["x_jitter"] = aco["n_nodes"] + rng.uniform(-0.18, 0.18, size=len(aco))

    ax.scatter(
        aco["x_jitter"],
        aco["gap_pct"],
        alpha=0.26,
        s=46,
        color=aco_vivid,
        linewidths=0,
        edgecolors="none",
        label="Pojedyncze przebiegi ACO",
        rasterized=True,
        zorder=2,
    )

    means = aco.groupby("n_nodes", as_index=False)["gap_pct"].mean()
    ax.scatter(
        means["n_nodes"],
        means["gap_pct"],
        marker="*",
        s=360,
        color="#FFD700",
        edgecolors="#EAAC02",
        linewidths=0.5,
        zorder=6,
        label="Średnie odchylenie od optymalności (ACO)",
    )

    ax.axhline(
        0.0,
        color=s["zero_line"],
        linestyle=(0, (5, 4)),
        linewidth=1.25,
        zorder=1,
        label="Odniesienie dokładne (0 %)",
    )
    xticks = sorted(pd.unique(raw_df["n_nodes"]))
    ax.set_xticks(xticks)
    ax.set_xlabel("Liczba węzłów", color=s["text"], fontsize=10.5)
    ax.set_ylabel("Odchylenie od optymalności [%]", color=s["text"], fontsize=10.5)
    ax.set_title(
        "Optymalność i stabilność algorytmu mrówkowego względem liczby węzłów",
        color=s["text"],
        fontsize=12.5,
        fontweight="600",
        pad=10,
    )
    ax.legend(loc="best", frameon=True, fancybox=False, shadow=False)
    _style_benchmark_legend(ax)
    _save_benchmark_figure(fig, out_path)


def _plot_metric_vs_nodes(
    summary_df: pd.DataFrame,
    out_path: Path,
    metric: str,
    y_label: str,
    title: str,
    log_scale: bool,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s = _BENCH_SURFACE
    fig, ax = plt.subplots(figsize=(8.6, 5.2), facecolor=s["fig"])
    _style_benchmark_axes(ax)

    for solver, chunk in summary_df.groupby("solver"):
        chunk_sorted = chunk.sort_values("n_nodes")
        color = _solver_color(str(solver))
        ax.plot(
            chunk_sorted["n_nodes"],
            chunk_sorted[metric],
            color=color,
            marker="o",
            linewidth=2.35,
            markersize=8.0,
            markerfacecolor=color,
            markeredgecolor="#FFFFFF",
            markeredgewidth=0.75,
            clip_on=False,
            label=_solver_legend_label(str(solver)),
            zorder=3,
        )
    xticks = sorted(pd.unique(summary_df["n_nodes"]))
    ax.set_xticks(xticks)
    ax.set_xlabel("Liczba węzłów", color=s["text"], fontsize=10.5)
    ax.set_ylabel(y_label, color=s["text"], fontsize=10.5)
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title, color=s["text"], fontsize=12.5, fontweight="600", pad=10)
    ax.legend(loc="best", frameon=True, fancybox=False, shadow=False)
    _style_benchmark_legend(ax)
    _save_benchmark_figure(fig, out_path)


def _plot_exact_validation(raw_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bb = raw_df[
        (raw_df["solver"] == "branch_and_bound")
        & (raw_df["status"].isin(["optimal", "feasible"]))
        & raw_df["total_cost"].notna()
    ][["instance_id", "n_nodes", "total_cost"]].rename(columns={"total_cost": "bb_cost"})
    ilp = raw_df[
        (raw_df["solver"] == "ilp")
        & (raw_df["status"].isin(["optimal", "feasible"]))
        & raw_df["total_cost"].notna()
    ][["instance_id", "n_nodes", "total_cost"]].rename(columns={"total_cost": "ilp_cost"})

    merged = bb.merge(ilp, on=["instance_id", "n_nodes"], how="inner")
    if merged.empty:
        return
    merged["delta"] = abs(merged["bb_cost"] - merged["ilp_cost"])

    grouped = merged.groupby("n_nodes", as_index=False)["delta"].mean()
    s = _BENCH_SURFACE
    line_color = _BENCH_SOLVER_COLOR["ilp"]
    fig, ax = plt.subplots(figsize=(8.6, 5.2), facecolor=s["fig"])
    _style_benchmark_axes(ax)

    ax.plot(
        grouped["n_nodes"],
        grouped["delta"],
        color=line_color,
        marker="o",
        linewidth=2.35,
        markersize=7.8,
        markerfacecolor=line_color,
        markeredgecolor="#FFFFFF",
        markeredgewidth=0.75,
        clip_on=False,
        label="Średnia |B&B - ILP|",
        zorder=3,
    )
    ax.axhline(
        0.0,
        color=s["zero_line"],
        linestyle=(0, (5, 4)),
        linewidth=1.25,
        zorder=1,
        label="Oczekiwane zero",
    )
    xticks = sorted(pd.unique(raw_df["n_nodes"]))
    ax.set_xticks(xticks)
    ax.set_xlabel("Liczba węzłów", color=s["text"], fontsize=10.5)
    ax.set_ylabel("Różnica kosztów", color=s["text"], fontsize=10.5)
    ax.set_title(
        f"Walidacja solverów dokładnych: |B&B - ILP|",
        color=s["text"],
        fontsize=12.5,
        fontweight="600",
        pad=10,
    )
    ax.legend(loc="best", frameon=True, fancybox=False, shadow=False)
    _style_benchmark_legend(ax)
    _save_benchmark_figure(fig, out_path)


def save_trajectory_plots(
    raw_df: pd.DataFrame,
    instances_by_instance_id: dict[str, TSPInstance],
    output_dir: Path,
    style: TrajectoryPlotStyle | None = None,
) -> None:
    plot_style = style or TrajectoryPlotStyle()
    for instance_id, chunk in raw_df.groupby("instance_id"):
        if instance_id not in instances_by_instance_id or chunk.empty:
            continue
        scenario_dir = output_dir / instance_id
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # New: best plot per solver inside scenario folder.
        for solver, solver_chunk in chunk.groupby("solver"):
            valid_solver_chunk = solver_chunk[solver_chunk["total_cost"].notna()]
            if valid_solver_chunk.empty:
                continue
            solver_best = valid_solver_chunk.sort_values("total_cost").iloc[0]
            route_tokens = [token for token in str(solver_best["route"]).split("-") if token]
            if not route_tokens:
                continue
            route = [int(token) for token in route_tokens]
            _plot_body_positions_with_route(
                instance=instances_by_instance_id[instance_id],
                route=route,
                solver_name=str(solver),
                out_path=scenario_dir / "3D" / f"{solver}_best.png",
                style=plot_style,
            )
            _plot_body_positions_with_route_2d(
                instance=instances_by_instance_id[instance_id],
                route=route,
                solver_name=str(solver),
                out_path=scenario_dir / "2D" / f"{solver}_best.png",
                style=plot_style,
            )


def _plot_body_positions_with_route(
    instance: TSPInstance,
    route: list[int],
    solver_name: str,
    out_path: Path,
    style: TrajectoryPlotStyle | None = None,
) -> None:
    plot_style = style or TrajectoryPlotStyle()
    pal = _trajectory_palette(plot_style)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab20", len(instance.nodes))

    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor(str(pal["fig_face"]))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(str(pal["ax_face"]))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(pal["pane_face"])  # type: ignore[arg-type]
        axis.pane.set_edgecolor(str(pal["pane_edge"]))
    cycle = route + [route[0]]
    for i in range(len(cycle) - 1):
        src = instance.nodes[cycle[i]]
        dst = instance.nodes[cycle[i + 1]]
        ax.plot(
            [src.x, dst.x],
            [src.y, dst.y],
            [src.z, dst.z],
            linewidth=1.5,
            color=str(pal["route"]),
            zorder=1,
        )

    for idx, body in enumerate(instance.nodes):
        ax.scatter(
            body.x,
            body.y,
            body.z,
            s=85,
            color=colors(idx),
            edgecolors=str(pal["scatter_edge"]),
            linewidths=0.9,
            label=f"{idx}: {body.name}",
            zorder=5,
        )

    ax.set_xlabel("Współrzędna X [AU]")
    ax.set_ylabel("Współrzędna Y [AU]")
    ax.set_zlabel("Współrzędna Z [AU]")
    if plot_style.show_title:
        who = _solver_legend_label(str(solver_name))
        ax.set_title(
            f"Najlepsza trasa (3D): {who} ({instance.epoch.date().isoformat()})"
        )
        ax.title.set_color(str(pal["title"]))
    ax.xaxis.label.set_color(str(pal["label"]))
    ax.yaxis.label.set_color(str(pal["label"]))
    ax.zaxis.label.set_color(str(pal["label"]))
    ax.tick_params(axis="both", which="both", colors=str(pal["tick"]), labelcolor=str(pal["tick"]))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.offsetText.set_color(str(pal["tick"]))
    gcolor, galpha = pal["grid"]  # type: ignore[misc]
    ax.grid(color=str(gcolor), alpha=float(galpha))
    legend = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    legend.get_frame().set_facecolor(str(pal["legend_face"]))
    legend.get_frame().set_edgecolor(str(pal["legend_edge"]))
    for text in legend.get_texts():
        text.set_color(str(pal["legend_text"]))
    fig.subplots_adjust(right=0.72)
    plt.savefig(
        out_path,
        bbox_inches="tight",
        pad_inches=0.2,
        facecolor=str(pal["fig_face"]),
        edgecolor="none",
    )
    plt.close()


def _plot_body_positions_with_route_2d(
    instance: TSPInstance,
    route: list[int],
    solver_name: str,
    out_path: Path,
    style: TrajectoryPlotStyle | None = None,
) -> None:
    plot_style = style or TrajectoryPlotStyle()
    pal = _trajectory_palette(plot_style)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(str(pal["fig_face"]))
    ax.set_facecolor(str(pal["ax_face"]))
    colors = plt.get_cmap("tab20", len(instance.nodes))
    cycle = route + [route[0]]
    for i in range(len(cycle) - 1):
        src = instance.nodes[cycle[i]]
        dst = instance.nodes[cycle[i + 1]]
        ax.plot([src.x, dst.x], [src.y, dst.y], linewidth=1.5, color=str(pal["route"]), zorder=1)

    for idx, body in enumerate(instance.nodes):
        ax.scatter(
            body.x,
            body.y,
            s=85,
            color=colors(idx),
            edgecolors=str(pal["scatter_edge"]),
            linewidths=0.9,
            label=f"{idx}: {body.name}",
            zorder=5,
        )

    ax.set_xlabel("Współrzędna X [AU]")
    ax.set_ylabel("Współrzędna Y [AU]")
    if plot_style.show_title:
        who = _solver_legend_label(str(solver_name))
        ax.set_title(
            f"Najlepsza trasa na płaszczyźnie XY: {who} ({instance.epoch.date().isoformat()})"
        )
        ax.title.set_color(str(pal["title"]))
    ax.xaxis.label.set_color(str(pal["label"]))
    ax.yaxis.label.set_color(str(pal["label"]))
    ax.tick_params(axis="both", which="both", colors=str(pal["tick"]), labelcolor=str(pal["tick"]))
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_color(str(pal["tick"]))
    ax.xaxis.offsetText.set_color(str(pal["tick"]))
    ax.yaxis.offsetText.set_color(str(pal["tick"]))
    for spine in ax.spines.values():
        spine.set_color(str(pal["spine"]))
    gcolor, galpha = pal["grid"]  # type: ignore[misc]
    ax.grid(color=str(gcolor), alpha=float(galpha))
    ax.axis("equal")
    legend = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    legend.get_frame().set_facecolor(str(pal["legend_face"]))
    legend.get_frame().set_edgecolor(str(pal["legend_edge"]))
    for text in legend.get_texts():
        text.set_color(str(pal["legend_text"]))
    fig.subplots_adjust(right=0.72)
    plt.savefig(
        out_path,
        bbox_inches="tight",
        pad_inches=0.2,
        facecolor=str(pal["fig_face"]),
        edgecolor="none",
    )
    plt.close(fig)
