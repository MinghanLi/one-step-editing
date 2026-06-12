#!/usr/bin/env python3
"""Merge and plot ChordEdit ablation summaries."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd


DEFAULT_METHOD = "chord default sd tstart0.9 tend0.3 delta0.15 cleanup"
DEFAULT_ROOT = "datasets/PIE-Bench_v1/output/ChordEdit/annotation_images/"
DEFAULT_INPUTS = (
    Path(
        DEFAULT_ROOT,
        "performance_comparison_ablation.csv"
    ),
)
DEFAULTS = {
    "t_start": 0.90,
    "t_end": 0.30,
    "t_delta": 0.15,
    "no_cleanup": False,
}
OLD_DELTA_NAME_SCALE = 10.0
METRICS = [
    "Structure Dist.",
    "PSNR",
    "LPIPS",
    "MSE",
    "SSIM",
    "CLIP Src.",
    "CLIP Tgt.",
    "CLIP Edit",
]
LOWER_IS_BETTER = {"Structure Dist.", "LPIPS", "MSE"}
COLORS = {
    "PSNR": "#2f6db5",
    "CLIP Edit": "#c45043",
    "SSIM": "#4d8f61",
    "LPIPS": "#8f63a8",
    "cleanup": "#2f6db5",
    "no cleanup": "#c45043",
}
STATE_DISPLAY_LABELS = {
    "cleanup": "w/ prox",
    "no cleanup": "w/o prox",
}
STATE_LINE_COLORS = {
    "cleanup": "#7f52c8",
    "no cleanup": "#28aebd",
}
PARAM_DISPLAY_LABELS = {
    "t_start": r"$t$",
    "t_delta": r"$\delta$",
    "t_end": r"$t^{**}$",
}
METRIC_DISPLAY_LABELS = {
    "CLIP Edit": "CLIP-Edited",
}
LINEWIDTH = 2.35
MARKERSIZE = 5.6
MARKER_EDGEWIDTH = 0.8
GRID_COLOR = "#dddddd"
MAIN_GRID_AXIS_LABEL_SIZE = 12
MAIN_GRID_TITLE_SIZE = 12
DEFAULT_LINE = {
    "color": "#555555",
    "linestyle": ":",
    "linewidth": 1.45,
    "alpha": 0.9,
    "zorder": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Existing merged CSV to plot. Defaults to --merged-csv when present.",
    )
    parser.add_argument("--merge-csvs", type=Path, nargs="*", default=list(DEFAULT_INPUTS))
    parser.add_argument(
        "--merged-csv",
        type=Path,
        default=Path(
            DEFAULT_ROOT,
            "performance_comparison_ablation_merged.csv"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def method_value(method: str, prefix: str) -> float | None:
    match = re.search(rf"\b{re.escape(prefix)}([0-9.]+)\b", method)
    return float(match.group(1)) if match else None


def format_param(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.0"
    return text


def method_params(method: str) -> dict[str, float | bool]:
    raw_delta = method_value(method, "delta")
    t_start = method_value(method, "tstart")
    t_end = method_value(method, "tend")
    if raw_delta is not None and raw_delta > 1.0:
        raw_delta = raw_delta / OLD_DELTA_NAME_SCALE
    return {
        "t_start": t_start if t_start is not None else DEFAULTS["t_start"],
        "t_end": t_end if t_end is not None else DEFAULTS["t_end"],
        "t_delta": raw_delta if raw_delta is not None else DEFAULTS["t_delta"],
        "cleanup": "no cleanup" not in method,
    }


def normalize_method_name(method: str) -> str:
    params = method_params(method)
    base = method_base(method)
    cleanup = "cleanup" if params["cleanup"] else "no cleanup"
    return (
        f"{base} "
        f"tstart{format_param(params['t_start'])} "
        f"tend{format_param(params['t_end'])} "
        f"delta{format_param(params['t_delta'])} "
        f"{cleanup}"
    )


def normalize_methods(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Method"] = df["Method"].map(normalize_method_name)
    return df


def method_base(method: str) -> str:
    base = re.sub(r"\b(?:tstart|tend|delta)[0-9.]+\b", "", method)
    base = re.sub(r"\bno cleanup\b|\bcleanup\b", "", base)
    return re.sub(r"\s+", " ", base).strip()


def plot_line(
    ax: plt.Axes,
    x: pd.Series,
    y: pd.Series,
    *,
    label: str,
    color: str,
    marker: str,
    linestyle: str = "-",
    markersize: float = MARKERSIZE,
) -> None:
    ax.plot(
        x,
        y,
        marker=marker,
        markersize=markersize,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=MARKER_EDGEWIDTH,
        linewidth=LINEWIDTH,
        linestyle=linestyle,
        solid_capstyle="round",
        label=label,
        color=color,
        zorder=2,
    )


def add_default_line(
    ax: plt.Axes,
    x_value: float,
    annotate: bool = True,
    label: str = "default",
) -> None:
    ax.axvline(x_value, **DEFAULT_LINE)
    if not annotate:
        return
    ax.annotate(
        label,
        xy=(x_value, 0.98),
        xycoords=("data", "axes fraction"),
        xytext=(4, -4),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8,
        color="#444444",
    )


def style_axis(ax: plt.Axes, xticks: list[float] | None = None) -> None:
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_xlabel("")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.82)
    ax.margins(x=0.03, y=0.08)
    ax.tick_params(axis="both", labelsize=8.5, length=3.5, width=0.8, color="#555555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")


def param_title(param: str) -> str:
    return f"{param_display_label(param)} sweep"


def param_display_label(param: str) -> str:
    return PARAM_DISPLAY_LABELS.get(param, param)


def metric_display_label(metric: str) -> str:
    return METRIC_DISPLAY_LABELS.get(metric, metric)


def default_label(param: str) -> str:
    return f"default {param_display_label(param)}={format_param(DEFAULTS[param])}"


def method_short_label(method: str) -> str:
    params = method_params(method)
    base = method_base(method).replace("chord default sd", "").strip()
    parts = []
    if base:
        parts.append(base)
    parts.extend(
        [
            f"{param_display_label('t_start')}={format_param(params['t_start'])}",
            f"{param_display_label('t_end')}={format_param(params['t_end'])}",
            f"{param_display_label('t_delta')}={format_param(params['t_delta'])}",
        ]
    )
    if not params["cleanup"]:
        parts.append(STATE_DISPLAY_LABELS["no cleanup"])
    return ", ".join(parts)


def state_display_label(state: str) -> str:
    return STATE_DISPLAY_LABELS.get(state, state)


def parse_method(method: str) -> dict[str, float | bool | str | None]:
    params = method_params(method)
    return {
        "Method": method,
        "base": method_base(method),
        "t_delta": params["t_delta"],
        "t_start": params["t_start"],
        "t_end": params["t_end"],
        "cleanup": params["cleanup"],
    }


def with_method_params(df: pd.DataFrame) -> pd.DataFrame:
    meta = pd.DataFrame([parse_method(method) for method in df["Method"]])
    return pd.concat([df.reset_index(drop=True), meta.drop(columns=["Method"])], axis=1)


def is_close(series: pd.Series, value: float) -> pd.Series:
    return (series.astype(float) - value).abs() < 1e-9


def merge_csvs(paths: list[Path], out_csv: Path) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["Source"] = path.name
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df = normalize_methods(df)
    df = df.drop_duplicates("Method", keep="last")
    df = df.drop(columns=["Source"])
    df = df.sort_values("Method").reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, float_format="%.4f")
    write_markdown_table(df, out_csv.with_suffix(".md"))
    df.to_latex(out_csv.with_suffix(".tex"), index=False, float_format="%.4f")
    return df


def format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown_table(df: pd.DataFrame, out_path: Path) -> None:
    headers = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(format_cell(row[column]) for column in df.columns) + " |")
    out_path.write_text("\n".join(lines) + "\n")


def read_or_merge(args: argparse.Namespace) -> tuple[pd.DataFrame, Path]:
    if args.csv is not None:
        return normalize_methods(pd.read_csv(args.csv)), args.csv
    if args.merge_csvs:
        return merge_csvs(args.merge_csvs, args.merged_csv), args.merged_csv
    return pd.read_csv(args.merged_csv), args.merged_csv


def sweep_rows(df: pd.DataFrame, default_row: pd.Series, param: str) -> pd.DataFrame:
    rows = with_method_params(df)
    rows = rows[(rows["base"] == "chord default sd") & rows["cleanup"]].copy()
    if param == "t_delta":
        rows = rows[
            is_close(rows["t_start"], DEFAULTS["t_start"])
            & is_close(rows["t_end"], DEFAULTS["t_end"])
        ].copy()
    elif param == "t_end":
        rows = rows[
            is_close(rows["t_start"], DEFAULTS["t_start"])
            & is_close(rows["t_delta"], DEFAULTS["t_delta"])
        ].copy()
    elif param == "t_start":
        rows = rows[
            is_close(rows["t_end"], DEFAULTS["t_end"])
            & is_close(rows["t_delta"], DEFAULTS["t_delta"])
        ].copy()
    else:
        raise ValueError(f"Unsupported parameter: {param}")

    return rows.sort_values(param).drop_duplicates(param, keep="last")


def plot_sweep(rows: pd.DataFrame, param: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.7), dpi=220)

    plot_line(ax, rows[param], rows["PSNR"], label="PSNR", color=COLORS["PSNR"], marker="o")
    plot_line(
        ax,
        rows[param],
        rows["CLIP Edit"],
        label=metric_display_label("CLIP Edit"),
        color=COLORS["CLIP Edit"],
        marker="s",
    )

    default_x = DEFAULTS[param]
    if rows[param].min() <= default_x <= rows[param].max():
        add_default_line(ax, default_x, label=default_label(param))

    ax.set_ylabel("Score")
    ax.set_title(param_title(param))
    style_axis(ax, rows[param].tolist())
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_main_grid(df: pd.DataFrame, default_row: pd.Series, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3), dpi=220, sharey=False)
    titles = {
        "t_delta": r"$t=0.9$, $t^{**}=0.3$",
        "t_start": r"$\delta=0.15$, $t^{**}=0.3$",
        "t_end": r"$\delta=0.15$, $t^{**}=0.3$",
    }
    for ax, param in zip(axes, ("t_delta", "t_start", "t_end")):
        rows = sweep_rows(df, default_row, param)
        plot_line(ax, rows[param], rows["PSNR"], label="PSNR", color=COLORS["PSNR"], marker="o", markersize=10)
        plot_line(
            ax,
            rows[param],
            rows["CLIP Edit"],
            label=metric_display_label("CLIP Edit"),
            color=COLORS["CLIP Edit"],
            marker="s",
            markersize=10,
        )
        default_x = DEFAULTS[param]
        if rows[param].min() <= default_x <= rows[param].max():
            add_default_line(ax, default_x, annotate=False)
        style_axis(ax, rows[param].tolist())
        ax.set_title(titles[param], fontsize=MAIN_GRID_TITLE_SIZE)
        ax.set_xlabel(param_display_label(param), fontsize=MAIN_GRID_AXIS_LABEL_SIZE)
        ax.legend(frameon=False, loc="upper right")
    axes[0].set_ylabel("Score", fontsize=MAIN_GRID_AXIS_LABEL_SIZE)
    # axes[-1].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def interaction_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = with_method_params(df)
    rows = rows[
        (rows["base"] == "chord default sd")
        & is_close(rows["t_delta"], 0.0)
        & is_close(rows["t_end"], DEFAULTS["t_end"])
    ].copy()
    rows["cleanup_state"] = rows["cleanup"].map(lambda value: "cleanup" if value else "no cleanup")
    return rows.sort_values(["cleanup_state", "t_start"])


def plot_interaction(df: pd.DataFrame, out_path: Path) -> None:
    rows = interaction_rows(df)
    if rows.empty:
        return
    default_matches = df[df["Method"] == DEFAULT_METHOD]
    default_row = None if default_matches.empty else default_matches.iloc[0]
    metric_colors = {
        "PSNR": COLORS["CLIP Edit"],
        "CLIP Edit": COLORS["PSNR"],
    }
    state_linestyles = {
        "cleanup": "-",
        "no cleanup": "--",
    }

    xticks = sorted(rows["t_start"].dropna().unique().tolist())
    x_min = min(min(xticks), DEFAULTS["t_start"]) - 0.03
    x_max = max(max(xticks), DEFAULTS["t_start"]) + 0.03
    default_setting_label = (
        rf"$t={format_param(DEFAULTS['t_start'])},\ "
        rf"\delta={format_param(DEFAULTS['t_delta'])},\ "
        rf"t^{{**}}={format_param(DEFAULTS['t_end'])}$"
    )
    interaction_xlabel = rf"{param_display_label('t_start')} ($\delta=0$, $t^{{**}}=0.3$)"

    def add_default_result(
        ax: plt.Axes,
        metric: str,
        label: str | None = None,
        color: str | None = None,
        marker: str = "D",
        size: float = 82,
    ) -> None:
        if default_row is None:
            return
        ax.scatter(
            DEFAULTS["t_start"],
            default_row[metric],
            s=size,
            marker=marker,
            facecolor=color or metric_colors[metric],
            edgecolor="black",
            linewidth=1.25,
            label=label,
            zorder=4,
        )

    def style_interaction_axis(ax: plt.Axes, default_line_label: str | None = None) -> None:
        add_default_line(ax, DEFAULTS["t_start"], label=default_line_label or default_label("t_start"))
        ax.set_xlim(x_min, x_max)
        style_axis(ax, xticks)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.7), dpi=220)
    for ax, metric in zip(axes, ("PSNR", "CLIP Edit")):
        for state in ("cleanup", "no cleanup"):
            state_rows = rows[rows["cleanup_state"] == state]
            if state_rows.empty:
                continue
            plot_line(
                ax,
                state_rows["t_start"],
                state_rows[metric],
                label=state_display_label(state),
                color=metric_colors[metric],
                marker="o" if state == "cleanup" else "s",
                linestyle=state_linestyles[state],
            )
        add_default_result(ax, metric, "default setting" if metric == "CLIP Edit" else None)
        ax.set_title(metric_display_label(metric))
        style_interaction_axis(ax)

    axes[0].set_ylabel("Score")
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle(r"$t$ interaction at $\delta=0.00$", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    by_state_path = out_path.with_name(f"{out_path.stem}_by_state{out_path.suffix}")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), dpi=220)
    y_values = rows[["PSNR", "CLIP Edit"]].to_numpy().ravel().tolist()
    if default_row is not None:
        y_values.extend([default_row["PSNR"], default_row["CLIP Edit"]])
    y_min = min(y_values)
    y_max = max(y_values)
    y_pad = max((y_max - y_min) * 0.08, 0.05)
    metric_linestyles = {
        "PSNR": "-",
        "CLIP Edit": "--",
    }
    for ax, state in zip(axes, ("cleanup", "no cleanup")):
        state_rows = rows[rows["cleanup_state"] == state]
        if state_rows.empty:
            continue
        state_color = STATE_LINE_COLORS[state]
        ax.plot(
            state_rows["t_start"],
            state_rows["PSNR"],
            marker="o",
            markersize=10,
            markerfacecolor=state_color,
            markeredgecolor="white",
            markeredgewidth=MARKER_EDGEWIDTH,
            linewidth=2.8,
            linestyle=metric_linestyles["PSNR"],
            solid_capstyle="round",
            label="PSNR",
            color=state_color,
            zorder=2,
        )
        ax.plot(
            state_rows["t_start"],
            state_rows["CLIP Edit"],
            marker="s",
            markersize=10,
            markerfacecolor=state_color,
            markeredgecolor="white",
            markeredgewidth=MARKER_EDGEWIDTH,
            linewidth=2.8,
            linestyle=metric_linestyles["CLIP Edit"],
            solid_capstyle="round",
            label=metric_display_label("CLIP Edit"),
            color=state_color,
            zorder=2,
        )
        add_default_result(
            ax,
            "PSNR",
            "PSNR default" if state == "no cleanup" else None,
            color=state_color,
            marker="o",
            size=112,
        )
        add_default_result(
            ax,
            "CLIP Edit",
            f"{metric_display_label('CLIP Edit')} default" if state == "no cleanup" else None,
            color=state_color,
            marker="s",
            size=112,
        )
        ax.set_title(state_display_label(state))
        style_interaction_axis(ax, default_setting_label)
        ax.set_xlabel(interaction_xlabel)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    axes[0].set_ylabel("Score")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=9.2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.subplots_adjust(wspace=0.34)
    fig.savefig(by_state_path, bbox_inches="tight")
    plt.close(fig)

    cleanup_path = out_path.with_name(f"{out_path.stem}_w_prox{out_path.suffix}")
    cleanup_rows = rows[rows["cleanup_state"] == "cleanup"]
    if not cleanup_rows.empty:
        fig, ax = plt.subplots(figsize=(5.2, 3.7), dpi=220)
        plot_line(
            ax,
            cleanup_rows["t_start"],
            cleanup_rows["PSNR"],
            label="PSNR",
            color=COLORS["PSNR"],
            marker="o",
            markersize=10,
        )
        plot_line(
            ax,
            cleanup_rows["t_start"],
            cleanup_rows["CLIP Edit"],
            label=metric_display_label("CLIP Edit"),
            color=COLORS["CLIP Edit"],
            marker="s",
            markersize=10,
        )
        add_default_result(
            ax,
            "PSNR",
            "Default PSNR",
            color=COLORS["PSNR"],
            marker="o",
            size=112,
        )
        add_default_result(
            ax,
            "CLIP Edit",
            f"Default {metric_display_label('CLIP Edit')}",
            color=COLORS["CLIP Edit"],
            marker="s",
            size=112,
        )
        ax.set_title(r"Naive ChordEdit ($\delta=0$) with different timestep ($t$)")
        add_default_line(ax, DEFAULTS["t_start"], annotate=False)
        ax.set_xlim(x_min, x_max)
        style_axis(ax, xticks)
        if default_row is not None:
            default_label_box = {"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.78}
            default_label_common = {
                "textcoords": "offset points",
                "fontsize": 8.5,
                "bbox": default_label_box,
            }
            ax.annotate(
                "Default PSNR",
                (DEFAULTS["t_start"], default_row["PSNR"]),
                xytext=(18, 10),
                color=COLORS["PSNR"],
                arrowprops={
                    "arrowstyle": "->",
                    "color": COLORS["PSNR"],
                    "linewidth": 1.2,
                    "shrinkA": 2,
                    "shrinkB": 6,
                },
                **default_label_common,
            )
            ax.annotate(
                f"Default {metric_display_label('CLIP Edit')}",
                (DEFAULTS["t_start"], default_row["CLIP Edit"]),
                xytext=(18, -34),
                color=COLORS["CLIP Edit"],
                arrowprops={
                    "arrowstyle": "->",
                    "color": COLORS["CLIP Edit"],
                    "linewidth": 1.2,
                    "shrinkA": 2,
                    "shrinkB": 6,
                },
                **default_label_common,
            )
        ax.set_xlabel(interaction_xlabel)
        ax.set_ylabel("Score")
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.legend(frameon=False, loc="upper right", fontsize=7.8)
        fig.tight_layout()
        fig.savefig(cleanup_path, bbox_inches="tight")
        plt.close(fig)


def plot_tradeoff(df: pd.DataFrame, out_path: Path) -> None:
    rows = with_method_params(df)
    fig, ax = plt.subplots(figsize=(6.2, 4.4), dpi=220)
    legend_seen = set()

    for _, row in rows.iterrows():
        is_no_cleanup = not bool(row["cleanup"])
        marker = "X" if is_no_cleanup else "o"
        state = "no cleanup" if is_no_cleanup else "cleanup"
        color = COLORS[state]
        ax.scatter(
            row["PSNR"],
            row["CLIP Edit"],
            s=64,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=MARKER_EDGEWIDTH,
            alpha=0.9,
            label=None if state in legend_seen else state_display_label(state),
            zorder=2,
        )
        legend_seen.add(state)
        label = method_short_label(row["Method"])
        ax.annotate(label, (row["PSNR"], row["CLIP Edit"]), xytext=(4, 3), textcoords="offset points", fontsize=6.8)

    ax.set_ylabel(metric_display_label("CLIP Edit"))
    ax.set_title("Preservation/editing trade-off")
    style_axis(ax)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def paired_cleanup_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = with_method_params(df)
    no_cleanup_rows = rows[~rows["cleanup"]].copy()
    pairs = []
    for _, no_cleanup in no_cleanup_rows.iterrows():
        cleanup_matches = rows[
            (rows["base"] == no_cleanup["base"])
            & is_close(rows["t_start"], no_cleanup["t_start"])
            & is_close(rows["t_end"], no_cleanup["t_end"])
            & is_close(rows["t_delta"], no_cleanup["t_delta"])
            & rows["cleanup"]
        ]
        if cleanup_matches.empty:
            continue
        cleanup = cleanup_matches.iloc[0]
        cleanup_method = cleanup["Method"]
        is_default = (
            is_close(pd.Series([cleanup["t_start"]]), DEFAULTS["t_start"]).iloc[0]
            and is_close(pd.Series([cleanup["t_end"]]), DEFAULTS["t_end"]).iloc[0]
            and is_close(pd.Series([cleanup["t_delta"]]), DEFAULTS["t_delta"]).iloc[0]
        )
        label = "default" if is_default else f"{param_display_label('t_start')}={cleanup['t_start']:.1f}"
        pairs.append(
            {
                "label": label,
                "cleanup_method": cleanup_method,
                "no_cleanup_method": no_cleanup["Method"],
                "cleanup_psnr": cleanup["PSNR"],
                "cleanup_clip": cleanup["CLIP Edit"],
                "no_cleanup_psnr": no_cleanup["PSNR"],
                "no_cleanup_clip": no_cleanup["CLIP Edit"],
                "clip_gain": cleanup["CLIP Edit"] - no_cleanup["CLIP Edit"],
            }
        )
    return pd.DataFrame(pairs)


def plot_cleanup_pair_tradeoff(df: pd.DataFrame, out_path: Path) -> None:
    pairs = paired_cleanup_rows(df)
    if pairs.empty:
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.7), dpi=220)
    label_offsets = {
        f"{param_display_label('t_start')}=0.6": (48, -10),
        f"{param_display_label('t_start')}=0.7": (-36, -30),
        f"{param_display_label('t_start')}=0.8": (-16, 10),
        f"{param_display_label('t_start')}=1.0": (8, 8),
        "default": (-58, 14),
    }
    for _, pair in pairs.iterrows():
        ax.scatter(
            pair["no_cleanup_psnr"],
            pair["no_cleanup_clip"],
            s=68,
            marker="s",
            color=COLORS["no cleanup"],
            edgecolor="white",
            linewidth=MARKER_EDGEWIDTH,
            zorder=3,
        )
        ax.scatter(
            pair["cleanup_psnr"],
            pair["cleanup_clip"],
            s=68,
            marker="o",
            color=COLORS["cleanup"],
            edgecolor="white",
            linewidth=MARKER_EDGEWIDTH,
            zorder=3,
        )
        ax.annotate(
            "",
            xy=(pair["cleanup_psnr"], pair["cleanup_clip"]),
            xytext=(pair["no_cleanup_psnr"], pair["no_cleanup_clip"]),
            arrowprops={
                "arrowstyle": "->",
                "color": "#3b6f3e",
                "linewidth": 1.6,
                "shrinkA": 6,
                "shrinkB": 6,
                "alpha": 0.9,
            },
            zorder=2,
        )
        mid_x = (pair["cleanup_psnr"] + pair["no_cleanup_psnr"]) / 2.0
        mid_y = (pair["cleanup_clip"] + pair["no_cleanup_clip"]) / 2.0
        ax.annotate(
            f"{pair['label']}\n+{pair['clip_gain']:.2f} CLIP",
            (mid_x, mid_y),
            xytext=label_offsets.get(pair["label"], (4, 4)),
            textcoords="offset points",
            fontsize=6.9,
            color="#333333",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.78},
        )

    ax.scatter(
        [],
        [],
        s=68,
        marker="s",
        color=COLORS["no cleanup"],
        edgecolor="white",
        label=state_display_label("no cleanup"),
    )
    ax.scatter(
        [],
        [],
        s=68,
        marker="o",
        color=COLORS["cleanup"],
        edgecolor="white",
        label=state_display_label("cleanup"),
    )
    ax.set_xlabel("PSNR")
    ax.set_ylabel(metric_display_label("CLIP Edit"))
    ax.set_title("Proximity paired trade-off")
    style_axis(ax)
    ax.set_xlabel("PSNR")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_summary(df: pd.DataFrame, out_path: Path) -> None:
    def get_row(method: str) -> pd.Series:
        method = normalize_method_name(method)
        matches = df[df["Method"] == method]
        if matches.empty:
            raise ValueError(f"Summary method not found: {method}")
        return matches.iloc[0]

    def delta(method: str, metric: str, base: str = DEFAULT_METHOD) -> float:
        return get_row(method)[metric] - get_row(base)[metric]

    lines = [
        "# ChordEdit ablation notes",
        "",
        f"Rows: {len(df)}",
        "",
        "## Best values",
        "",
    ]
    for metric in METRICS:
        ascending = metric in LOWER_IS_BETTER
        row = df.sort_values(metric, ascending=ascending).iloc[0]
        direction = "min" if ascending else "max"
        lines.append(f"- {metric_display_label(metric)} ({direction}): {row['Method']} = {row[metric]:.4f}")

    default = df[df["Method"] == DEFAULT_METHOD].iloc[0]
    lines.extend(["", f"## Default reference", "", f"`{DEFAULT_METHOD}`"])
    for metric in ("Structure Dist.", "PSNR", "LPIPS", "SSIM", "CLIP Edit"):
        lines.append(f"- {metric_display_label(metric)}: {default[metric]:.4f}")

    lines.extend(
        [
            "",
            "## Reading the ablation",
            "",
            "- Lower `t_start` preserves the unedited region better but weakens the edit: "
            f"`tstart0.6` changes PSNR by {delta('chord default sd tstart0.6', 'PSNR'):+.4f} "
            f"and {metric_display_label('CLIP Edit')} by {delta('chord default sd tstart0.6', 'CLIP Edit'):+.4f} versus default.",
            "- Higher `t_end` pushes edit strength at the cost of preservation: "
            f"`tend0.5` changes {metric_display_label('CLIP Edit')} by {delta('chord default sd tend0.5', 'CLIP Edit'):+.4f} "
            f"and PSNR by {delta('chord default sd tend0.5', 'PSNR'):+.4f} versus default.",
            "- `delta0.25` is the best single-parameter preservation point among the delta sweep, "
            f"with PSNR {get_row('chord default sd delta0.25')['PSNR']:.4f} and "
            f"{metric_display_label('CLIP Edit')} {get_row('chord default sd delta0.25')['CLIP Edit']:.4f}.",
            "- The `delta0.0, t_start=0.6, no cleanup` run is the preservation extreme, "
            f"but it gives up {abs(delta('chord default sd delta0.0 tstart0.6 no cleanup', 'CLIP Edit')):.4f} "
            f"{metric_display_label('CLIP Edit')} points versus default.",
        ]
    )

    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    df, csv_path = read_or_merge(args)
    out_dir = args.out_dir or csv_path.parent
    out_dir = out_dir.parent / "evaluation_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    default_matches = df[df["Method"] == DEFAULT_METHOD]
    if default_matches.empty:
        raise ValueError(f"Default row not found: {DEFAULT_METHOD}")
    default_row = default_matches.iloc[0]

    for param in ("t_start", "t_end", "t_delta"):
        rows = sweep_rows(df, default_row, param)
        plot_sweep(rows, param, out_dir / f"performance_ablation_{param}.png")
    plot_main_grid(df, default_row, out_dir / "performance_ablation_main_grid.png")
    plot_interaction(df, out_dir / "performance_ablation_delta0_tstart_cleanup.png")
    plot_tradeoff(df, out_dir / "performance_ablation_tradeoff.png")
    plot_cleanup_pair_tradeoff(df, out_dir / "performance_ablation_tradeoff_cleanup_pairs.png")
    write_summary(df, out_dir / "performance_comparison_ablation_merged_summary.md")


if __name__ == "__main__":
    main()
