#!/usr/bin/env python3
"""Plot PSNR and CLIP Edit sweeps from performance_comparison_ablation.csv."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd


DEFAULT_METHOD = "chord default sd delta1.5"
DEFAULTS = {
    "t_start": 0.90,
    "t_end": 0.30,
    "t_delta": 0.15,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "datasets/PIE-Bench_v1/output/ChordEdit/annotation_images/"
            "performance_comparison_ablation.csv"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def method_value(method: str, prefix: str) -> float | None:
    match = re.search(rf"{re.escape(prefix)}([0-9.]+)$", method)
    return float(match.group(1)) if match else None


def sweep_rows(df: pd.DataFrame, default_row: pd.Series, param: str) -> pd.DataFrame:
    if param == "t_delta":
        rows = df[df["Method"].str.contains(r" delta[0-9.]+$", regex=True)].copy()
        rows[param] = rows["Method"].map(lambda method: method_value(method, "delta") / 10.0)
    elif param == "t_end":
        rows = df[df["Method"].str.contains(r" tend[0-9.]+$", regex=True)].copy()
        rows[param] = rows["Method"].map(lambda method: method_value(method, "tend"))
        rows = pd.concat(
            [rows, default_row.to_frame().T.assign(**{param: DEFAULTS[param]})],
            ignore_index=True,
        )
    elif param == "t_start":
        rows = df[df["Method"].str.contains(r" tstart[0-9.]+$", regex=True)].copy()
        rows[param] = rows["Method"].map(lambda method: method_value(method, "tstart"))
        rows = pd.concat(
            [rows, default_row.to_frame().T.assign(**{param: DEFAULTS[param]})],
            ignore_index=True,
        )
    else:
        raise ValueError(f"Unsupported parameter: {param}")

    return rows.sort_values(param).drop_duplicates(param, keep="last")


def plot_sweep(rows: pd.DataFrame, param: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.7), dpi=220)

    ax.plot(rows[param], rows["PSNR"], marker="o", linewidth=2.1, label="PSNR")
    ax.plot(
        rows[param],
        rows["CLIP Edit"],
        marker="s",
        linewidth=2.1,
        label="CLIP Edit",
    )

    default_x = DEFAULTS[param]
    if rows[param].min() <= default_x <= rows[param].max():
        ax.axvline(default_x, color="#555555", linestyle="--", linewidth=1.1, alpha=0.8)
        ax.annotate(
            "default",
            xy=(default_x, 0.98),
            xycoords=("data", "axes fraction"),
            xytext=(4, -4),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8,
            color="#444444",
        )

    ax.set_xlabel(param)
    ax.set_ylabel("Score")
    ax.set_title(f"{param} ablation")
    ax.set_xticks(rows[param].tolist())
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.75)
    ax.legend(frameon=False, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    default_matches = df[df["Method"] == DEFAULT_METHOD]
    if default_matches.empty:
        raise ValueError(f"Default row not found: {DEFAULT_METHOD}")
    default_row = default_matches.iloc[0]

    for param in ("t_start", "t_end", "t_delta"):
        rows = sweep_rows(df, default_row, param)
        plot_sweep(rows, param, out_dir / f"performance_ablation_{param}.png")


if __name__ == "__main__":
    main()
