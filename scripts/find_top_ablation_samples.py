#!/usr/bin/env python3
"""Find top per-sample PSNR and CLIP-Edited cases for selected ablations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_ROOT = Path("datasets/PIE-Bench_v1/output/ChordEdit/annotation_images")
DEFAULT_EVALUATION_RESULTS_ROOT = DEFAULT_ROOT.parent / "evaluation_results"
DEFAULT_EVALUATION_CSVS = (
    DEFAULT_ROOT / "evaluation_result_ablation.csv",
)
DEFAULT_SAMPLE_META = Path("scripts/anchor_samples.csv")
DEFAULT_OUT_CSV = DEFAULT_EVALUATION_RESULTS_ROOT / "top5_selected_ablation_samples.csv"
DEFAULT_OUT_MD = DEFAULT_EVALUATION_RESULTS_ROOT / "top5_selected_ablation_samples.md"

METRIC_COLUMNS = {
    "PSNR": "psnr_unedit_part",
    "CLIP-Edited": "clip_similarity_target_image_edit_part",
}
EXTRA_COLUMNS = {
    "Structure Dist.": "structure_distance",
    "LPIPS": "lpips_unedit_part",
    "MSE": "mse_unedit_part",
    "SSIM": "ssim_unedit_part",
    "CLIP Src.": "clip_similarity_source_image",
    "CLIP Tgt.": "clip_similarity_target_image",
}


@dataclass(frozen=True)
class Experiment:
    label: str
    prefix: str


EXPERIMENTS = (
    Experiment(
        label="tstart0.9 tend0.3 delta0.15 w/ cleanup",
        prefix="0_chord_default_sd_delta0.15",
    ),
    Experiment(
        label="tstart0.9 tend0.3 delta0.15 w/o cleanup",
        prefix="0_chord_default_sd_delta0.15_no_cleanup",
    ),
    Experiment(
        label="tstart0.9 tend0.3 delta0 w/ cleanup",
        prefix="0_chord_default_sd_delta0.0",
    ),
    Experiment(
        label="tstart0.9 tend0.3 delta0 w/o cleanup",
        prefix="0_chord_default_sd_delta0.0_tstart0.9_no_cleanup",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For four selected ChordEdit ablations, find the top-K samples by "
            "PSNR and CLIP-Edited score across the PIE-Bench 700-sample metrics."
        )
    )
    parser.add_argument("--evaluation-csvs", type=Path, nargs="+", default=list(DEFAULT_EVALUATION_CSVS))
    parser.add_argument("--sample-meta", type=Path, default=DEFAULT_SAMPLE_META)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def read_wide_metrics(paths: list[Path]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype={"file_id": str})
        merged = frame if merged is None else merged.merge(frame, on="file_id", how="outer")
    if merged is None:
        raise ValueError("No evaluation CSVs were provided.")
    return merged


def read_sample_meta(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["file_id", "image_path", "editing_type_id"])
    return pd.read_csv(path, dtype={"file_id": str})


def require_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise KeyError(f"Missing required column: {column}")


def experiment_table(df: pd.DataFrame, experiment: Experiment) -> pd.DataFrame:
    columns = {"file_id": df["file_id"]}
    for label, suffix in {**METRIC_COLUMNS, **EXTRA_COLUMNS}.items():
        source_column = f"{experiment.prefix}|{suffix}"
        require_column(df, source_column)
        columns[label] = df[source_column]
    table = pd.DataFrame(columns)
    table.insert(1, "experiment", experiment.label)
    return table


def top_rows(table: pd.DataFrame, rank_metric: str, top_k: int) -> pd.DataFrame:
    ranked = table.sort_values(rank_metric, ascending=False).head(top_k).copy()
    ranked.insert(2, "rank_metric", rank_metric)
    ranked.insert(3, "rank", range(1, len(ranked) + 1))
    return ranked


def write_markdown(df: pd.DataFrame, out_path: Path) -> None:
    def format_value(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def markdown_table(group: pd.DataFrame, columns: list[str]) -> list[str]:
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for _, row in group.iterrows():
            lines.append("| " + " | ".join(format_value(row[column]) for column in columns) + " |")
        return lines

    lines = ["# Top selected ablation samples", ""]
    for (experiment, rank_metric), group in df.groupby(["experiment", "rank_metric"], sort=False):
        lines.extend([f"## {experiment} | Top {len(group)} {rank_metric}", ""])
        display_columns = ["rank", "file_id", "image_path", "PSNR", "CLIP-Edited", "SSIM", "LPIPS"]
        lines.extend(markdown_table(group, display_columns))
        lines.append("")
    out_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    metrics = read_wide_metrics(args.evaluation_csvs)
    sample_meta = read_sample_meta(args.sample_meta)

    rows = []
    for experiment in EXPERIMENTS:
        table = experiment_table(metrics, experiment)
        for rank_metric in METRIC_COLUMNS:
            rows.append(top_rows(table, rank_metric, args.top_k))

    result = pd.concat(rows, ignore_index=True)
    if not sample_meta.empty:
        meta_columns = [column for column in ("file_id", "image_path", "editing_type_id") if column in sample_meta.columns]
        result = result.merge(sample_meta[meta_columns], on="file_id", how="left")
        ordered = [
            "experiment",
            "rank_metric",
            "rank",
            "file_id",
            "image_path",
            "editing_type_id",
            "PSNR",
            "CLIP-Edited",
            "Structure Dist.",
            "LPIPS",
            "MSE",
            "SSIM",
            "CLIP Src.",
            "CLIP Tgt.",
        ]
        result = result[[column for column in ordered if column in result.columns]]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_csv, index=False, float_format="%.6f")
    write_markdown(result, args.out_md)

    print(f"Wrote {len(result)} rows to {args.out_csv}")
    print(f"Wrote markdown summary to {args.out_md}")


if __name__ == "__main__":
    main()
