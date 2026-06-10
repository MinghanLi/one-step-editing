#!/usr/bin/env python3
"""Calculate mean PIE-Bench metrics from a CSV file."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DEFAULT_CSV = Path(
    "datasets/PIE-Bench_v1/output/ChordEdit/annotation_images/"
    "chordedit_pie700_metrics.csv"
)
SKIP_COLUMNS = {"file_id", "image_path", "editing_type_id"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    return parser.parse_args()


def metric_columns(fieldnames: list[str]) -> list[str]:
    return [name for name in fieldnames if name not in SKIP_COLUMNS]


def calculate_means(csv_path: Path) -> tuple[int, dict[str, float], dict[str, int]]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header row")

        columns = metric_columns(reader.fieldnames)
        totals = {column: 0.0 for column in columns}
        counts = {column: 0 for column in columns}
        row_count = 0

        for row in reader:
            row_count += 1
            for column in columns:
                value = row[column].strip()
                if not value:
                    continue
                metric = float(value)
                if not math.isfinite(metric):
                    continue
                totals[column] += metric
                counts[column] += 1

    means = {
        column: totals[column] / counts[column]
        for column in columns
        if counts[column] > 0
    }
    return row_count, means, counts


def main() -> None:
    args = parse_args()
    row_count, means, counts = calculate_means(args.csv)

    print(f"rows,{row_count}")
    print("metric,mean,valid_count")
    for column, mean in means.items():
        print(f"{column},{mean:.6f},{counts[column]}")


if __name__ == "__main__":
    main()
