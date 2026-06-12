#!/usr/bin/env python3
"""Make image rows for selected ablation samples."""

from __future__ import annotations

import argparse
import base64
import html
import io
import math
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DEFAULT_ROOT = Path("datasets/PIE-Bench_v1/output/ChordEdit/annotation_images")
DEFAULT_EVALUATION_RESULTS_ROOT = DEFAULT_ROOT.parent / "evaluation_results"
DEFAULT_PIE_ROOT = Path("datasets/PIE-Bench_v1")
DEFAULT_SELECTED_CSV = DEFAULT_EVALUATION_RESULTS_ROOT / "top5_selected_ablation_samples.csv"
DEFAULT_OUT_DIR = DEFAULT_EVALUATION_RESULTS_ROOT / "selected_ablation_image_rows"
DEFAULT_VS_T075_OUT_DIR = DEFAULT_EVALUATION_RESULTS_ROOT / "selected_default_vs_t075_image_rows"
DEFAULT_ANCHOR_DELTA_SWEEP_OUT_DIR = DEFAULT_EVALUATION_RESULTS_ROOT / "anchor_delta_sweep_image_rows"
DEFAULT_ANCHOR_TSTART_SWEEP_OUT_DIR = DEFAULT_EVALUATION_RESULTS_ROOT / "anchor_tstart_sweep_image_rows"
DEFAULT_ANCHOR_TEND_SWEEP_OUT_DIR = DEFAULT_EVALUATION_RESULTS_ROOT / "anchor_tend_sweep_image_rows"
DEFAULT_EVALUATION_CSVS = (
    DEFAULT_ROOT / "evaluation_result_ablation.csv",
)

HEADER_LABELS = (
    "Real image",
    "Naive ChordEdit w/ Prox",
    "Naive ChordEdit w/o Prox",
    "ChordEdit w/ Prox",
    "ChordEdit w/o Prox",
)
DEFAULT_MAPPING = DEFAULT_PIE_ROOT / "mapping_file.json"


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    directory: str
    metric_prefix: str


@dataclass(frozen=True)
class TileSource:
    path: Path | None
    width: int
    height: int
    image: Image.Image | None = None


COLUMNS = (
    ColumnSpec(
        name="tstart0.9_tend0.3_delta0_cleanup",
        directory="chord_default_auto_None_None_0.0",
        metric_prefix="0_chord_default_sd_delta0.0",
    ),
    ColumnSpec(
        name="tstart0.9_tend0.3_delta0_no_cleanup",
        directory="chord_default_auto_0.9_None_0.0_no_cleanup",
        metric_prefix="0_chord_default_sd_delta0.0_tstart0.9_no_cleanup",
    ),
    ColumnSpec(
        name="tstart0.9_tend0.3_delta0.15_cleanup",
        directory="chord_default_auto_None_None_None",
        metric_prefix="0_chord_default_sd_delta0.15",
    ),
    ColumnSpec(
        name="tstart0.9_tend0.3_delta0.15_no_cleanup",
        directory="chord_default_sd_None_None_None_no_cleanup",
        metric_prefix="0_chord_default_sd_delta0.15_no_cleanup",
    ),
)

DELTA_SWEEP_COLUMNS = (
    ColumnSpec(
        name="delta0.05",
        directory="chord_default_auto_None_None_0.05",
        metric_prefix="0_chord_default_sd_delta0.05",
    ),
    ColumnSpec(
        name="delta0.10",
        directory="chord_default_auto_None_None_0.1",
        metric_prefix="0_chord_default_sd_delta0.1",
    ),
    ColumnSpec(
        name="delta0.15",
        directory="chord_default_auto_None_None_None",
        metric_prefix="0_chord_default_sd_delta0.15",
    ),
    ColumnSpec(
        name="delta0.20",
        directory="chord_default_auto_None_None_0.2",
        metric_prefix="0_chord_default_sd_delta0.2",
    ),
    ColumnSpec(
        name="delta0.25",
        directory="chord_default_auto_None_None_0.25",
        metric_prefix="0_chord_default_sd_delta0.25",
    ),
)

TSTART_SWEEP_COLUMNS = (
    ColumnSpec(
        name="t0.6",
        directory="chord_default_auto_0.6_None_None",
        metric_prefix="0_chord_default_sd_tstart0.6",
    ),
    ColumnSpec(
        name="t0.7",
        directory="chord_default_auto_0.7_None_None",
        metric_prefix="0_chord_default_sd_tstart0.7",
    ),
    ColumnSpec(
        name="t0.8",
        directory="chord_default_auto_0.8_None_None",
        metric_prefix="0_chord_default_sd_tstart0.8",
    ),
    ColumnSpec(
        name="t0.9",
        directory="chord_default_auto_None_None_None",
        metric_prefix="0_chord_default_sd_tstart0.9",
    ),
    ColumnSpec(
        name="t1.0",
        directory="chord_default_auto_1.0_None_None",
        metric_prefix="0_chord_default_sd_tstart1.0",
    ),
)

TEND_SWEEP_COLUMNS = (
    ColumnSpec(
        name="tend0.1",
        directory="chord_default_auto_None_0.1_None",
        metric_prefix="0_chord_default_sd_tend0.1",
    ),
    ColumnSpec(
        name="tend0.2",
        directory="chord_default_auto_None_0.2_None",
        metric_prefix="0_chord_default_sd_tend0.2",
    ),
    ColumnSpec(
        name="tend0.3",
        directory="chord_default_auto_None_None_None",
        metric_prefix="0_chord_default_sd_tend0.3",
    ),
    ColumnSpec(
        name="tend0.4",
        directory="chord_default_auto_None_0.4_None",
        metric_prefix="0_chord_default_sd_tend0.4",
    ),
    ColumnSpec(
        name="tend0.5",
        directory="chord_default_auto_None_0.5_None",
        metric_prefix="0_chord_default_sd_tend0.5",
    ),
)

METHOD_SETS = {
    "prox_ablation": {
        "headers": HEADER_LABELS,
        "columns": COLUMNS,
        "out_dir": DEFAULT_OUT_DIR,
        "add_relative_diff": False,
    },
    "default_vs_t075": {
        "headers": (
            "Real image",
            r"ChordEdit (t=0.9, δ=0.15)",
            r"ChordEdit (t=0.75, δ=0)",
            "Relative difference",
        ),
        "columns": (
            ColumnSpec(
                name="default_t0.9_delta0.15_tend0.3",
                directory="chord_default_auto_None_None_None",
                metric_prefix="0_chord_default_sd_delta0.15",
            ),
            ColumnSpec(
                name="t0.75_delta0_tend0.3",
                directory="chord_default_auto_0.75_None_0.0",
                metric_prefix="0_chord_default_sd_delta0.0_tstart0.75",
            ),
        ),
        "out_dir": DEFAULT_VS_T075_OUT_DIR,
        "add_relative_diff": True,
    },
    "anchor_delta_sweep": {
        "headers": ("Real image", "δ=0.05", "δ=0.10", "δ=0.15", "δ=0.20", "δ=0.25"),
        "columns": DELTA_SWEEP_COLUMNS,
        "out_dir": DEFAULT_ANCHOR_DELTA_SWEEP_OUT_DIR,
        "add_relative_diff": False,
    },
    "anchor_tstart_sweep": {
        "headers": ("Real image", "t=0.6", "t=0.7", "t=0.8", "t=0.9", "t=1.0"),
        "columns": TSTART_SWEEP_COLUMNS,
        "out_dir": DEFAULT_ANCHOR_TSTART_SWEEP_OUT_DIR,
        "add_relative_diff": False,
    },
    "anchor_tend_sweep": {
        "headers": ("Real image", "t**=0.1", "t**=0.2", "t**=0.3", "t**=0.4", "t**=0.5"),
        "columns": TEND_SWEEP_COLUMNS,
        "out_dir": DEFAULT_ANCHOR_TEND_SWEEP_OUT_DIR,
        "add_relative_diff": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For selected sample IDs, concatenate four ChordEdit ablation outputs "
            "plus the original image into a single row with white gaps."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pie-root", type=Path, default=DEFAULT_PIE_ROOT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--evaluation-csvs", type=Path, nargs="+", default=list(DEFAULT_EVALUATION_CSVS))
    parser.add_argument("--selected-csv", type=Path, default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--method-set", choices=sorted(METHOD_SETS), default="prox_ablation")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--sample-ids", nargs="*", default=None, help="Optional file_id list. Defaults to all IDs in --selected-csv.")
    parser.add_argument("--gap", type=int, default=24, help="White gap in pixels between columns.")
    parser.add_argument("--height", type=int, default=None, help="Optional output tile height. Defaults to source image height.")
    parser.add_argument("--header-height", type=int, default=96, help="White label area above each image row.")
    parser.add_argument("--caption-height", type=int, default=74, help="White caption area under each image row.")
    parser.add_argument("--font-size", type=int, default=30)
    parser.add_argument("--metric-font-size", type=int, default=22)
    parser.add_argument(
        "--formats",
        choices=("png", "svg", "pdf"),
        nargs="+",
        default=["pdf"],
        help="Output formats. SVG keeps labels/metrics/caption as vector text. Defaults to PDF only.",
    )
    return parser.parse_args()


def normalize_file_id(sample_id: str) -> str:
    return Path(sample_id).stem


def selected_samples(
    path: Path,
    sample_ids: list[str] | None,
    mapping: dict[str, dict[str, object]],
) -> pd.DataFrame:
    if sample_ids is not None:
        rows = []
        missing = []
        for sample_id in sample_ids:
            file_id = normalize_file_id(sample_id)
            record = mapping.get(file_id)
            if record is None or "image_path" not in record:
                missing.append(sample_id)
                continue
            rows.append({"file_id": file_id, "image_path": str(record["image_path"])})
        if missing:
            raise ValueError(f"Sample IDs not found in mapping file: {', '.join(missing)}")
        return pd.DataFrame(rows).drop_duplicates("file_id", keep="first").reset_index(drop=True)

    if not path.exists():
        raise FileNotFoundError(path)
    samples = pd.read_csv(path, dtype={"file_id": str})
    samples = samples[["file_id", "image_path"]].drop_duplicates("file_id", keep="first")
    if mapping:
        mapped_paths = samples["file_id"].map(lambda file_id: mapping.get(file_id, {}).get("image_path"))
        samples["image_path"] = samples["image_path"].fillna(mapped_paths)
    missing_paths = samples[samples["image_path"].isna()]["file_id"].tolist()
    if missing_paths:
        raise ValueError(f"Missing image_path for sample IDs: {', '.join(missing_paths)}")
    return samples.sort_values("file_id").reset_index(drop=True)


def resize_to_height(image: Image.Image, height: int | None) -> Image.Image:
    image = image.convert("RGB")
    if height is None or image.height == height:
        return image
    width = round(image.width * height / image.height)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def image_to_float_pixels(image: Image.Image) -> list[tuple[float, float, float]]:
    return [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in image.convert("RGB").getdata()]


def heat_color(value: float) -> tuple[int, int, int]:
    anchors = (
        (0.00, (10, 13, 24)),
        (0.22, (45, 28, 86)),
        (0.48, (151, 47, 100)),
        (0.72, (231, 102, 55)),
        (1.00, (255, 232, 112)),
    )
    value = max(0.0, min(1.0, value))
    for (left_t, left_color), (right_t, right_color) in zip(anchors, anchors[1:]):
        if value <= right_t:
            span = right_t - left_t
            alpha = 0.0 if span == 0 else (value - left_t) / span
            return tuple(
                round(left_channel + (right_channel - left_channel) * alpha)
                for left_channel, right_channel in zip(left_color, right_color)
            )
    return anchors[-1][1]


def relative_difference_tile(left: Image.Image, right: Image.Image) -> Image.Image:
    if left.size != right.size:
        right = right.resize(left.size, Image.Resampling.LANCZOS)
    left_pixels = image_to_float_pixels(left)
    right_pixels = image_to_float_pixels(right)
    values = []
    for left_rgb, right_rgb in zip(left_pixels, right_pixels):
        diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(left_rgb, right_rgb)))
        values.append(diff / math.sqrt(3.0))

    heatmap = Image.new("RGB", left.size)
    heatmap.putdata([heat_color(value) for value in values])
    return heatmap


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def load_mapping(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def read_wide_metrics(paths: list[Path]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype={"file_id": str})
        merged = frame if merged is None else merged.merge(frame, on="file_id", how="outer")
    if merged is None:
        raise ValueError("No evaluation CSVs were provided.")
    return merged.set_index("file_id")


def bracket_text(prompt: str) -> list[str]:
    return [match.strip() for match in re.findall(r"\[([^\]]+)\]", prompt) if match.strip()]


def clean_prompt(prompt: str) -> str:
    return re.sub(r"\[|\]", "", prompt).strip()


def edit_context(mapping: dict[str, dict[str, object]], file_id: str) -> str:
    record = mapping.get(file_id, {})
    original_prompt = str(record.get("original_prompt", "")).strip()
    editing_prompt = str(record.get("editing_prompt", "")).strip()
    instruction = str(record.get("editing_instruction", "")).strip()

    original_terms = bracket_text(original_prompt)
    edited_terms = bracket_text(editing_prompt)
    if original_terms and edited_terms:
        return f"{' / '.join(original_terms)} --> {' / '.join(edited_terms)}"
    if instruction:
        return instruction
    if original_prompt and editing_prompt:
        return f"{clean_prompt(original_prompt)} --> {clean_prompt(editing_prompt)}"
    return file_id


def metric_line(metrics: pd.DataFrame, file_id: str, column: ColumnSpec) -> str:
    if file_id not in metrics.index:
        return ""
    psnr_column = f"{column.metric_prefix}|psnr_unedit_part"
    clip_column = f"{column.metric_prefix}|clip_similarity_target_image_edit_part"
    if psnr_column not in metrics.columns or clip_column not in metrics.columns:
        return ""
    psnr = metrics.at[file_id, psnr_column]
    clip = metrics.at[file_id, clip_column]
    psnr_text = f"{psnr:.2f}" if pd.notna(psnr) else "N/A"
    clip_text = f"{clip:.2f}" if pd.notna(clip) else "N/A"
    return f"PSNR: {psnr_text} | CLIP-Edited: {clip_text}"


def load_tile_paths(pie_root: Path, root: Path, image_path: str, columns: tuple[ColumnSpec, ...]) -> list[Path]:
    paths = [pie_root / "annotation_images" / image_path]
    paths.extend(root / column.directory / image_path for column in columns)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    return paths


def load_tiles(
    pie_root: Path,
    root: Path,
    image_path: str,
    height: int | None,
    columns: tuple[ColumnSpec, ...],
    add_relative_diff: bool,
) -> tuple[list[Image.Image], list[TileSource]]:
    tiles = []
    sources = []
    for path in load_tile_paths(pie_root, root, image_path, columns):
        with Image.open(path) as image:
            tile = resize_to_height(image, height)
            tiles.append(tile)
            sources.append(TileSource(path=path, width=tile.width, height=tile.height))
    if add_relative_diff:
        if len(tiles) < 3:
            raise ValueError("Relative difference requires two generated method images.")
        diff_tile = relative_difference_tile(tiles[1], tiles[2])
        tiles.append(diff_tile)
        sources.append(TileSource(path=None, width=diff_tile.width, height=diff_tile.height, image=diff_tile))
    return tiles, sources


def fit_text(text: str, font: ImageFont.ImageFont, max_width: int, draw: ImageDraw.ImageDraw) -> str:
    if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
        return text
    ellipsis = "..."
    words = text.split()
    if len(words) > 1:
        fitted = ""
        for word in words:
            candidate = f"{fitted} {word}".strip()
            if draw.textbbox((0, 0), candidate + ellipsis, font=font)[2] > max_width:
                return (fitted + ellipsis).strip()
            fitted = candidate
    while text and draw.textbbox((0, 0), text + ellipsis, font=font)[2] > max_width:
        text = text[:-1]
    return text + ellipsis


def make_row(
    tiles: list[Image.Image],
    header_labels: tuple[str, ...],
    gap: int,
    caption: str,
    metric_lines: list[str],
    header_height: int,
    caption_height: int,
    font: ImageFont.ImageFont,
    metric_font: ImageFont.ImageFont,
) -> Image.Image:
    max_height = max(tile.height for tile in tiles)
    total_width = sum(tile.width for tile in tiles) + gap * (len(tiles) - 1)
    row = Image.new("RGB", (total_width, header_height + max_height + caption_height), "white")
    draw = ImageDraw.Draw(row)
    x = 0
    for tile, label, metrics_text in zip(tiles, header_labels, metric_lines):
        label_text = fit_text(label, font, tile.width - 12, draw)
        label_bbox = draw.textbbox((0, 0), label_text, font=font)
        label_width = label_bbox[2] - label_bbox[0]
        label_height = label_bbox[3] - label_bbox[1]
        label_x = x + (tile.width - label_width) // 2
        label_y = 12
        draw.text((label_x, label_y), label_text, fill="black", font=font)

        if metrics_text:
            metrics_text = fit_text(metrics_text, metric_font, tile.width - 12, draw)
            metrics_bbox = draw.textbbox((0, 0), metrics_text, font=metric_font)
            metrics_width = metrics_bbox[2] - metrics_bbox[0]
            metrics_x = x + (tile.width - metrics_width) // 2
            metrics_y = label_y + label_height + 10
            draw.text((metrics_x, metrics_y), metrics_text, fill="#333333", font=metric_font)

        y = header_height + (max_height - tile.height) // 2
        row.paste(tile, (x, y))
        x += tile.width + gap

    fitted_caption = fit_text(caption, font, total_width - 32, draw)
    text_bbox = draw.textbbox((0, 0), fitted_caption, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (total_width - text_width) // 2
    text_y = header_height + max_height + (caption_height - text_height) // 2 - 2
    draw.text((text_x, text_y), fitted_caption, fill="black", font=font)
    return row


def image_data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def tile_data_uri(source: TileSource) -> str:
    if source.path is not None:
        return image_data_uri(source.path)
    if source.image is None:
        raise ValueError("TileSource must have either path or image.")
    buffer = io.BytesIO()
    source.image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def svg_text(
    text: str,
    x: int,
    y: int,
    *,
    size: int,
    fill: str = "black",
    weight: str = "400",
) -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="middle" dominant-baseline="hanging" '
        f'font-family="DejaVu Sans, Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{html.escape(text)}</text>'
    )


def make_svg_row(
    sources: list[TileSource],
    header_labels: tuple[str, ...],
    gap: int,
    caption: str,
    metric_lines: list[str],
    header_height: int,
    caption_height: int,
    font: ImageFont.ImageFont,
    metric_font: ImageFont.ImageFont,
    font_size: int,
    metric_font_size: int,
) -> str:
    max_height = max(source.height for source in sources)
    total_width = sum(source.width for source in sources) + gap * (len(sources) - 1)
    total_height = header_height + max_height + caption_height
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="{total_height}" '
        f'viewBox="0 0 {total_width} {total_height}">',
        f'<rect width="{total_width}" height="{total_height}" fill="white"/>',
    ]

    x = 0
    for source, label, metrics_text in zip(sources, header_labels, metric_lines):
        center_x = x + source.width // 2
        label_text = fit_text(label, font, source.width - 12, measure)
        parts.append(svg_text(label_text, center_x, 12, size=font_size, weight="600"))

        if metrics_text:
            metrics_text = fit_text(metrics_text, metric_font, source.width - 12, measure)
            label_height = measure.textbbox((0, 0), label_text, font=font)[3]
            parts.append(svg_text(metrics_text, center_x, 12 + label_height + 10, size=metric_font_size, fill="#333333"))

        y = header_height + (max_height - source.height) // 2
        parts.append(
            f'<image x="{x}" y="{y}" width="{source.width}" height="{source.height}" '
            f'href="{tile_data_uri(source)}" preserveAspectRatio="none"/>'
        )
        x += source.width + gap

    fitted_caption = fit_text(caption, font, total_width - 32, measure)
    text_height = measure.textbbox((0, 0), fitted_caption, font=font)[3]
    text_y = header_height + max_height + (caption_height - text_height) // 2 - 2
    parts.append(svg_text(fitted_caption, total_width // 2, text_y, size=font_size))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    args = parse_args()
    method_set = METHOD_SETS[args.method_set]
    header_labels = method_set["headers"]
    columns = method_set["columns"]
    out_dir = args.out_dir or method_set["out_dir"]

    mapping = load_mapping(args.mapping)
    samples = selected_samples(args.selected_csv, args.sample_ids, mapping)
    metrics = read_wide_metrics(args.evaluation_csvs)
    font = load_font(args.font_size)
    metric_font = load_font(args.metric_font_size)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for sample in samples.itertuples(index=False):
        tiles, sources = load_tiles(
            args.pie_root,
            args.root,
            sample.image_path,
            args.height,
            columns,
            bool(method_set["add_relative_diff"]),
        )
        caption = edit_context(mapping, sample.file_id)
        metric_lines = [""] + [metric_line(metrics, sample.file_id, column) for column in columns]
        if method_set["add_relative_diff"]:
            metric_lines.append("")
        base_out_path = out_dir / f"{sample.file_id}_{args.method_set}"
        out_paths = {}

        if "png" in args.formats:
            row = make_row(
                tiles,
                header_labels,
                args.gap,
                caption,
                metric_lines,
                args.header_height,
                args.caption_height,
                font,
                metric_font,
            )
            png_path = base_out_path.with_suffix(".png")
            row.save(png_path)
            out_paths["png_path"] = str(png_path)

        if "pdf" in args.formats:
            row = make_row(
                tiles,
                header_labels,
                args.gap,
                caption,
                metric_lines,
                args.header_height,
                args.caption_height,
                font,
                metric_font,
            )
            pdf_path = base_out_path.with_suffix(".pdf")
            row.save(pdf_path)
            out_paths["pdf_path"] = str(pdf_path)

        if "svg" in args.formats:
            svg = make_svg_row(
                sources,
                header_labels,
                args.gap,
                caption,
                metric_lines,
                args.header_height,
                args.caption_height,
                font,
                metric_font,
                args.font_size,
                args.metric_font_size,
            )
            svg_path = base_out_path.with_suffix(".svg")
            svg_path.write_text(svg)
            out_paths["svg_path"] = str(svg_path)

        manifest_row = {
            "file_id": sample.file_id,
            "image_path": sample.image_path,
            "caption": caption,
            **out_paths,
        }
        for column, line in zip(columns, metric_lines[1:]):
            manifest_row[f"{column.name}_metrics"] = line
        manifest_rows.append(manifest_row)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Wrote {len(manifest)} image rows to {out_dir}")
    print(f"Method set: {args.method_set}")
    print(f"Formats: {', '.join(args.formats)}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
