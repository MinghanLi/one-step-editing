from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

from run_local_ablation import (
    DEFAULT_CENTER_CROP,
    DEFAULT_DATA_ROOT,
    DEFAULT_EDIT_CONFIG,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL_ROOT,
    DEFAULT_PRECISION,
    DEFAULT_SEED,
    DEFAULT_USE_ATTENTION_MASK,
    DEFAULT_USE_SAFETY_CHECKER,
    dtype_from_precision,
    ensure_dir,
    load_local_records,
    param_slug,
    resolve_component_paths,
    run_single_edit,
    write_json,
)

if TYPE_CHECKING:
    from PIL import Image


LOGGER = logging.getLogger("grid_ablation")
GRID_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 5x5 ChordEdit grid ablation: x-axis=t_start, y-axis=t_end."
    )
    parser.add_argument(
        "--model-root",
        type=str,
        default=DEFAULT_MODEL_ROOT,
        help="Root folder containing unet/scheduler/text_encoder/tokenizer/vae subfolders.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help="Folder containing example subdirectories with i.jpg and meta.jsonl.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="ablation_outputs/grid_t_start_t_end_sym_sdxl_turbo",
        help="Where to save 5x5 ablation grids and cell images.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default=DEFAULT_PRECISION,
        help="Model loading precision.",
    )
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--max-records", type=int, default=None, help="Only process the first N prompt records.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated images.")
    parser.add_argument(
        "--no-center-crop",
        dest="center_crop",
        action="store_false",
        default=DEFAULT_CENTER_CROP,
        help="Disable center crop before resizing.",
    )
    parser.add_argument(
        "--use-attention-mask",
        action="store_true",
        default=DEFAULT_USE_ATTENTION_MASK,
        help="Pass attention masks to the text encoder.",
    )
    parser.add_argument(
        "--safety-checker",
        action="store_true",
        default=DEFAULT_USE_SAFETY_CHECKER,
        help="Enable StableDiffusion safety checker.",
    )
    parser.add_argument("--cell-size", type=int, default=192, help="Pixel size of each image in the grid.")
    return parser.parse_args()


def make_axis_grid(
    *,
    title: str,
    cells: List[List["Image.Image"]],
    t_start_values: List[float],
    t_end_values: List[float],
    destination: Path,
    cell_size: int,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    label_width = 96
    header_height = 72
    title_height = 34
    padding = 8
    font = ImageFont.load_default()

    width = label_width + len(t_start_values) * cell_size + padding * 2
    height = title_height + header_height + len(t_end_values) * cell_size + padding * 2
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((padding, 10), title, fill="black", font=font)
    draw.text((label_width + padding, title_height + 8), "x axis: t_start", fill="black", font=font)
    draw.text((padding, title_height + header_height // 2), "y: t_end", fill="black", font=font)

    x0 = label_width + padding
    y0 = title_height + header_height
    for col, value in enumerate(t_start_values):
        x = x0 + col * cell_size
        draw.text((x + 8, title_height + 34), f"{value:.1f}", fill="black", font=font)

    for row, value in enumerate(t_end_values):
        y = y0 + row * cell_size
        draw.text((padding, y + 8), f"{value:.1f}", fill="black", font=font)

        for col, image in enumerate(cells[row]):
            x = x0 + col * cell_size
            thumb = image.convert("RGB").resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (x, y))
            draw.rectangle((x, y, x + cell_size - 1, y + cell_size - 1), outline="#d0d0d0")

    canvas.save(destination)


def save_source_copy(source_image: "Image.Image", destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        return
    source_image.save(destination)


def value_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def t_delta_conditions(base_config: Dict[str, Any]) -> List[float]:
    values = [float(base_config.get("t_delta", 0.0)), 0.0]
    unique_values: List[float] = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)
    return unique_values


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    import torch
    from PIL import Image

    from pipeline_chord import ChordEditPipeline

    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    component_paths = resolve_component_paths(args.model_root)
    base_config = dict(DEFAULT_EDIT_CONFIG)
    records = load_local_records(data_root, args.max_records)
    values = list(GRID_VALUES)
    delta_values = t_delta_conditions(base_config)

    LOGGER.info("Loaded %d record(s) from %s", len(records), data_root)
    LOGGER.info("Base edit config copied from app.py: %s", base_config)
    LOGGER.info("Grid values: %s", values)
    LOGGER.info("t_delta conditions: %s", delta_values)
    LOGGER.info("Output root: %s", output_root)

    torch_dtype = dtype_from_precision(args.precision)
    pipeline = ChordEditPipeline.from_local_weights(
        component_paths=component_paths,
        default_edit_config=base_config,
        device=args.device,
        torch_dtype=torch_dtype,
        image_size=args.image_size,
        use_center_crop=args.center_crop,
        compute_dtype=torch.float32,
        use_attention_mask=args.use_attention_mask,
        use_safety_checker=args.safety_checker,
    )

    manifest: Dict[str, Any] = {
        "data_root": str(data_root),
        "model_root": str(Path(args.model_root).expanduser().resolve()),
        "base_edit_config": base_config,
        "seed": args.seed,
        "x_axis": {"name": "t_start", "values": values},
        "y_axis": {"name": "t_end", "values": values},
        "t_delta_values": delta_values,
        "records": [],
    }

    for record_index, record in enumerate(records, start=1):
        record_key = f"{record.sample_name}_{record.edit_id}"
        sample_dir = output_root / record_key
        ensure_dir(sample_dir)

        LOGGER.info("Processing %d/%d: %s", record_index, len(records), record_key)
        with Image.open(record.image_path) as img:
            source_image = img.convert("RGB")
        save_source_copy(source_image, sample_dir / "source.png", args.overwrite)

        record_payload = {
            "sample": record.sample_name,
            "edit_id": record.edit_id,
            "image_path": str(record.image_path),
            "source_prompt": record.source_prompt,
            "target_prompt": record.target_prompt,
            "edit_prompt": record.edit_prompt,
            "t_delta_results": [],
        }

        for t_delta in delta_values:
            condition_name = f"t_delta_{value_slug(t_delta)}"
            condition_dir = sample_dir / condition_name
            cells_dir = condition_dir / "cells"
            ensure_dir(cells_dir)

            grid_cells: List[List[Image.Image]] = []
            cell_outputs: List[Dict[str, Any]] = []
            for t_end in values:
                row_images: List[Image.Image] = []
                for t_start in values:
                    edit_config = dict(base_config)
                    edit_config["t_start"] = t_start
                    edit_config["t_end"] = t_end
                    edit_config["t_delta"] = t_delta

                    filename = f"{param_slug('t_start', t_start)}__{param_slug('t_end', t_end)}.png"
                    out_path = cells_dir / filename
                    if out_path.exists() and not args.overwrite:
                        with Image.open(out_path) as cached:
                            generated = cached.convert("RGB")
                    else:
                        generated = run_single_edit(
                            pipeline=pipeline,
                            source_image=source_image,
                            record=record,
                            edit_config=edit_config,
                            seed=args.seed,
                        )
                        generated.save(out_path)

                    row_images.append(generated)
                    cell_outputs.append(
                        {
                            "t_start": t_start,
                            "t_end": t_end,
                            "t_delta": t_delta,
                            "path": str(out_path),
                        }
                    )
                grid_cells.append(row_images)

            grid_path = condition_dir / "grid_t_start_x_t_end.png"
            make_axis_grid(
                title=f"{record_key} | x=t_start, y=t_end | t_delta={t_delta:g}",
                cells=grid_cells,
                t_start_values=values,
                t_end_values=values,
                destination=grid_path,
                cell_size=args.cell_size,
            )

            record_payload["t_delta_results"].append(
                {
                    "t_delta": t_delta,
                    "directory": str(condition_dir),
                    "grid": str(grid_path),
                    "cells": cell_outputs,
                }
            )

        write_json(sample_dir / "meta.json", record_payload)
        manifest["records"].append(record_payload)

    ensure_dir(output_root)
    write_json(output_root / "manifest.json", manifest)
    LOGGER.info("Finished grid ablations. Results saved in %s", output_root)


if __name__ == "__main__":
    main()
