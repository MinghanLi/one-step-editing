from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from PIL import Image

    from pipeline_chord import ChordEditPipeline


LOGGER = logging.getLogger("local_ablation")

COMPONENT_SUBDIRS: Dict[str, str] = {
    "unet_path": "unet",
    "scheduler_path": "scheduler",
    "text_encoder_path": "text_encoder",
    "tokenizer_path": "tokenizer",
    "vae_path": "vae",
}

# Defaults copied from app.py.
DEFAULT_MODEL_ROOT = "/sd-turbo"
DEFAULT_EDIT_CONFIG: Dict[str, Any] = {
    "noise_samples": 1,
    "n_steps": 1,
    "t_start": 0.90,
    "t_end": 0.30,
    "t_delta": 0.15,
    "step_scale": 1.0,
    "cleanup": True,
}
DEFAULT_SEED = 42
DEFAULT_PRECISION = "fp32"
DEFAULT_IMAGE_SIZE = 512
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "images"
DEFAULT_CENTER_CROP = True
DEFAULT_USE_ATTENTION_MASK = False
DEFAULT_USE_SAFETY_CHECKER = False
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class LocalRecord:
    sample_name: str
    image_path: Path
    source_prompt: str
    target_prompt: str
    edit_prompt: str
    edit_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ChordEdit t_start/t_end ablations on local ./images examples."
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
        default="ablation_outputs",
        help="Where to save ablation images.",
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
    return parser.parse_args()


def ablation_values() -> List[float]:
    return [round(i / 10.0, 1) for i in range(11)]


def dtype_from_precision(value: Optional[str]) -> "torch.dtype":
    import torch

    precision = (value or DEFAULT_PRECISION).lower()
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if precision not in mapping:
        raise ValueError(f"Unsupported precision '{value}'. Choose from {list(mapping)}.")
    return mapping[precision]


def resolve_component_paths(model_root: str | Path) -> Dict[str, str]:
    root = Path(model_root).expanduser().resolve()
    return {key: str((root / subdir).resolve()) for key, subdir in COMPONENT_SUBDIRS.items()}


def select_image_file(folder: Path) -> Path:
    candidates = [
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    ]
    if not candidates:
        raise FileNotFoundError(f"No RGB image found inside {folder}")

    preferred = sorted(
        (p for p in candidates if p.stem.lower() in {"i", "image", "original"}),
        key=lambda p: p.name,
    )
    if preferred:
        return preferred[0]
    return sorted(candidates, key=lambda p: p.name)[0]


def param_slug(param_name: str, value: float) -> str:
    return f"{param_name}_{value:.1f}".replace(".", "p")


def load_local_records(data_root: Path, max_records: Optional[int]) -> List[LocalRecord]:
    records: List[LocalRecord] = []
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    for subdir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        meta_file = subdir / "meta.jsonl"
        if not meta_file.exists():
            continue
        image_path = select_image_file(subdir)

        with meta_file.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    meta = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {meta_file}:{line_number}: {exc}") from exc

                edit_id = str(meta.get("edit_id") or f"line{line_number}")
                records.append(
                    LocalRecord(
                        sample_name=subdir.name,
                        image_path=image_path,
                        source_prompt=str(meta.get("original_prompt", "")).strip(),
                        target_prompt=str(meta.get("edited_prompt", "")).strip(),
                        edit_prompt=str(meta.get("edit_prompt", "")).strip(),
                        edit_id=edit_id,
                    )
                )
                if max_records is not None and len(records) >= max_records:
                    return records

    if not records:
        raise FileNotFoundError(f"No records found under {data_root}; expected */meta.jsonl files.")
    return records


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_source_copy(source_image: Image.Image, destination: Path, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        return
    source_image.save(destination)


def run_single_edit(
    pipeline: "ChordEditPipeline",
    source_image: Image.Image,
    record: LocalRecord,
    edit_config: Dict[str, Any],
    seed: int,
) -> Image.Image:
    result = pipeline(
        image=source_image,
        source_prompt=record.source_prompt,
        target_prompt=record.target_prompt,
        edit_config=edit_config,
        seed=seed,
        output_type="pil",
    )
    images = result.images
    if not isinstance(images, list) or not images:
        raise RuntimeError(f"Pipeline returned no PIL image for {record.sample_name}/{record.edit_id}")
    return images[0]


def make_labeled_grid(
    title: str,
    source_image: Image.Image,
    image_items: Iterable[tuple[str, Image.Image]],
    destination: Path,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    items = [("source", source_image), *image_items]
    thumb_size = 192
    label_height = 26
    padding = 10
    columns = 4
    rows = (len(items) + columns - 1) // columns
    title_height = 34
    width = columns * thumb_size + (columns + 1) * padding
    height = title_height + rows * (thumb_size + label_height + padding) + padding

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((padding, 10), title, fill="black", font=font)

    for idx, (label, image) in enumerate(items):
        row, col = divmod(idx, columns)
        x = padding + col * (thumb_size + padding)
        y = title_height + row * (thumb_size + label_height + padding)
        thumb = image.convert("RGB").resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (x, y))
        draw.text((x, y + thumb_size + 6), label, fill="black", font=font)

    canvas.save(destination)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    component_paths = resolve_component_paths(args.model_root)
    base_config = dict(DEFAULT_EDIT_CONFIG)
    records = load_local_records(data_root, args.max_records)
    values = ablation_values()

    LOGGER.info("Loaded %d record(s) from %s", len(records), data_root)
    LOGGER.info("Base edit config copied from app.py: %s", base_config)
    LOGGER.info("Ablation values: %s", values)
    LOGGER.info("Output root: %s", output_root)

    import torch
    from PIL import Image

    from pipeline_chord import ChordEditPipeline

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
        "values": values,
        "records": [],
    }

    for record_index, record in enumerate(records, start=1):
        record_key = f"{record.sample_name}_{record.edit_id}"
        LOGGER.info("Processing %d/%d: %s", record_index, len(records), record_key)
        with Image.open(record.image_path) as img:
            source_image = img.convert("RGB")

        record_entry: Dict[str, Any] = {
            "sample": record.sample_name,
            "edit_id": record.edit_id,
            "image_path": str(record.image_path),
            "source_prompt": record.source_prompt,
            "target_prompt": record.target_prompt,
            "edit_prompt": record.edit_prompt,
            "sweeps": {},
        }

        for sweep_param in ("t_start", "t_end"):
            sweep_dir = output_root / sweep_param / record_key
            ensure_dir(sweep_dir)
            save_source_copy(source_image, sweep_dir / "source.png", args.overwrite)

            images_for_grid: List[tuple[str, Image.Image]] = []
            saved_paths: List[str] = []
            for value in values:
                edit_config = dict(base_config)
                edit_config[sweep_param] = value
                out_path = sweep_dir / f"{param_slug(sweep_param, value)}.png"

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

                images_for_grid.append((f"{sweep_param}={value:.1f}", generated))
                saved_paths.append(str(out_path))

            grid_path = sweep_dir / f"{sweep_param}_grid.png"
            make_labeled_grid(
                title=f"{record_key} | fixed params except {sweep_param}",
                source_image=source_image,
                image_items=images_for_grid,
                destination=grid_path,
            )

            meta_path = sweep_dir / "meta.json"
            write_json(
                meta_path,
                {
                    "record": record_entry,
                    "sweep_param": sweep_param,
                    "fixed_edit_config": base_config,
                    "values": values,
                    "outputs": saved_paths,
                    "grid": str(grid_path),
                },
            )
            record_entry["sweeps"][sweep_param] = {
                "directory": str(sweep_dir),
                "grid": str(grid_path),
                "outputs": saved_paths,
            }

        manifest["records"].append(record_entry)

    ensure_dir(output_root)
    write_json(output_root / "manifest.json", manifest)
    LOGGER.info("Finished ablations. Results saved in %s", output_root)


if __name__ == "__main__":
    main()
