#!/usr/bin/env python3
"""Generate the dataset overview figure used by README and docs.

The script reads local development dataset paths from an ignored config file
(`config/dev_paths.json` by default). It does not store machine-specific paths in
the repository, and it regenerates the full figure from data samples rather than
editing an existing PNG.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from datasets import disable_progress_bar
except ImportError:  # pragma: no cover - handled when loaders are used.
    disable_progress_bar = None

from dabench.datasets import load_hf_dataset


@dataclass(frozen=True)
class DatasetCard:
    name: str
    source_badge: str
    input_text: str
    prediction_text: str
    domain_text: str
    source_label: str
    target_label: str
    classes: str
    domains: str
    examples: str
    source_query: dict[str, Any]
    target_query: dict[str, Any]


CARDS = [
    DatasetCard(
        name="iWildCam",
        source_badge="HF",
        input_text="camera trap photo",
        prediction_text="animal species",
        domain_text="camera trap",
        source_label="source camera",
        target_label="new camera",
        classes="182",
        domains="325 train / 91 test",
        examples="217,959 train / 62,894 test",
        source_query={"dataset": "iwildcam", "split": "train", "index": 38577},
        target_query={"dataset": "iwildcam", "split": "test", "index": 1000},
    ),
    DatasetCard(
        name="Camelyon17",
        source_badge="HF",
        input_text="tissue slide",
        prediction_text="tumor",
        domain_text="hospital",
        source_label="source hospital",
        target_label="OOD hospital",
        classes="2",
        domains="5 hospitals",
        examples="455,954",
        source_query={"dataset": "camelyon17", "split": "id_train", "index": 10},
        target_query={"dataset": "camelyon17", "split": "ood_test", "index": 10},
    ),
    DatasetCard(
        name="DomainNet",
        source_badge="HF",
        input_text="object image",
        prediction_text="object class",
        domain_text="visual style",
        source_label="real",
        target_label="sketch",
        classes="345",
        domains="6 styles",
        examples="~587k",
        source_query={"dataset": "domainnet", "split": "train", "domains": ["real"], "index": 20},
        target_query={"dataset": "domainnet", "split": "train", "domains": ["sketch"], "index": 20},
    ),
    DatasetCard(
        name="Office-Home",
        source_badge="HF",
        input_text="office object",
        prediction_text="object class",
        domain_text="visual style",
        source_label="Art",
        target_label="Real World",
        classes="65",
        domains="4 styles",
        examples="~15.6k",
        source_query={"dataset": "office-home", "split": "train", "domains": ["Art"], "index": 20},
        target_query={"dataset": "office-home", "split": "train", "domains": ["Real World"], "index": 20},
    ),
    DatasetCard(
        name="Office-31",
        source_badge="MS",
        input_text="office object",
        prediction_text="object class",
        domain_text="image source",
        source_label="amazon",
        target_label="webcam",
        classes="31",
        domains="3 sources",
        examples="4.1k",
        source_query={"dataset": "office-31", "split": "all", "domains": ["amazon"], "index": 10},
        target_query={"dataset": "office-31", "split": "all", "domains": ["webcam"], "index": 10},
    ),
    DatasetCard(
        name="VisDA-2017",
        source_badge="GH",
        input_text="synthetic / real image",
        prediction_text="object class",
        domain_text="synthetic -> real",
        source_label="synthetic",
        target_label="real target",
        classes="12",
        domains="2",
        examples="152k+ train",
        source_query={"dataset": "visda-2017", "split": "train", "index": 50},
        target_query={"dataset": "visda-2017", "split": "validation", "index": 50},
    ),
]

PATH_KEYS = {
    "iwildcam": ("iwildcam", "hf_prepared"),
    "camelyon17": ("camelyon17", "hf_prepared"),
    "domainnet": ("domainnet", "hf_prepared"),
    "office-home": ("office-home", "hf_prepared"),
    "office-31": ("office-31", "images"),
    "visda-2017": ("visda-2017", "official"),
}

BADGE_COLORS = {
    "HF": "#ffd96a",
    "MS": "#9dccff",
    "GH": "#d9d5e4",
}


def load_paths(config_path: Path) -> dict[str, str]:
    if config_path.is_file():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        datasets = payload.get("datasets", {})
    else:
        datasets = {}

    paths: dict[str, str] = {}
    for dataset_name, (group, key) in PATH_KEYS.items():
        env_name = "DABENCH_" + dataset_name.upper().replace("-", "_") + "_PATH"
        value = os.environ.get(env_name) or datasets.get(group, {}).get(key)
        if value:
            paths[dataset_name] = value
    return paths


def image_from_row(row: dict[str, Any]) -> Image.Image:
    image = row["image"]
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict):
        if image.get("bytes"):
            return Image.open(BytesIO(image["bytes"])).convert("RGB")
        if image.get("path"):
            return Image.open(image["path"]).convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(image)!r}")


def load_sample(query: dict[str, Any], paths: dict[str, str]) -> Image.Image | None:
    dataset_name = query["dataset"]
    path = paths.get(dataset_name)
    if not path:
        return None

    kwargs = {"path": path, "split": query.get("split", "train")}
    if "domains" in query:
        kwargs["domains"] = query["domains"]
    try:
        dataset = load_hf_dataset(dataset_name, **kwargs)
        index = min(int(query.get("index", 0)), len(dataset) - 1)
        return image_from_row(dataset[index])
    except Exception as exc:
        print(f"[warn] failed to load {dataset_name} sample: {exc}", file=sys.stderr)
        return None


def fit_image(image: Image.Image | None, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "#f7f7f4")
    if image is None:
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline="#c6cac7", width=2)
        draw.line((size[0] * 0.16, size[1] * 0.65, size[0] * 0.42, size[1] * 0.36), fill="#8f9a9a", width=6)
        draw.line((size[0] * 0.42, size[1] * 0.36, size[0] * 0.66, size[1] * 0.70), fill="#8f9a9a", width=6)
        draw.line((size[0] * 0.66, size[1] * 0.70, size[0] * 0.90, size[1] * 0.42), fill="#8f9a9a", width=6)
        draw.ellipse((size[0] * 0.18, size[1] * 0.18, size[0] * 0.34, size[1] * 0.42), fill="#839193")
        placeholder_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 20)
        draw.text((size[0] * 0.42, size[1] * 0.38), "not local", font=placeholder_font, fill="#697476")
        return canvas
    image = image.copy()
    image.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def draw_centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    text: str,
    font_obj: ImageFont.FreeTypeFont,
    *,
    fill: str = "#242424",
    spacing: int = 3,
) -> None:
    x1, y1, x2, y2 = box
    max_width = x2 - x1 - 22
    lines: list[str] = []
    for raw_line in text.split("\n"):
        current = ""
        for word in raw_line.split():
            trial = (current + " " + word).strip()
            if draw.textlength(trial, font=font_obj) <= max_width or not current:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)

    line_height = font_obj.getbbox("Ag")[3] - font_obj.getbbox("Ag")[1]
    total_height = len(lines) * line_height + max(0, len(lines) - 1) * spacing
    y = y1 + (y2 - y1 - total_height) / 2 - 1
    for line in lines:
        width = draw.textlength(line, font=font_obj)
        draw.text((x1 + (x2 - x1 - width) / 2, y), line, font=font_obj, fill=fill)
        y += line_height + spacing


def generate(paths: dict[str, str], output: Path) -> None:
    width, height = 1920, 920
    figure = Image.new("RGB", (width, height), "#f4f2ec")
    draw = ImageDraw.Draw(figure)

    regular = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
    bold = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    fonts = {
        "title": font(bold, 35),
        "subtitle": font(regular, 16),
        "header": font(bold, 20),
        "cell": font(regular, 17),
        "cell_bold": font(bold, 18),
        "label": font(bold, 18),
        "badge": font(bold, 10),
        "footnote": font(regular, 11),
    }

    left, label_width, right = 32, 170, width - 34
    col_width = (right - left - label_width) / len(CARDS)
    x_edges = [left, left + label_width] + [
        round(left + label_width + col_width * (idx + 1)) for idx in range(len(CARDS))
    ]
    y0 = 96
    row_heights = [48, 50, 50, 50, 180, 180, 50, 50, 50]
    y_edges = [y0]
    for row_height in row_heights:
        y_edges.append(y_edges[-1] + row_height)

    draw.text((32, 26), "dabench dataset overview", font=fonts["title"], fill="#242424")
    draw.text(
        (33, 70),
        "Inputs, labels, domains, and scale for the current dataset loaders",
        font=fonts["subtitle"],
        fill="#737a7f",
    )

    legend_x = 1225
    for badge, name in [("HF", "Hugging Face"), ("MS", "ModelScope"), ("GH", "GitHub / official")]:
        box_width = int(draw.textlength(name, font=fonts["cell"]) + 58)
        draw.rounded_rectangle((legend_x, 40, legend_x + box_width, 68), radius=9, fill=BADGE_COLORS[badge])
        draw.text((legend_x + 18, 48), badge, font=fonts["badge"], fill="#444444")
        draw.text((legend_x + 48, 45), name, font=fonts["cell"], fill="#3d4145")
        legend_x += box_width + 28

    row_colors = ["#8e8f8c", "#e4e4e1", "#e4e4e1", "#e4e4e1", "#dfeaf2", "#e4f0dc", "#eeeeec", "#eeeeec", "#eeeeec"]
    for idx, color in enumerate(row_colors):
        draw.rectangle((left, y_edges[idx], right, y_edges[idx + 1]), fill=color)
    for x in x_edges:
        draw.line((x, y0, x, y_edges[-1]), fill="#ffffff", width=3)
    for y in y_edges:
        draw.line((left, y, right, y), fill="#ffffff", width=3)

    row_labels = ["dataset", "input (x)", "prediction (y)", "domain (d)", "source example", "target example", "classes", "domains", "examples"]
    for idx, label in enumerate(row_labels):
        draw_centered(draw, (left, y_edges[idx], x_edges[1], y_edges[idx + 1]), label, fonts["label"])

    for col_idx, card in enumerate(CARDS):
        x1, x2 = x_edges[col_idx + 1], x_edges[col_idx + 2]
        cx = (x1 + x2) / 2
        badge_color = BADGE_COLORS[card.source_badge]
        draw.rounded_rectangle((x1 + 28, y_edges[0] + 14, x1 + 58, y_edges[0] + 34), radius=6, fill=badge_color)
        draw.text((x1 + 37, y_edges[0] + 18), card.source_badge, font=fonts["badge"], fill="#333333")
        draw.text((cx - draw.textlength(card.name, font=fonts["header"]) / 2 + 8, y_edges[0] + 12), card.name, font=fonts["header"], fill="#1e1e1e")

        for row_idx, text in enumerate([card.input_text, card.prediction_text, card.domain_text], start=1):
            draw_centered(draw, (x1 + 8, y_edges[row_idx], x2 - 8, y_edges[row_idx + 1]), text, fonts["cell"])

        for offset, (query, label) in enumerate(
            [(card.source_query, card.source_label), (card.target_query, card.target_label)]
        ):
            box = (
                int(x1 + 18),
                int(y_edges[4 + offset] + 18),
                int(x2 - 18),
                int(y_edges[5 + offset] - 18),
            )
            sample = fit_image(load_sample(query, paths), (box[2] - box[0], box[3] - box[1]))
            figure.paste(sample, (box[0], box[1]))
            draw.rounded_rectangle(box, radius=7, outline="#b8bdb9", width=2)
            label_width = int(draw.textlength(label, font=fonts["cell"]) + 16)
            draw.rounded_rectangle((box[0] + 8, box[1] + 8, box[0] + 8 + label_width, box[1] + 31), radius=5, fill="#ffffff")
            draw.text((box[0] + 16, box[1] + 11), label, font=fonts["cell"], fill="#53585c")

        for row_idx, text in zip([6, 7, 8], [card.classes, card.domains, card.examples]):
            draw_centered(draw, (x1 + 8, y_edges[row_idx], x2 - 8, y_edges[row_idx + 1]), text, fonts["cell_bold"])

    draw.text(
        (32, height - 44),
        "HF = Hugging Face Hub, MS = ModelScope, GH = GitHub / official source. "
        "iWildCam domains are camera traps encoded by location ids in the local HF metadata.",
        font=fonts["footnote"],
        fill="#737a7f",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.save(output, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the dabench dataset overview figure.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "dev_paths.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "assets" / "dataset_matrix_overview.png")
    args = parser.parse_args()

    if disable_progress_bar is not None:
        disable_progress_bar()
    generate(load_paths(args.config), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
