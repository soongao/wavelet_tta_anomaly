#!/usr/bin/env python3
"""Build a Figure 1 candidate from a real WPTA mechanism visualization.

The source PNG is a 7-panel real model-output visualization. This script
keeps the original rendered outputs intact, crops the panel strip, and adds
paper-facing labels plus a short visual note. It does not synthesize heatmaps
or edit model predictions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_SOURCE = Path(
    "/Users/bytedance/code/AnomalyCLIP/cached_results/prototype_tuned/"
    "mechanism_viz/mvtec_cable_000083_cable.png"
)
DEFAULT_OUT = Path("outputs/figures/figure1_motivated_example_mvtec_cable.png")
DEFAULT_PDF_OUT = Path("outputs/figures/figure1_motivated_example_mvtec_cable.pdf")
DEFAULT_PANELS_DIR = Path("outputs/figures/figure1_panels")


PANEL_TITLES = [
    "Input",
    "GT / target",
    "Fixed prototype map",
    "Direct wavelet cue",
    "Boundary-aware reliability",
    "Selected evidence",
    "WPTA final map",
]


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    width: int,
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    tw, th = text_size(draw, text, font)
    x, y = xy
    draw.text((x + max(0, (width - tw) // 2), y), text, font=font, fill=fill)


def crop_panels(image: Image.Image, n_panels: int = 7) -> list[Image.Image]:
    width, height = image.size
    panel_w = width // n_panels
    panels = []
    for idx in range(n_panels):
        left = idx * panel_w
        right = width if idx == n_panels - 1 else (idx + 1) * panel_w
        panels.append(image.crop((left, 0, right, height)))
    return panels


def add_panel_labels(panels: list[Image.Image], title_h: int = 54) -> list[Image.Image]:
    title_font = load_font(25, bold=True)
    index_font = load_font(19, bold=True)
    labelled = []
    for idx, panel in enumerate(panels):
        canvas = Image.new("RGB", (panel.width, panel.height + title_h), "white")
        canvas.paste(panel.convert("RGB"), (0, title_h))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, panel.width - 1, title_h - 1), fill=(248, 248, 248), outline=(210, 210, 210))
        draw.text((12, 15), f"({chr(ord('a') + idx)})", font=index_font, fill=(45, 45, 45))
        draw_centered_text(draw, (0, 13), panel.width, PANEL_TITLES[idx], title_font, (25, 25, 25))
        labelled.append(canvas)
    return labelled


def assemble_figure(panels: list[Image.Image]) -> Image.Image:
    margin = 28
    gutter = 14
    note_h = 92
    title_h = 42
    width = sum(p.width for p in panels) + gutter * (len(panels) - 1) + 2 * margin
    height = max(p.height for p in panels) + note_h + title_h + 2 * margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(26, bold=True)
    note_font = load_font(22)
    small_font = load_font(17)

    title = "Wavelet cues should supervise evidence selection, not replace semantic anomaly scoring"
    draw.text((margin, margin - 4), title, font=title_font, fill=(20, 20, 20))

    y0 = margin + title_h
    x = margin
    for panel in panels:
        canvas.paste(panel, (x, y0))
        draw.rectangle((x, y0, x + panel.width - 1, y0 + panel.height - 1), outline=(190, 190, 190), width=1)
        x += panel.width + gutter

    note_y = y0 + max(p.height for p in panels) + 18
    draw.rounded_rectangle(
        (margin, note_y, width - margin, note_y + 58),
        radius=10,
        fill=(252, 252, 252),
        outline=(215, 215, 215),
    )
    draw.text(
        (margin + 16, note_y + 13),
        "Observation: direct wavelet cues may activate structure boundaries; WPTA uses boundary-aware reliability to select visual anchors.",
        font=note_font,
        fill=(30, 30, 30),
    )
    draw.text(
        (margin + 16, note_y + 43),
        "Source: real model-output strip, MVTec cable sample 000083. Labels are added; heatmaps and evidence overlays are not manually edited.",
        font=small_font,
        fill=(95, 95, 95),
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pdf-out", type=Path, default=DEFAULT_PDF_OUT)
    parser.add_argument("--panels-dir", type=Path, default=DEFAULT_PANELS_DIR)
    args = parser.parse_args()

    image = Image.open(args.source).convert("RGBA")
    panels = crop_panels(image)
    args.panels_dir.mkdir(parents=True, exist_ok=True)
    for idx, panel in enumerate(panels):
        panel.save(args.panels_dir / f"panel_{idx + 1:02d}_{PANEL_TITLES[idx].lower().replace(' / ', '_').replace(' ', '_')}.png")

    labelled = add_panel_labels(panels)
    figure = assemble_figure(labelled)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.save(args.out)
    args.pdf_out.parent.mkdir(parents=True, exist_ok=True)
    figure.save(args.pdf_out, "PDF", resolution=300.0)
    print(f"Wrote {args.out} ({figure.width}x{figure.height})")
    print(f"Wrote {args.pdf_out}")
    print(f"Wrote panel crops to {args.panels_dir}")


if __name__ == "__main__":
    main()
