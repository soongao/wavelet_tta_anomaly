from __future__ import annotations

import base64
import html
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


OUT_DIR = Path(__file__).resolve().parent
ASSET_DIR = OUT_DIR / "mvtec_three_row_selection_assets"

PNG_OUT = OUT_DIR / "figure4_three_row_clip_arch_wavelet.png"
DRAWIO_OUT = OUT_DIR / "figure4_three_row_clip_arch_wavelet.drawio"
AUDIT_OUT = OUT_DIR / "figure4_three_row_clip_arch_wavelet.audit.md"

INPUT_IMG = ASSET_DIR / "mvtec_cable_input.png"
MAP_FIXED = ASSET_DIR / "mvtec_cable_fixed.png"
MAP_DIRECT = ASSET_DIR / "mvtec_cable_direct.png"
MAP_FINAL = ASSET_DIR / "mvtec_cable_final.png"

W, H = 2400, 1360


PALETTE = {
    "bg": "#F6F8FB",
    "ink": "#0F172A",
    "muted": "#475569",
    "subtle": "#64748B",
    "line": "#CBD5E1",
    "panel": "#FFFFFF",
    "panel_alt": "#F8FAFC",
    "blue": "#2563EB",
    "blue_fill": "#DBEAFE",
    "blue_soft": "#EFF6FF",
    "amber": "#D97706",
    "amber_fill": "#FEF3C7",
    "amber_soft": "#FFFBEB",
    "red": "#DC2626",
    "red_fill": "#FEE2E2",
    "red_soft": "#FEF2F2",
    "green": "#059669",
    "green_fill": "#D1FAE5",
    "green_soft": "#ECFDF5",
    "teal": "#0F766E",
    "teal_fill": "#CCFBF1",
    "orange": "#EA580C",
    "orange_fill": "#FFEDD5",
    "gray": "#334155",
    "gray_fill": "#E2E8F0",
}


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "/Library/Fonts/Arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
        )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


F = {
    "title": load_font(34, True),
    "subtitle": load_font(19),
    "section": load_font(20, True),
    "row_title": load_font(25, True),
    "body": load_font(17),
    "body_bold": load_font(17, True),
    "small": load_font(14),
    "small_bold": load_font(14, True),
    "tiny": load_font(12),
    "module": load_font(19, True),
    "module_small": load_font(15, True),
}


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    if not text:
        return 0, 0
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if text_size(draw, candidate, font)[0] <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines


def draw_text_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str = PALETTE["ink"],
    align: str = "center",
    valign: str = "center",
    line_gap: int = 5,
) -> None:
    x, y, w, h = box
    lines: list[str] = []
    for raw in text.split("\n"):
        lines.extend(wrap_text(draw, raw, font, w - 8) or [""])
    heights = [text_size(draw, line, font)[1] for line in lines]
    total_h = sum(heights) + line_gap * max(0, len(lines) - 1)
    if valign == "top":
        cy = y
    elif valign == "bottom":
        cy = y + h - total_h
    else:
        cy = y + (h - total_h) / 2
    for line, lh in zip(lines, heights):
        lw, _ = text_size(draw, line, font)
        if align == "left":
            tx = x
        elif align == "right":
            tx = x + w - lw
        else:
            tx = x + (w - lw) / 2
        draw.text((tx, cy), line, font=font, fill=fill)
        cy += lh + line_gap


def rounded_rect(
    img: Image.Image,
    box: tuple[int, int, int, int],
    fill: str,
    outline: str = PALETTE["line"],
    radius: int = 18,
    width: int = 2,
    shadow: bool = False,
) -> None:
    draw = ImageDraw.Draw(img)
    x, y, w, h = box
    if shadow:
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rounded_rectangle(
            (x + 5, y + 7, x + w + 5, y + h + 7),
            radius=radius,
            fill=(15, 23, 42, 18),
        )
        img.alpha_composite(overlay)
    draw.rounded_rectangle(
        (x, y, x + w, y + h),
        radius=radius,
        fill=fill,
        outline=outline,
        width=width,
    )


def pill(
    img: Image.Image,
    box: tuple[int, int, int, int],
    text: str,
    fill: str,
    outline: str,
    text_color: str,
    font: ImageFont.ImageFont,
) -> None:
    rounded_rect(img, box, fill, outline, radius=box[3] // 2, width=2, shadow=False)
    draw = ImageDraw.Draw(img)
    draw_text_box(draw, box, text, font, fill=text_color)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = PALETTE["gray"],
    width: int = 4,
    dashed: bool = False,
    head: int = 14,
) -> None:
    sx, sy = start
    ex, ey = end
    if dashed:
        length = math.hypot(ex - sx, ey - sy)
        if length == 0:
            return
        dash = 16
        gap = 10
        ux = (ex - sx) / length
        uy = (ey - sy) / length
        dist = 0.0
        while dist < length - head:
            seg_end = min(dist + dash, length - head)
            draw.line(
                (sx + ux * dist, sy + uy * dist, sx + ux * seg_end, sy + uy * seg_end),
                fill=color,
                width=width,
            )
            dist += dash + gap
    else:
        draw.line((sx, sy, ex, ey), fill=color, width=width)
    angle = math.atan2(ey - sy, ex - sx)
    left = (
        ex - head * math.cos(angle) + head * 0.58 * math.sin(angle),
        ey - head * math.sin(angle) - head * 0.58 * math.cos(angle),
    )
    right = (
        ex - head * math.cos(angle) - head * 0.58 * math.sin(angle),
        ey - head * math.sin(angle) + head * 0.58 * math.cos(angle),
    )
    draw.polygon([end, left, right], fill=color)


def paste_rounded_image(
    img: Image.Image,
    path: Path,
    box: tuple[int, int, int, int],
    radius: int = 16,
    border: str = PALETTE["line"],
) -> None:
    x, y, w, h = box
    src = Image.open(path).convert("RGB")
    fitted = ImageOps.fit(src, (w, h), method=Image.Resampling.LANCZOS)
    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle((0, 0, w, h), radius=radius, fill=255)
    img.paste(fitted, (x, y), mask)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, outline=border, width=2)


def draw_encoder(
    img: Image.Image,
    box: tuple[int, int, int, int],
    label: str,
    fill: str,
    stroke: str,
    frozen: bool = True,
) -> None:
    x, y, w, h = box
    rounded_rect(img, box, fill, stroke, radius=16, width=2, shadow=True)
    draw = ImageDraw.Draw(img)
    draw_text_box(draw, (x + 12, y + 9, w - 24, h - 18), label, F["module"], fill=PALETTE["ink"])
    if frozen:
        pill(img, (x + w - 74, y + 9, 58, 24), "frozen", "#FFFFFF", stroke, PALETTE["subtle"], F["tiny"])


def draw_feature_grid(
    img: Image.Image,
    x: int,
    y: int,
    w: int,
    h: int,
    accent: str,
    red_cells: set[tuple[int, int]] | None = None,
    green_cells: set[tuple[int, int]] | None = None,
) -> None:
    red_cells = red_cells or set()
    green_cells = green_cells or set()
    rounded_rect(img, (x, y, w, h), "#FFFFFF", PALETTE["line"], radius=14, width=2, shadow=False)
    draw = ImageDraw.Draw(img)
    draw_text_box(draw, (x + 10, y + 6, w - 20, 20), "Patch features", F["small_bold"], fill=PALETTE["gray"])
    cols, rows = 8, 4
    pad_x, pad_y = 16, 34
    cell_w = (w - 2 * pad_x - (cols - 1) * 8) // cols
    cell_h = (h - pad_y - 12 - (rows - 1) * 8) // rows
    for r in range(rows):
        for c in range(cols):
            cx = x + pad_x + c * (cell_w + 8)
            cy = y + pad_y + r * (cell_h + 8)
            if (r, c) in red_cells:
                fill, outline = PALETTE["red_fill"], PALETTE["red"]
            elif (r, c) in green_cells:
                fill, outline = PALETTE["green_fill"], PALETTE["green"]
            else:
                fill, outline = accent, "#93A4B8"
            draw.rounded_rectangle((cx, cy, cx + cell_w, cy + cell_h), radius=5, fill=fill, outline=outline, width=1)


def draw_prompt_box(img: Image.Image, x: int, y: int) -> None:
    draw = ImageDraw.Draw(img)
    rounded_rect(img, (x, y, 252, 98), "#FFFFFF", PALETTE["line"], radius=16, width=2, shadow=True)
    draw_text_box(draw, (x + 14, y + 9, 224, 20), "Text prompts", F["small_bold"], fill=PALETTE["gray"], align="left")
    draw.rounded_rectangle((x + 14, y + 38, x + 238, y + 62), radius=8, fill=PALETTE["blue_soft"], outline="#BFDBFE")
    draw.rounded_rectangle((x + 14, y + 66, x + 238, y + 90), radius=8, fill=PALETTE["red_soft"], outline="#FECACA")
    draw.text((x + 22, y + 42), 'Normal: "normal cable"', font=F["tiny"], fill=PALETTE["blue"])
    draw.text((x + 22, y + 70), 'Abnormal: "damaged cable"', font=F["tiny"], fill=PALETTE["red"])


def draw_prototypes(img: Image.Image, x: int, y: int, variant: str) -> None:
    draw = ImageDraw.Draw(img)
    rounded_rect(img, (x, y, 222, 98), "#FFFFFF", PALETTE["line"], radius=14, width=2, shadow=False)
    draw_text_box(draw, (x + 12, y + 6, 198, 20), "Text prototypes", F["small_bold"], fill=PALETTE["gray"])
    if variant == "drift":
        draw.ellipse((x + 26, y + 45, x + 66, y + 85), outline="#94A3B8", width=2)
        draw.ellipse((x + 55, y + 40, x + 99, y + 84), fill=PALETTE["red_fill"], outline=PALETTE["red"], width=2)
        draw_text_box(draw, (x + 55, y + 50, 44, 22), "pN'", F["tiny"], fill=PALETTE["red"])
        draw_text_box(draw, (x + 15, y + 75, 70, 15), "old pN", F["tiny"], fill=PALETTE["subtle"])
    else:
        draw.ellipse((x + 34, y + 42, x + 78, y + 86), fill=PALETTE["blue_fill"], outline=PALETTE["blue"], width=2)
        draw_text_box(draw, (x + 34, y + 52, 44, 20), "pN", F["tiny"], fill=PALETTE["blue"])
    draw.ellipse((x + 137, y + 42, x + 181, y + 86), fill=PALETTE["red_fill"], outline=PALETTE["red"], width=2)
    draw_text_box(draw, (x + 137, y + 52, 44, 20), "pA", F["tiny"], fill=PALETTE["red"])
    if variant == "stable":
        pill(img, (x + 79, y + 72, 54, 22), "gate", PALETTE["green_fill"], PALETTE["green"], PALETTE["green"], F["tiny"])
    elif variant == "fixed":
        pill(img, (x + 83, y + 72, 48, 22), "lock", "#FFFFFF", "#94A3B8", PALETTE["subtle"], F["tiny"])


def draw_similarity(img: Image.Image, x: int, y: int, variant: str) -> None:
    draw = ImageDraw.Draw(img)
    rounded_rect(img, (x, y, 262, 158), "#FFFFFF", PALETTE["line"], radius=16, width=2, shadow=True)
    draw_text_box(draw, (x + 14, y + 8, 234, 22), "Cosine similarity", F["small_bold"], fill=PALETTE["gray"])
    left = x + 42
    top = y + 48
    cw, ch = 74, 20
    draw.text((left + 16, y + 31), "normal", font=F["tiny"], fill=PALETTE["blue"])
    draw.text((left + cw + 14, y + 31), "abnormal", font=F["tiny"], fill=PALETTE["red"])
    values = {
        "fixed": [(0.72, 0.21), (0.64, 0.33), (0.57, 0.41), (0.51, 0.48)],
        "drift": [(0.56, 0.42), (0.50, 0.55), (0.44, 0.68), (0.47, 0.61)],
        "stable": [(0.78, 0.18), (0.73, 0.24), (0.38, 0.83), (0.70, 0.29)],
    }[variant]
    for r, (vn, va) in enumerate(values):
        for c, val in enumerate((vn, va)):
            if c == 0:
                color = (219, 234, 254) if val < 0.65 else (147, 197, 253)
            else:
                color = (254, 226, 226) if val < 0.62 else (248, 113, 113)
            x0 = left + c * cw
            y0 = top + r * (ch + 9)
            draw.rounded_rectangle((x0, y0, x0 + cw - 8, y0 + ch), radius=5, fill=color, outline="#E2E8F0")
            draw.text((x0 + 16, y0 + 3), f"{val:.2f}", font=F["tiny"], fill=PALETTE["ink"])
    draw_text_box(draw, (x + 14, y + 132, 234, 18), "patch-to-text logits", F["tiny"], fill=PALETTE["subtle"])


def draw_adapter(img: Image.Image, box: tuple[int, int, int, int], kind: str) -> None:
    x, y, w, h = box
    draw = ImageDraw.Draw(img)
    if kind == "fixed":
        rounded_rect(img, box, "#FFFFFF", "#94A3B8", radius=18, width=2, shadow=True)
        draw_text_box(draw, (x + 18, y + 12, w - 36, 26), "Fixed CLIP matching", F["module"], fill=PALETTE["ink"])
        draw_text_box(draw, (x + 28, y + 52, w - 56, 46), "Text anchors are frozen; image patches are scored directly.", F["body"], fill=PALETTE["muted"])
        pill(img, (x + 72, y + 116, 116, 32), "no update", PALETTE["gray_fill"], "#94A3B8", PALETTE["gray"], F["small_bold"])
        pill(img, (x + 208, y + 116, 100, 32), "zero-shot", PALETTE["blue_fill"], "#93C5FD", PALETTE["blue"], F["small_bold"])
        draw.line((x + 62, y + 170, x + w - 62, y + 170), fill="#94A3B8", width=3)
        draw.line((x + w // 2 - 16, y + 154, x + w // 2 + 16, y + 186), fill=PALETTE["red"], width=4)
        draw.line((x + w // 2 + 16, y + 154, x + w // 2 - 16, y + 186), fill=PALETTE["red"], width=4)
    elif kind == "direct":
        rounded_rect(img, box, "#FFFFFF", "#F97316", radius=18, width=2, shadow=True)
        draw_text_box(draw, (x + 18, y + 12, w - 36, 26), "Direct prototype adaptation", F["module"], fill=PALETTE["ink"])
        draw_text_box(draw, (x + 24, y + 48, w - 48, 42), "Unfiltered test patches update the anchors.", F["body"], fill=PALETTE["muted"])
        for i, color in enumerate([PALETTE["blue_fill"], PALETTE["red_fill"], PALETTE["blue_fill"], PALETTE["red_fill"], PALETTE["gray_fill"]]):
            cx = x + 56 + i * 48
            draw.rounded_rectangle((cx, y + 104, cx + 30, y + 132), radius=6, fill=color, outline="#94A3B8")
        draw_arrow(draw, (x + 100, y + 150), (x + 245, y + 150), PALETTE["red"], width=4, dashed=False)
        draw.ellipse((x + 270, y + 127, x + 322, y + 179), fill=PALETTE["red_fill"], outline=PALETTE["red"], width=3)
        draw_text_box(draw, (x + 270, y + 142, 52, 20), "drift", F["tiny"], fill=PALETTE["red"])
        draw_text_box(draw, (x + 48, y + 178, w - 96, 22), "abnormal evidence can pull prototypes", F["small"], fill=PALETTE["red"])
    else:
        rounded_rect(img, box, "#FFFFFF", PALETTE["green"], radius=18, width=2, shadow=True)
        draw_text_box(draw, (x + 18, y + 12, w - 36, 26), "Wavelet-guided prototype adapter", F["module"], fill=PALETTE["ink"])
        dwt_box = (x + 30, y + 56, 92, 58)
        rounded_rect(img, dwt_box, PALETTE["teal_fill"], PALETTE["teal"], radius=14, width=2)
        draw_text_box(draw, dwt_box, "DWT", F["module"], fill=PALETTE["teal"])
        pill(img, (x + 146, y + 52, 118, 30), "LL semantic", PALETTE["green_fill"], PALETTE["green"], PALETTE["green"], F["small_bold"])
        pill(img, (x + 146, y + 88, 150, 30), "LH/HL/HH detail", PALETTE["orange_fill"], PALETTE["orange"], PALETTE["orange"], F["small_bold"])
        draw_arrow(draw, (x + 122, y + 85), (x + 146, y + 67), PALETTE["teal"], width=3, head=10)
        draw_arrow(draw, (x + 122, y + 85), (x + 146, y + 103), PALETTE["teal"], width=3, head=10)
        rounded_rect(img, (x + 42, y + 144, w - 84, 50), PALETTE["green_soft"], "#86EFAC", radius=14, width=2)
        draw_text_box(draw, (x + 54, y + 150, w - 108, 36), "reliability gate selects stable normal evidence", F["body"], fill=PALETTE["green"])


def draw_output(img: Image.Image, row_y: int, map_path: Path, note: str, color: str) -> None:
    draw = ImageDraw.Draw(img)
    x = 1742
    y = row_y + 92
    rounded_rect(img, (x - 18, y - 50, 574, 244), "#FFFFFF", PALETTE["line"], radius=18, width=2, shadow=True)
    draw_text_box(draw, (x - 4, y - 39, 176, 24), "Anomaly map", F["small_bold"], fill=PALETTE["gray"])
    paste_rounded_image(img, map_path, (x, y, 154, 154), radius=14, border=color)
    pill(img, (x + 194, y - 4, 156, 32), "output evidence", "#FFFFFF", color, color, F["small_bold"])
    draw_text_box(draw, (x + 192, y + 46, 330, 72), note, F["body"], fill=PALETTE["muted"], align="left")


def draw_header(img: Image.Image) -> None:
    draw = ImageDraw.Draw(img)
    draw_text_box(
        draw,
        (54, 28, 1460, 42),
        "CLIP-style comparison of prototype adaptation paradigms",
        F["title"],
        fill=PALETTE["ink"],
        align="left",
    )
    draw_text_box(
        draw,
        (56, 70, 1540, 30),
        "Same MVTec cable image and same normal/abnormal prompts; highlighted paths show what changes at test time.",
        F["subtitle"],
        fill=PALETTE["muted"],
        align="left",
    )
    legend_x = 1668
    rounded_rect(img, (legend_x, 34, 656, 56), "#FFFFFF", PALETTE["line"], radius=18, width=2, shadow=True)
    pill(img, (legend_x + 20, 47, 118, 30), "image tower", PALETTE["blue_fill"], "#93C5FD", PALETTE["blue"], F["tiny"])
    pill(img, (legend_x + 154, 47, 112, 30), "text tower", PALETTE["amber_fill"], "#FCD34D", PALETTE["amber"], F["tiny"])
    pill(img, (legend_x + 282, 47, 128, 30), "drift risk", PALETTE["red_fill"], "#FCA5A5", PALETTE["red"], F["tiny"])
    pill(img, (legend_x + 426, 47, 154, 30), "wavelet gate", PALETTE["green_fill"], "#86EFAC", PALETTE["green"], F["tiny"])


def draw_row(
    img: Image.Image,
    row_y: int,
    label: str,
    subtitle: str,
    accent: str,
    adapter_kind: str,
    map_path: Path,
    output_note: str,
) -> None:
    draw = ImageDraw.Draw(img)
    rounded_rect(img, (44, row_y, W - 88, 372), "#FFFFFF", "#D8E0EA", radius=24, width=2, shadow=True)
    draw.rectangle((48, row_y + 74, 52, row_y + 330), fill=accent)
    draw_text_box(draw, (70, row_y + 20, 720, 30), label, F["row_title"], fill=PALETTE["ink"], align="left")
    draw_text_box(draw, (72, row_y + 54, 850, 28), subtitle, F["body"], fill=PALETTE["muted"], align="left")

    img_x = 80
    paste_rounded_image(img, INPUT_IMG, (img_x, row_y + 106, 152, 148), radius=14, border="#94A3B8")
    draw_text_box(draw, (img_x, row_y + 82, 152, 22), "Image input", F["small_bold"], fill=PALETTE["gray"])
    draw_prompt_box(img, 66, row_y + 262)

    enc_x = 356
    image_enc = (enc_x, row_y + 126, 214, 76)
    text_enc = (enc_x, row_y + 274, 214, 66)
    draw_encoder(img, image_enc, "Image\nEncoder", PALETTE["blue_fill"], "#60A5FA")
    draw_encoder(img, text_enc, "Text\nEncoder", PALETTE["amber_fill"], "#FBBF24")

    feat_x = 642
    red_cells = {(1, 5), (2, 4)} if adapter_kind == "direct" else set()
    green_cells = {(0, 1), (1, 2), (2, 2)} if adapter_kind == "wavelet" else set()
    draw_feature_grid(img, feat_x, row_y + 112, 238, 112, PALETTE["blue_soft"], red_cells=red_cells, green_cells=green_cells)
    proto_variant = {"fixed": "fixed", "direct": "drift", "wavelet": "stable"}[adapter_kind]
    draw_prototypes(img, feat_x + 8, row_y + 254, proto_variant)

    adapter_x = 968
    adapter_box = (adapter_x, row_y + 102, 384, 216)
    draw_adapter(img, adapter_box, adapter_kind)

    sim_x = 1436
    sim_variant = {"fixed": "fixed", "direct": "drift", "wavelet": "stable"}[adapter_kind]
    draw_similarity(img, sim_x, row_y + 126, sim_variant)
    draw_output(img, row_y, map_path, output_note, accent)

    draw_arrow(draw, (232, row_y + 180), (image_enc[0], row_y + 164), PALETTE["blue"], width=4)
    draw_arrow(draw, (318, row_y + 311), (text_enc[0], row_y + 307), PALETTE["amber"], width=4)
    draw_arrow(draw, (image_enc[0] + image_enc[2], row_y + 164), (feat_x, row_y + 168), PALETTE["blue"], width=4)
    draw_arrow(draw, (text_enc[0] + text_enc[2], row_y + 307), (feat_x + 8, row_y + 303), PALETTE["amber"], width=4)
    draw_arrow(draw, (feat_x + 238, row_y + 168), (adapter_x, row_y + 164), accent, width=4)
    draw_arrow(draw, (feat_x + 230, row_y + 303), (adapter_x, row_y + 268), accent, width=4)
    draw_arrow(draw, (adapter_x + 384, row_y + 207), (sim_x, row_y + 205), accent, width=4)
    draw_arrow(draw, (sim_x + 262, row_y + 205), (1724, row_y + 186), accent, width=4)
    if adapter_kind == "direct":
        draw_arrow(draw, (788, row_y + 206), (1050, row_y + 250), PALETTE["red"], width=3, dashed=True)
    elif adapter_kind == "wavelet":
        draw_arrow(draw, (792, row_y + 206), (1002, row_y + 186), PALETTE["green"], width=3, dashed=True)


def render_png() -> None:
    img = Image.new("RGBA", (W, H), hex_to_rgb(PALETTE["bg"]) + (255,))
    draw_header(img)
    rows = [
        (
            126,
            "(a) Conventional CLIP-ZSAD",
            "Frozen CLIP towers compare image patches with unchanged normal/abnormal text prototypes.",
            PALETTE["blue"],
            "fixed",
            MAP_FIXED,
            "No test-time correction. Prototype mismatch can leave diffuse or weak localization.",
        ),
        (
            520,
            "(b) Direct test-time prototype adaptation",
            "The test image itself updates prototypes, but abnormal patches may contaminate the update path.",
            PALETTE["red"],
            "direct",
            MAP_DIRECT,
            "Adaptation becomes input-sensitive; defect patches can pull the normal anchor and cause drift.",
        ),
        (
            914,
            "(c) Ours: wavelet-guided prototype adaptation",
            "Wavelet bands separate stable structure from high-frequency defect cues before prototype updates.",
            PALETTE["green"],
            "wavelet",
            MAP_FINAL,
            "LL preserves object semantics, while detail bands gate unreliable evidence for cleaner maps.",
        ),
    ]
    for row in rows:
        draw_row(img, *row)
    img.convert("RGB").save(PNG_OUT, quality=95)


class DrawioBuilder:
    def __init__(self) -> None:
        self.cells: list[str] = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
        self.idx = 2

    def _id(self) -> str:
        self.idx += 1
        return f"c{self.idx}"

    def rect(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        value: str = "",
        fill: str = "#FFFFFF",
        stroke: str = "#CBD5E1",
        radius: int = 12,
        font_size: int = 15,
        font_color: str = "#0F172A",
        font_style: int = 0,
        align: str = "center",
        valign: str = "middle",
        dashed: bool = False,
        stroke_width: int = 2,
    ) -> str:
        cid = self._id()
        dash = "dashed=1;" if dashed else ""
        html_value = html.escape(value).replace("\n", "&lt;br&gt;")
        style = (
            "rounded=1;whiteSpace=wrap;html=1;"
            f"arcSize={radius};fillColor={fill};strokeColor={stroke};strokeWidth={stroke_width};"
            f"fontSize={font_size};fontColor={font_color};fontStyle={font_style};align={align};verticalAlign={valign};"
            f"{dash}"
        )
        self.cells.append(
            f'<mxCell id="{cid}" value="{html_value}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )
        return cid

    def text(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        value: str,
        font_size: int = 15,
        color: str = "#0F172A",
        bold: bool = False,
        align: str = "left",
    ) -> str:
        cid = self._id()
        html_value = html.escape(value).replace("\n", "&lt;br&gt;")
        style = (
            "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;rounded=0;"
            f"align={align};verticalAlign=middle;fontSize={font_size};fontColor={color};fontStyle={1 if bold else 0};"
        )
        self.cells.append(
            f'<mxCell id="{cid}" value="{html_value}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )
        return cid

    def image(self, x: int, y: int, w: int, h: int, path: Path) -> str:
        cid = self._id()
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        style = (
            "shape=image;html=1;imageAspect=1;aspect=fixed;rounded=1;"
            f"image=data:image/png;base64,{payload};"
        )
        self.cells.append(
            f'<mxCell id="{cid}" value="" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )
        return cid

    def arrow(
        self,
        sx: int,
        sy: int,
        ex: int,
        ey: int,
        color: str = "#334155",
        width: int = 3,
        dashed: bool = False,
    ) -> str:
        cid = self._id()
        dash = "dashed=1;" if dashed else ""
        style = f"endArrow=block;html=1;rounded=1;strokeWidth={width};strokeColor={color};{dash}"
        self.cells.append(
            f'<mxCell id="{cid}" value="" style="{style}" edge="1" parent="1">'
            '<mxGeometry width="50" height="50" relative="1" as="geometry">'
            f'<mxPoint x="{sx}" y="{sy}" as="sourcePoint"/>'
            f'<mxPoint x="{ex}" y="{ey}" as="targetPoint"/>'
            "</mxGeometry></mxCell>"
        )
        return cid

    def to_xml(self) -> str:
        body = "\n".join(self.cells)
        return (
            '<mxfile host="app.diagrams.net" modified="2026-07-08T00:00:00.000Z" '
            'agent="Codex" version="24.7.17" type="device">\n'
            '  <diagram id="figure4-clip-arch-wavelet" name="Figure 4 CLIP arch wavelet">\n'
            f'    <mxGraphModel dx="{W}" dy="{H}" grid="1" gridSize="10" guides="1" tooltips="1" '
            f'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{W}" pageHeight="{H}" math="1" shadow="0">\n'
            "      <root>\n"
            f"{body}\n"
            "      </root>\n"
            "    </mxGraphModel>\n"
            "  </diagram>\n"
            "</mxfile>\n"
        )


def drawio_prompt_block(b: DrawioBuilder, x: int, y: int) -> None:
    b.rect(x, y, 252, 98, "", "#FFFFFF", "#CBD5E1", radius=12)
    b.text(x + 16, y + 7, 220, 18, "Text prompts", 13, PALETTE["gray"], True)
    b.rect(x + 14, y + 36, 224, 24, 'Normal: "normal cable"', PALETTE["blue_soft"], "#BFDBFE", radius=8, font_size=12, font_color=PALETTE["blue"], align="left")
    b.rect(x + 14, y + 66, 224, 24, 'Abnormal: "damaged cable"', PALETTE["red_soft"], "#FECACA", radius=8, font_size=12, font_color=PALETTE["red"], align="left")


def drawio_row(
    b: DrawioBuilder,
    row_y: int,
    label: str,
    subtitle: str,
    accent: str,
    kind: str,
    map_path: Path,
    note: str,
) -> None:
    b.rect(44, row_y, W - 88, 372, "", "#FFFFFF", "#D8E0EA", radius=12)
    b.rect(48, row_y + 74, 4, 256, "", accent, accent, radius=0, stroke_width=0)
    b.text(70, row_y + 20, 720, 30, label, 25, PALETTE["ink"], True)
    b.text(72, row_y + 54, 850, 28, subtitle, 17, PALETTE["muted"])
    b.text(80, row_y + 82, 152, 22, "Image input", 13, PALETTE["gray"], True, align="center")
    b.image(80, row_y + 106, 152, 148, INPUT_IMG)
    drawio_prompt_block(b, 66, row_y + 262)
    b.rect(356, row_y + 126, 214, 76, "Image\nEncoder", PALETTE["blue_fill"], "#60A5FA", radius=12, font_size=19, font_style=1)
    b.rect(496, row_y + 136, 58, 24, "frozen", "#FFFFFF", "#60A5FA", radius=10, font_size=11, font_color=PALETTE["subtle"])
    b.rect(356, row_y + 274, 214, 66, "Text\nEncoder", PALETTE["amber_fill"], "#FBBF24", radius=12, font_size=19, font_style=1)
    b.rect(496, row_y + 282, 58, 24, "frozen", "#FFFFFF", "#FBBF24", radius=10, font_size=11, font_color=PALETTE["subtle"])
    b.rect(642, row_y + 112, 238, 112, "Patch features", "#FFFFFF", "#CBD5E1", radius=12, font_size=14, font_style=1, valign="top")
    for rr in range(4):
        for cc in range(8):
            fill = PALETTE["blue_soft"]
            stroke = "#93A4B8"
            if kind == "direct" and (rr, cc) in {(1, 5), (2, 4)}:
                fill, stroke = PALETTE["red_fill"], PALETTE["red"]
            if kind == "wavelet" and (rr, cc) in {(0, 1), (1, 2), (2, 2)}:
                fill, stroke = PALETTE["green_fill"], PALETTE["green"]
            b.rect(658 + cc * 27, row_y + 146 + rr * 20, 19, 12, "", fill, stroke, radius=4, stroke_width=1)
    b.rect(650, row_y + 254, 222, 98, "Text prototypes", "#FFFFFF", "#CBD5E1", radius=12, font_size=14, font_style=1, valign="top")
    if kind == "direct":
        b.rect(676, row_y + 297, 44, 44, "old pN", "#FFFFFF", "#94A3B8", radius=20, font_size=10)
        b.rect(705, row_y + 292, 44, 44, "pN'", PALETTE["red_fill"], PALETTE["red"], radius=20, font_size=12, font_color=PALETTE["red"], font_style=1)
    else:
        b.rect(684, row_y + 296, 44, 44, "pN", PALETTE["blue_fill"], PALETTE["blue"], radius=20, font_size=12, font_color=PALETTE["blue"], font_style=1)
    b.rect(787, row_y + 296, 44, 44, "pA", PALETTE["red_fill"], PALETTE["red"], radius=20, font_size=12, font_color=PALETTE["red"], font_style=1)
    if kind == "fixed":
        b.rect(733, row_y + 326, 48, 22, "lock", "#FFFFFF", "#94A3B8", radius=12, font_size=10, font_color=PALETTE["subtle"])
    if kind == "wavelet":
        b.rect(729, row_y + 326, 54, 22, "gate", PALETTE["green_fill"], PALETTE["green"], radius=12, font_size=10, font_color=PALETTE["green"])

    adapter_x = 968
    if kind == "fixed":
        b.rect(adapter_x, row_y + 102, 384, 216, "Fixed CLIP matching", "#FFFFFF", "#94A3B8", radius=12, font_size=19, font_style=1, valign="top")
        b.text(adapter_x + 28, row_y + 154, 328, 52, "Text anchors are frozen; image patches are scored directly.", 16, PALETTE["muted"], align="center")
        b.rect(adapter_x + 72, row_y + 218, 116, 32, "no update", PALETTE["gray_fill"], "#94A3B8", radius=12, font_size=14, font_color=PALETTE["gray"], font_style=1)
        b.rect(adapter_x + 208, row_y + 218, 100, 32, "zero-shot", PALETTE["blue_fill"], "#93C5FD", radius=12, font_size=14, font_color=PALETTE["blue"], font_style=1)
    elif kind == "direct":
        b.rect(adapter_x, row_y + 102, 384, 216, "Direct prototype adaptation", "#FFFFFF", "#F97316", radius=12, font_size=19, font_style=1, valign="top")
        b.text(adapter_x + 26, row_y + 150, 332, 36, "Unfiltered test patches update the anchors.", 16, PALETTE["muted"], align="center")
        b.rect(adapter_x + 58, row_y + 206, 30, 28, "", PALETTE["blue_fill"], "#94A3B8", radius=6)
        b.rect(adapter_x + 106, row_y + 206, 30, 28, "", PALETTE["red_fill"], PALETTE["red"], radius=6)
        b.rect(adapter_x + 154, row_y + 206, 30, 28, "", PALETTE["blue_fill"], "#94A3B8", radius=6)
        b.rect(adapter_x + 202, row_y + 206, 30, 28, "", PALETTE["red_fill"], PALETTE["red"], radius=6)
        b.arrow(adapter_x + 112, row_y + 252, adapter_x + 250, row_y + 252, PALETTE["red"], 3)
        b.rect(adapter_x + 270, row_y + 229, 52, 52, "drift", PALETTE["red_fill"], PALETTE["red"], radius=20, font_size=11, font_color=PALETTE["red"], font_style=1)
        b.text(adapter_x + 50, row_y + 280, 284, 22, "abnormal evidence can pull prototypes", 13, PALETTE["red"], align="center")
    else:
        b.rect(adapter_x, row_y + 102, 384, 216, "Wavelet-guided prototype adapter", "#FFFFFF", PALETTE["green"], radius=12, font_size=19, font_style=1, valign="top")
        b.rect(adapter_x + 30, row_y + 158, 92, 58, "DWT", PALETTE["teal_fill"], PALETTE["teal"], radius=12, font_size=19, font_color=PALETTE["teal"], font_style=1)
        b.rect(adapter_x + 146, row_y + 154, 118, 30, "LL semantic", PALETTE["green_fill"], PALETTE["green"], radius=12, font_size=13, font_color=PALETTE["green"], font_style=1)
        b.rect(adapter_x + 146, row_y + 190, 150, 30, "LH/HL/HH detail", PALETTE["orange_fill"], PALETTE["orange"], radius=12, font_size=13, font_color=PALETTE["orange"], font_style=1)
        b.rect(adapter_x + 42, row_y + 246, 300, 50, "reliability gate selects stable normal evidence", PALETTE["green_soft"], "#86EFAC", radius=12, font_size=15, font_color=PALETTE["green"])

    b.rect(1436, row_y + 126, 262, 158, "Cosine similarity\n\nnormal     abnormal\n0.72        0.21\n0.64        0.33\n0.57        0.41\npatch-to-text logits", "#FFFFFF", "#CBD5E1", radius=12, font_size=13, font_color=PALETTE["gray"], font_style=1, valign="top")
    if kind == "direct":
        b.rect(1436, row_y + 126, 262, 158, "Cosine similarity\n\nnormal     abnormal\n0.56        0.42\n0.50        0.55\n0.44        0.68\npatch-to-text logits", "#FFFFFF", "#CBD5E1", radius=12, font_size=13, font_color=PALETTE["gray"], font_style=1, valign="top")
    if kind == "wavelet":
        b.rect(1436, row_y + 126, 262, 158, "Cosine similarity\n\nnormal     abnormal\n0.78        0.18\n0.73        0.24\n0.38        0.83\npatch-to-text logits", "#FFFFFF", "#CBD5E1", radius=12, font_size=13, font_color=PALETTE["gray"], font_style=1, valign="top")
    b.rect(1724, row_y + 42, 574, 244, "", "#FFFFFF", "#CBD5E1", radius=12)
    b.text(1738, row_y + 53, 176, 24, "Anomaly map", 14, PALETTE["gray"], True, align="center")
    b.image(1742, row_y + 92, 154, 154, map_path)
    b.rect(1936, row_y + 88, 156, 32, "output evidence", "#FFFFFF", accent, radius=12, font_size=14, font_color=accent, font_style=1)
    b.text(1934, row_y + 138, 330, 82, note, 16, PALETTE["muted"])

    b.arrow(232, row_y + 180, 356, row_y + 164, PALETTE["blue"], 3)
    b.arrow(318, row_y + 311, 356, row_y + 307, PALETTE["amber"], 3)
    b.arrow(570, row_y + 164, 642, row_y + 168, PALETTE["blue"], 3)
    b.arrow(570, row_y + 307, 650, row_y + 303, PALETTE["amber"], 3)
    b.arrow(880, row_y + 168, 968, row_y + 164, accent, 3)
    b.arrow(872, row_y + 303, 968, row_y + 268, accent, 3)
    b.arrow(1352, row_y + 207, 1436, row_y + 205, accent, 3)
    b.arrow(1698, row_y + 205, 1724, row_y + 186, accent, 3)
    if kind == "direct":
        b.arrow(788, row_y + 206, 1050, row_y + 250, PALETTE["red"], 2, dashed=True)
    if kind == "wavelet":
        b.arrow(792, row_y + 206, 1002, row_y + 186, PALETTE["green"], 2, dashed=True)


def write_drawio() -> None:
    b = DrawioBuilder()
    b.text(54, 28, 1460, 42, "CLIP-style comparison of prototype adaptation paradigms", 34, PALETTE["ink"], True)
    b.text(56, 70, 1540, 30, "Same MVTec cable image and same normal/abnormal prompts; highlighted paths show what changes at test time.", 19, PALETTE["muted"])
    b.rect(1668, 34, 656, 56, "", "#FFFFFF", "#CBD5E1", radius=12)
    b.rect(1688, 47, 118, 30, "image tower", PALETTE["blue_fill"], "#93C5FD", radius=12, font_size=12, font_color=PALETTE["blue"])
    b.rect(1822, 47, 112, 30, "text tower", PALETTE["amber_fill"], "#FCD34D", radius=12, font_size=12, font_color=PALETTE["amber"])
    b.rect(1950, 47, 128, 30, "drift risk", PALETTE["red_fill"], "#FCA5A5", radius=12, font_size=12, font_color=PALETTE["red"])
    b.rect(2094, 47, 154, 30, "wavelet gate", PALETTE["green_fill"], "#86EFAC", radius=12, font_size=12, font_color=PALETTE["green"])
    rows = [
        (
            126,
            "(a) Conventional CLIP-ZSAD",
            "Frozen CLIP towers compare image patches with unchanged normal/abnormal text prototypes.",
            PALETTE["blue"],
            "fixed",
            MAP_FIXED,
            "No test-time correction. Prototype mismatch can leave diffuse or weak localization.",
        ),
        (
            520,
            "(b) Direct test-time prototype adaptation",
            "The test image itself updates prototypes, but abnormal patches may contaminate the update path.",
            PALETTE["red"],
            "direct",
            MAP_DIRECT,
            "Adaptation becomes input-sensitive; defect patches can pull the normal anchor and cause drift.",
        ),
        (
            914,
            "(c) Ours: wavelet-guided prototype adaptation",
            "Wavelet bands separate stable structure from high-frequency defect cues before prototype updates.",
            PALETTE["green"],
            "wavelet",
            MAP_FINAL,
            "LL preserves object semantics, while detail bands gate unreliable evidence for cleaner maps.",
        ),
    ]
    for row in rows:
        drawio_row(b, *row)
    DRAWIO_OUT.write_text(b.to_xml(), encoding="utf-8")


def write_audit() -> None:
    AUDIT_OUT.write_text(
        """# Figure 4 CLIP Architecture Wavelet Audit

- Canvas: 2400 x 1360, three horizontal rows in CLIP dual-tower architecture style.
- Real imagery: each row uses the same real MVTec cable input image; right-side anomaly maps use local cable result assets.
- Text prompts: every row includes explicit normal and abnormal prompts.
- Row (a): frozen CLIP image/text encoders, fixed text prototypes, direct cosine similarity.
- Row (b): direct test-time prototype adaptation, with abnormal patch contamination and prototype drift highlighted in red.
- Row (c): wavelet-guided prototype adaptation, with DWT, LL semantic branch, LH/HL/HH detail branch, and reliability gate highlighted in green/orange.
- Draw.io editability: architecture boxes, text, arrows, prompts, patch tokens, prototypes, and similarity blocks are native Draw.io cells; only real input/output images are embedded rasters.
- Manual review point: the PNG preview is rendered by the paired PIL generator because diagrams.net CLI is not available in this environment.
""",
        encoding="utf-8",
    )


def main() -> None:
    for path in [INPUT_IMG, MAP_FIXED, MAP_DIRECT, MAP_FINAL]:
        if not path.exists():
            raise FileNotFoundError(path)
    render_png()
    write_drawio()
    write_audit()
    print(PNG_OUT)
    print(DRAWIO_OUT)
    print(AUDIT_OUT)


if __name__ == "__main__":
    main()
