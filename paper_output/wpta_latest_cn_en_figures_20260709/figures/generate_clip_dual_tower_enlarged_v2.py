from __future__ import annotations

import base64
import html
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


OUT_DIR = Path(__file__).resolve().parent
ASSET_DIR = OUT_DIR / "mvtec_three_row_selection_assets"

PNG_OUT = OUT_DIR / "figure4_clip_dual_tower_enlarged_v2.png"
DRAWIO_OUT = OUT_DIR / "figure4_clip_dual_tower_enlarged_v2.drawio"
AUDIT_OUT = OUT_DIR / "figure4_clip_dual_tower_enlarged_v2.audit.md"

INPUT_IMG = ASSET_DIR / "mvtec_cable_input.png"
MAP_FIXED = ASSET_DIR / "mvtec_cable_fixed.png"
MAP_DIRECT = ASSET_DIR / "mvtec_cable_direct.png"
MAP_FINAL = ASSET_DIR / "mvtec_cable_final.png"

W, H = 2600, 1580


COL = {
    "bg": "#F6F8FB",
    "paper": "#FFFFFF",
    "ink": "#111827",
    "muted": "#475569",
    "subtle": "#64748B",
    "line": "#CBD5E1",
    "line_dark": "#94A3B8",
    "image": "#2563EB",
    "image_soft": "#DBEAFE",
    "image_pale": "#EFF6FF",
    "text": "#D97706",
    "text_soft": "#FEF3C7",
    "text_pale": "#FFFBEB",
    "violet": "#7C3AED",
    "violet_soft": "#EDE9FE",
    "red": "#DC2626",
    "red_soft": "#FEE2E2",
    "red_pale": "#FEF2F2",
    "green": "#059669",
    "green_soft": "#D1FAE5",
    "green_pale": "#ECFDF5",
    "teal": "#0F766E",
    "teal_soft": "#CCFBF1",
    "orange": "#EA580C",
    "orange_soft": "#FFEDD5",
    "gray_soft": "#E5E7EB",
    "gray_pale": "#F8FAFC",
}


def rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    families = (
        [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        if bold
        else [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for f in families:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


F = {
    "title": font(38, True),
    "sub": font(20),
    "row": font(25, True),
    "body": font(16),
    "body_b": font(16, True),
    "module": font(18, True),
    "small": font(13),
    "small_b": font(13, True),
    "tiny": font(11),
    "tiny_b": font(11, True),
}


def tsize(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    if not text:
        return 0, 0
    b = draw.textbbox((0, 0), text, font=fnt)
    return b[2] - b[0], b[3] - b[1]


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont, width: int) -> list[str]:
    lines: list[str] = []
    for raw in text.split("\n"):
        words = raw.split(" ")
        cur = ""
        for word in words:
            cand = word if not cur else cur + " " + word
            if tsize(draw, cand, fnt)[0] <= width:
                cur = cand
            else:
                if cur:
                    lines.append(cur)
                cur = word
        lines.append(cur)
    return lines


def text_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: str = COL["ink"],
    align: str = "center",
    valign: str = "center",
    gap: int = 4,
) -> None:
    x, y, w, h = box
    lines = wrap(draw, text, fnt, max(12, w - 8))
    hs = [tsize(draw, line, fnt)[1] for line in lines]
    total = sum(hs) + gap * max(0, len(lines) - 1)
    if valign == "top":
        cy = y
    elif valign == "bottom":
        cy = y + h - total
    else:
        cy = y + (h - total) / 2
    for line, lh in zip(lines, hs):
        lw, _ = tsize(draw, line, fnt)
        if align == "left":
            tx = x
        elif align == "right":
            tx = x + w - lw
        else:
            tx = x + (w - lw) / 2
        draw.text((tx, cy), line, font=fnt, fill=fill)
        cy += lh + gap


def rr(
    img: Image.Image,
    box: tuple[int, int, int, int],
    fill: str,
    outline: str = COL["line"],
    radius: int = 12,
    width: int = 2,
    shadow: bool = False,
) -> None:
    draw = ImageDraw.Draw(img)
    x, y, w, h = box
    if shadow:
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        od.rounded_rectangle((x + 4, y + 7, x + w + 4, y + h + 7), radius=radius, fill=(15, 23, 42, 18))
        img.alpha_composite(ov)
    draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill, outline=outline, width=width)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str,
    width: int = 4,
    dashed: bool = False,
    head: int = 14,
) -> None:
    sx, sy = start
    ex, ey = end
    length = math.hypot(ex - sx, ey - sy)
    if length <= 1:
        return
    ux = (ex - sx) / length
    uy = (ey - sy) / length
    if dashed:
        dist = 0.0
        while dist < length - head:
            seg = min(dist + 15, length - head)
            draw.line((sx + ux * dist, sy + uy * dist, sx + ux * seg, sy + uy * seg), fill=color, width=width)
            dist += 25
    else:
        draw.line((sx, sy, ex, ey), fill=color, width=width)
    ang = math.atan2(ey - sy, ex - sx)
    p1 = (ex - head * math.cos(ang) + head * 0.55 * math.sin(ang), ey - head * math.sin(ang) - head * 0.55 * math.cos(ang))
    p2 = (ex - head * math.cos(ang) - head * 0.55 * math.sin(ang), ey - head * math.sin(ang) + head * 0.55 * math.cos(ang))
    draw.polygon([end, p1, p2], fill=color)


def pill(img: Image.Image, box: tuple[int, int, int, int], text: str, fill: str, stroke: str, color: str) -> None:
    rr(img, box, fill, stroke, radius=max(8, box[3] // 2), width=2)
    text_box(ImageDraw.Draw(img), box, text, F["tiny_b"], color)


def paste_img(img: Image.Image, path: Path, box: tuple[int, int, int, int], radius: int = 12, border: str = COL["line_dark"]) -> None:
    x, y, w, h = box
    src = Image.open(path).convert("RGB")
    fitted = ImageOps.fit(src, (w, h), Image.Resampling.LANCZOS)
    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle((0, 0, w, h), radius=radius, fill=255)
    img.paste(fitted, (x, y), mask)
    ImageDraw.Draw(img).rounded_rectangle((x, y, x + w, y + h), radius=radius, outline=border, width=2)


def encoder_stack(img: Image.Image, box: tuple[int, int, int, int], label: str, fill: str, stroke: str) -> None:
    rr(img, box, fill, stroke, radius=14, width=2, shadow=True)
    x, y, w, h = box
    draw = ImageDraw.Draw(img)
    for i in range(6):
        xx = x + 16 + i * 10
        draw.line((xx, y + 14, xx, y + h - 14), fill=stroke, width=2)
    text_box(draw, (x + 70, y + 10, w - 110, h - 20), label, F["module"], COL["ink"])
    pill(img, (x + w - 78, y + 10, 62, 24), "frozen", "#FFFFFF", stroke, COL["subtle"])


def token_block(img: Image.Image, x: int, y: int, mode: str) -> None:
    draw = ImageDraw.Draw(img)
    rr(img, (x, y, 238, 86), "#FFFFFF", COL["line"], radius=12, width=2)
    text_box(draw, (x + 14, y + 8, 210, 16), "Patch embeddings {x_i}", F["small_b"], COL["muted"], align="left")
    bad = {(1, 5), (2, 4), (2, 5), (1, 6)}
    stable = {(0, 1), (1, 2), (2, 2), (0, 3)}
    for r in range(3):
        for c in range(8):
            fill, stroke = COL["image_pale"], "#93C5FD"
            if mode == "direct" and (r, c) in bad:
                fill, stroke = COL["red_soft"], COL["red"]
            if mode == "wavelet" and (r, c) in stable:
                fill, stroke = COL["green_soft"], COL["green"]
            draw.rounded_rectangle((x + 16 + c * 26, y + 34 + r * 15, x + 35 + c * 26, y + 45 + r * 15), radius=4, fill=fill, outline=stroke, width=1)
    if mode == "direct":
        text_box(draw, (x + 18, y + 68, 202, 14), "includes defect tokens", F["tiny_b"], COL["red"])
    elif mode == "wavelet":
        text_box(draw, (x + 18, y + 68, 202, 14), "reliable candidates", F["tiny_b"], COL["green"])
    else:
        text_box(draw, (x + 18, y + 68, 202, 14), "no adaptation source", F["tiny"], COL["subtle"])


def prompt_block(img: Image.Image, x: int, y: int) -> None:
    draw = ImageDraw.Draw(img)
    rr(img, (x, y, 238, 86), "#FFFFFF", COL["line"], radius=12, width=2)
    text_box(draw, (x + 14, y + 8, 210, 16), "Text prompts", F["small_b"], COL["muted"], align="left")
    rr(img, (x + 16, y + 34, 206, 21), COL["image_pale"], "#BFDBFE", radius=7, width=1)
    rr(img, (x + 16, y + 59, 206, 21), COL["red_pale"], "#FECACA", radius=7, width=1)
    draw.text((x + 25, y + 38), 'Normal: "normal cable"', font=F["tiny_b"], fill=COL["image"])
    draw.text((x + 25, y + 63), 'Abnormal: "damaged cable"', font=F["tiny_b"], fill=COL["red"])


def prototypes(img: Image.Image, x: int, y: int, mode: str) -> None:
    draw = ImageDraw.Draw(img)
    rr(img, (x, y, 238, 86), "#FFFFFF", COL["line"], radius=12, width=2)
    text_box(draw, (x + 14, y + 8, 210, 16), "Text prototypes", F["small_b"], COL["muted"], align="left")
    if mode == "direct":
        draw.ellipse((x + 36, y + 35, x + 78, y + 77), fill="#FFFFFF", outline=COL["line_dark"], width=2)
        text_box(draw, (x + 32, y + 48, 50, 14), "p_N", F["tiny"], COL["subtle"])
        draw.ellipse((x + 67, y + 31, x + 113, y + 77), fill=COL["red_soft"], outline=COL["red"], width=3)
        text_box(draw, (x + 67, y + 44, 46, 17), "p_N'", F["tiny_b"], COL["red"])
    else:
        draw.ellipse((x + 51, y + 31, x + 97, y + 77), fill=COL["image_soft"], outline=COL["image"], width=3)
        text_box(draw, (x + 51, y + 44, 46, 17), "p_N", F["tiny_b"], COL["image"])
    draw.ellipse((x + 153, y + 31, x + 199, y + 77), fill=COL["red_soft"], outline=COL["red"], width=3)
    text_box(draw, (x + 153, y + 44, 46, 17), "p_A", F["tiny_b"], COL["red"])
    if mode == "fixed":
        pill(img, (x + 101, y + 54, 50, 22), "lock", "#FFFFFF", COL["line_dark"], COL["subtle"])
    if mode == "wavelet":
        pill(img, (x + 100, y + 54, 52, 22), "gate", COL["green_soft"], COL["green"], COL["green"])


def image_method_slot(img: Image.Image, x: int, y: int, mode: str) -> None:
    draw = ImageDraw.Draw(img)
    if mode == "fixed":
        rr(img, (x, y, 300, 86), "#FFFFFF", COL["line_dark"], radius=12, width=2)
        text_box(draw, (x + 16, y + 10, 268, 18), "No update branch", F["small_b"], COL["muted"])
        draw.line((x + 88, y + 54, x + 212, y + 54), fill=COL["line_dark"], width=3)
        draw.line((x + 140, y + 38, x + 160, y + 70), fill=COL["red"], width=4)
        draw.line((x + 160, y + 38, x + 140, y + 70), fill=COL["red"], width=4)
        text_box(draw, (x + 44, y + 66, 212, 15), "zero-shot matching only", F["tiny"], COL["subtle"])
    elif mode == "direct":
        rr(img, (x, y, 300, 86), COL["red_pale"], COL["red"], radius=12, width=2)
        text_box(draw, (x + 16, y + 10, 268, 18), "Raw test-token update", F["small_b"], COL["red"])
        for i, c in enumerate([COL["image_soft"], COL["red_soft"], COL["image_soft"], COL["red_soft"], COL["gray_soft"]]):
            draw.rounded_rectangle((x + 44 + i * 39, y + 38, x + 70 + i * 39, y + 63), radius=6, fill=c, outline=COL["line_dark"], width=1)
        text_box(draw, (x + 34, y + 66, 232, 15), "abnormal patches are not filtered", F["tiny_b"], COL["red"])
    else:
        rr(img, (x, y, 300, 86), "#FFFFFF", COL["green"], radius=12, width=2)
        text_box(draw, (x + 16, y + 8, 268, 16), "Wavelet adapter", F["small_b"], COL["green"])
        rr(img, (x + 18, y + 34, 58, 34), COL["teal_soft"], COL["teal"], radius=9, width=2)
        text_box(draw, (x + 18, y + 42, 58, 13), "DWT", F["tiny_b"], COL["teal"])
        pill(img, (x + 96, y + 31, 88, 23), "LL pass", COL["green_soft"], COL["green"], COL["green"])
        pill(img, (x + 96, y + 58, 108, 23), "HF suppress", COL["orange_soft"], COL["orange"], COL["orange"])
        rr(img, (x + 218, y + 38, 62, 30), COL["green_pale"], COL["green"], radius=9, width=2)
        text_box(draw, (x + 218, y + 45, 62, 13), "gate", F["tiny_b"], COL["green"])


def text_update_slot(img: Image.Image, x: int, y: int, mode: str) -> None:
    draw = ImageDraw.Draw(img)
    if mode == "fixed":
        rr(img, (x, y, 300, 86), COL["gray_pale"], COL["line_dark"], radius=12, width=2)
        text_box(draw, (x + 20, y + 15, 260, 22), "p_N and p_A stay fixed", F["small_b"], COL["muted"])
        text_box(draw, (x + 28, y + 47, 244, 24), "The text tower only supplies fixed anchors.", F["tiny"], COL["subtle"])
    elif mode == "direct":
        rr(img, (x, y, 300, 86), COL["red_pale"], COL["red"], radius=12, width=2)
        text_box(draw, (x + 18, y + 12, 264, 18), "Prototype drift inside text tower", F["small_b"], COL["red"])
        draw.ellipse((x + 52, y + 42, x + 88, y + 78), fill="#FFFFFF", outline=COL["line_dark"], width=2)
        text_box(draw, (x + 52, y + 51, 36, 14), "p_N", F["tiny"], COL["subtle"])
        arrow(draw, (x + 93, y + 60), (x + 157, y + 60), COL["red"], width=3, head=11)
        draw.ellipse((x + 164, y + 35, x + 216, y + 87), fill=COL["red_soft"], outline=COL["red"], width=3)
        text_box(draw, (x + 164, y + 52, 52, 15), "p_N'", F["tiny_b"], COL["red"])
    else:
        rr(img, (x, y, 300, 86), COL["green_pale"], COL["green"], radius=12, width=2)
        text_box(draw, (x + 16, y + 12, 268, 18), "Gated update to normal prototype", F["small_b"], COL["green"])
        text_box(draw, (x + 28, y + 42, 244, 16), "p_N <- reliable LL semantics", F["tiny_b"], COL["green"])
        text_box(draw, (x + 28, y + 62, 244, 14), "p_A remains the abnormal anchor", F["tiny"], COL["subtle"])


def similarity_block(img: Image.Image, x: int, y: int, mode: str) -> None:
    draw = ImageDraw.Draw(img)
    rr(img, (x, y, 282, 210), "#FFFFFF", COL["violet"], radius=14, width=2, shadow=True)
    text_box(draw, (x + 18, y + 12, 246, 20), "CLIP patch-text similarity", F["small_b"], COL["violet"])
    text_box(draw, (x + 80, y + 42, 64, 16), "normal", F["tiny_b"], COL["image"])
    text_box(draw, (x + 154, y + 42, 76, 16), "abnormal", F["tiny_b"], COL["red"])
    vals = {
        "fixed": [(0.72, 0.21), (0.64, 0.33), (0.57, 0.41), (0.51, 0.48)],
        "direct": [(0.56, 0.42), (0.50, 0.55), (0.44, 0.68), (0.47, 0.61)],
        "wavelet": [(0.78, 0.18), (0.73, 0.24), (0.38, 0.83), (0.70, 0.29)],
    }[mode]
    for r, pair in enumerate(vals):
        text_box(draw, (x + 28, y + 68 + r * 27, 32, 15), f"x{r+1}", F["tiny"], COL["subtle"])
        for c, val in enumerate(pair):
            strong = val > 0.62
            fill = "#93C5FD" if c == 0 and strong else COL["image_pale"] if c == 0 else "#F87171" if strong else COL["red_pale"]
            stroke = "#60A5FA" if c == 0 else "#FCA5A5"
            xx = x + 82 + c * 78
            yy = y + 66 + r * 27
            draw.rounded_rectangle((xx, yy, xx + 60, yy + 20), radius=4, fill=fill, outline=stroke, width=1)
            text_box(draw, (xx, yy + 2, 60, 15), f"{val:.2f}", F["tiny"], COL["ink"])
    text_box(draw, (x + 20, y + 184, 242, 16), "score = sim(x_i,p_A) - sim(x_i,p_N)", F["tiny"], COL["subtle"])


def output_panel(img: Image.Image, row_y: int, path: Path, note: str, accent: str) -> None:
    draw = ImageDraw.Draw(img)
    x, y = 2248, row_y + 132
    rr(img, (x - 20, y - 42, 294, 280), "#FFFFFF", COL["line"], radius=14, width=2, shadow=True)
    text_box(draw, (x, y - 30, 150, 18), "Anomaly map", F["small_b"], COL["muted"])
    paste_img(img, path, (x, y, 158, 158), radius=12, border=accent)
    text_box(draw, (x + 174, y + 2, 78, 18), "Output", F["small_b"], accent)
    text_box(draw, (x + 174, y + 34, 82, 118), note, F["tiny"], COL["muted"], align="left")
    rr(img, (x + 14, y + 181, 226, 34), COL["gray_pale"], COL["line"], radius=9, width=1)
    text_box(draw, (x + 22, y + 189, 210, 14), "same input and prompts", F["tiny"], COL["subtle"])


def dual_tower(img: Image.Image, row_y: int, mode: str, accent: str) -> None:
    draw = ImageDraw.Draw(img)
    x0, y0, w, h = 310, row_y + 92, 1500, 304
    rr(img, (x0, y0, w, h), "#FFFFFF", COL["line"], radius=18, width=2, shadow=True)
    text_box(draw, (x0 + 18, y0 + 10, 318, 20), "Enlarged CLIP dual tower", F["small_b"], COL["ink"], align="left")
    pill(img, (x0 + 1248, y0 + 9, 104, 24), "image tower", COL["image_soft"], "#93C5FD", COL["image"])
    pill(img, (x0 + 1362, y0 + 9, 90, 24), "text tower", COL["text_soft"], "#FCD34D", COL["text"])

    img_lane = (x0 + 24, y0 + 42, w - 48, 112)
    txt_lane = (x0 + 24, y0 + 174, w - 48, 112)
    rr(img, img_lane, COL["image_pale"], "#BFDBFE", radius=14, width=2)
    rr(img, txt_lane, COL["text_pale"], "#FCD34D", radius=14, width=2)
    text_box(draw, (img_lane[0] + 12, img_lane[1] + 10, 96, 18), "IMAGE", F["small_b"], COL["image"])
    text_box(draw, (img_lane[0] + 12, img_lane[1] + 30, 96, 18), "TOWER", F["small_b"], COL["image"])
    text_box(draw, (txt_lane[0] + 12, txt_lane[1] + 10, 96, 18), "TEXT", F["small_b"], COL["text"])
    text_box(draw, (txt_lane[0] + 12, txt_lane[1] + 30, 96, 18), "TOWER", F["small_b"], COL["text"])

    x_prompt = x0 + 126
    x_enc = x0 + 394
    x_feat = x0 + 700
    x_slot = x0 + 974
    y_img = img_lane[1] + 13
    y_txt = txt_lane[1] + 13

    rr(img, (x_prompt, y_img, 238, 86), "#FFFFFF", COL["line"], radius=12, width=2)
    text_box(draw, (x_prompt + 14, y_img + 8, 210, 16), "Image patches", F["small_b"], COL["muted"], align="left")
    for r in range(3):
        for c in range(5):
            draw.rounded_rectangle((x_prompt + 28 + c * 37, y_img + 34 + r * 15, x_prompt + 53 + c * 37, y_img + 45 + r * 15), radius=4, fill=COL["image_soft"], outline="#93C5FD", width=1)
    text_box(draw, (x_prompt + 20, y_img + 68, 198, 14), "from the MVTec cable image", F["tiny"], COL["subtle"])

    encoder_stack(img, (x_enc, y_img, 260, 86), "Image Encoder", COL["image_soft"], "#60A5FA")
    token_block(img, x_feat, y_img, mode)
    image_method_slot(img, x_slot, y_img, mode)

    prompt_block(img, x_prompt, y_txt)
    encoder_stack(img, (x_enc, y_txt, 260, 86), "Text Encoder", COL["text_soft"], "#FBBF24")
    prototypes(img, x_feat, y_txt, mode)
    text_update_slot(img, x_slot, y_txt, mode)

    arrow(draw, (x_prompt + 238, y_img + 43), (x_enc, y_img + 43), COL["image"], width=4)
    arrow(draw, (x_enc + 260, y_img + 43), (x_feat, y_img + 43), COL["image"], width=4)
    arrow(draw, (x_feat + 238, y_img + 43), (x_slot, y_img + 43), accent, width=4)
    arrow(draw, (x_prompt + 238, y_txt + 43), (x_enc, y_txt + 43), COL["text"], width=4)
    arrow(draw, (x_enc + 260, y_txt + 43), (x_feat, y_txt + 43), COL["text"], width=4)
    arrow(draw, (x_feat + 238, y_txt + 43), (x_slot, y_txt + 43), accent, width=4)

    if mode == "direct":
        arrow(draw, (x_feat + 160, y_img + 75), (x_feat + 110, y_txt + 36), COL["red"], width=4, dashed=True, head=13)
        text_box(draw, (x_feat + 170, y0 + 152, 206, 18), "unfiltered cross-tower update", F["tiny_b"], COL["red"])
    elif mode == "wavelet":
        arrow(draw, (x_slot + 244, y_img + 58), (x_slot + 245, y_txt + 30), COL["green"], width=4, dashed=True, head=13)
        text_box(draw, (x_slot + 24, y0 + 152, 252, 18), "wavelet-gated cross-tower update", F["tiny_b"], COL["green"])
    else:
        draw.line((x0 + 834, y0 + 159, x0 + 880, y0 + 159), fill=COL["line_dark"], width=3)
        draw.line((x0 + 853, y0 + 144, x0 + 867, y0 + 174), fill=COL["red"], width=4)
        draw.line((x0 + 867, y0 + 144, x0 + 853, y0 + 174), fill=COL["red"], width=4)
        text_box(draw, (x0 + 782, y0 + 150, 180, 18), "no adaptation bridge", F["tiny_b"], COL["subtle"])


def row(
    img: Image.Image,
    row_y: int,
    title: str,
    subtitle: str,
    mode: str,
    accent: str,
    map_path: Path,
    note: str,
) -> None:
    draw = ImageDraw.Draw(img)
    rr(img, (48, row_y, W - 96, 438), COL["paper"], "#D9E2EC", radius=22, width=2, shadow=True)
    draw.rectangle((54, row_y + 86, 60, row_y + 386), fill=accent)
    text_box(draw, (78, row_y + 19, 760, 30), title, F["row"], COL["ink"], align="left")
    text_box(draw, (80, row_y + 53, 1110, 24), subtitle, F["body"], COL["muted"], align="left")

    text_box(draw, (86, row_y + 104, 160, 18), "Image input", F["small_b"], COL["muted"])
    paste_img(img, INPUT_IMG, (86, row_y + 132, 160, 148), radius=12, border=COL["line_dark"])
    rr(img, (78, row_y + 302, 176, 74), "#FFFFFF", COL["line"], radius=12, width=2)
    text_box(draw, (92, row_y + 314, 148, 16), "Prompt pair", F["small_b"], COL["muted"])
    pill(img, (94, row_y + 339, 132, 22), "normal cable", COL["image_pale"], "#BFDBFE", COL["image"])
    pill(img, (94, row_y + 364, 132, 22), "damaged cable", COL["red_pale"], "#FECACA", COL["red"])

    dual_tower(img, row_y, mode, accent)
    similarity_block(img, 1842, row_y + 136, mode)
    output_panel(img, row_y, map_path, note, accent)

    arrow(draw, (246, row_y + 206), (310, row_y + 190), COL["image"], width=4)
    arrow(draw, (254, row_y + 342), (310, row_y + 313), COL["text"], width=4)
    arrow(draw, (1810, row_y + 190), (1842, row_y + 207), COL["image"], width=4)
    arrow(draw, (1810, row_y + 324), (1842, row_y + 275), COL["text"], width=4)
    arrow(draw, (2124, row_y + 241), (2228, row_y + 238), accent, width=4)


def header(img: Image.Image) -> None:
    draw = ImageDraw.Draw(img)
    text_box(draw, (60, 26, 1440, 46), "Enlarged CLIP dual-tower comparison for zero-shot anomaly detection", F["title"], COL["ink"], align="left")
    text_box(
        draw,
        (62, 72, 1540, 28),
        "Each row keeps the same image and normal/abnormal text prompts; the difference is placed inside the dual-tower adaptation path.",
        F["sub"],
        COL["muted"],
        align="left",
    )
    rr(img, (1656, 36, 860, 58), "#FFFFFF", COL["line"], radius=16, width=2, shadow=True)
    pill(img, (1680, 50, 128, 30), "image tower", COL["image_soft"], "#93C5FD", COL["image"])
    pill(img, (1830, 50, 116, 30), "text tower", COL["text_soft"], "#FCD34D", COL["text"])
    pill(img, (1968, 50, 118, 30), "CLIP sim", COL["violet_soft"], "#C4B5FD", COL["violet"])
    pill(img, (2108, 50, 130, 30), "drift path", COL["red_soft"], "#FCA5A5", COL["red"])
    pill(img, (2260, 50, 150, 30), "wavelet gate", COL["green_soft"], "#86EFAC", COL["green"])
    pill(img, (2426, 50, 64, 30), "v2", "#FFFFFF", COL["line_dark"], COL["subtle"])


def render_png() -> None:
    img = Image.new("RGBA", (W, H), rgb(COL["bg"]) + (255,))
    header(img)
    rows = [
        (
            120,
            "(a) Conventional CLIP-ZSAD",
            "Two frozen CLIP towers only perform patch-text matching with fixed normal and abnormal text prototypes.",
            "fixed",
            COL["image"],
            MAP_FIXED,
            "Fixed anchors give weak or diffuse localization when the test domain shifts.",
        ),
        (
            586,
            "(b) Direct test-time prototype adaptation",
            "Visual test tokens are fed back to the text-prototype side without filtering, so defect tokens can move p_N.",
            "direct",
            COL["red"],
            MAP_DIRECT,
            "The normal prototype drifts, producing unstable heatmaps.",
        ),
        (
            1052,
            "(c) Ours: wavelet-guided prototype adaptation",
            "A wavelet adapter gates unreliable high-frequency evidence before reliable semantics update the normal prototype.",
            "wavelet",
            COL["green"],
            MAP_FINAL,
            "p_N adapts from reliable semantics while defect cues are suppressed.",
        ),
    ]
    for args in rows:
        row(img, *args)
    img.convert("RGB").save(PNG_OUT, quality=95)


class Drawio:
    def __init__(self) -> None:
        self.cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
        self.i = 1

    def _id(self) -> str:
        self.i += 1
        return f"c{self.i}"

    def rect(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        value: str = "",
        fill: str = "#FFFFFF",
        stroke: str = "#CBD5E1",
        font_size: int = 13,
        font_color: str = "#111827",
        bold: bool = False,
        radius: int = 12,
        valign: str = "middle",
        align: str = "center",
        dashed: bool = False,
        stroke_width: int = 2,
    ) -> None:
        cid = self._id()
        v = html.escape(value).replace("\n", "&lt;br&gt;")
        dash = "dashed=1;" if dashed else ""
        style = (
            "rounded=1;whiteSpace=wrap;html=1;"
            f"arcSize={radius};fillColor={fill};strokeColor={stroke};strokeWidth={stroke_width};"
            f"fontSize={font_size};fontColor={font_color};fontStyle={1 if bold else 0};"
            f"align={align};verticalAlign={valign};{dash}"
        )
        self.cells.append(
            f'<mxCell id="{cid}" value="{v}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )

    def ellipse(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        value: str,
        fill: str,
        stroke: str,
        font_size: int = 11,
        font_color: str = "#111827",
        bold: bool = False,
        stroke_width: int = 2,
    ) -> None:
        cid = self._id()
        v = html.escape(value).replace("\n", "&lt;br&gt;")
        style = (
            "ellipse;whiteSpace=wrap;html=1;"
            f"fillColor={fill};strokeColor={stroke};strokeWidth={stroke_width};"
            f"fontSize={font_size};fontColor={font_color};fontStyle={1 if bold else 0};"
            "align=center;verticalAlign=middle;"
        )
        self.cells.append(
            f'<mxCell id="{cid}" value="{v}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )

    def text(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        value: str,
        font_size: int = 13,
        color: str = "#111827",
        bold: bool = False,
        align: str = "left",
    ) -> None:
        cid = self._id()
        v = html.escape(value).replace("\n", "&lt;br&gt;")
        style = (
            "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;rounded=0;"
            f"align={align};verticalAlign=middle;fontSize={font_size};fontColor={color};fontStyle={1 if bold else 0};"
        )
        self.cells.append(
            f'<mxCell id="{cid}" value="{v}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )

    def image(self, x: int, y: int, w: int, h: int, path: Path) -> None:
        cid = self._id()
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        style = "shape=image;html=1;imageAspect=1;aspect=fixed;rounded=1;image=data:image/png;base64," + payload + ";"
        self.cells.append(
            f'<mxCell id="{cid}" value="" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )

    def arrow(self, sx: int, sy: int, ex: int, ey: int, color: str, width: int = 3, dashed: bool = False, end: bool = True) -> None:
        cid = self._id()
        dash = "dashed=1;" if dashed else ""
        head = "endArrow=block;" if end else "endArrow=none;"
        style = f"{head}html=1;rounded=1;strokeWidth={width};strokeColor={color};{dash}"
        self.cells.append(
            f'<mxCell id="{cid}" value="" style="{style}" edge="1" parent="1">'
            '<mxGeometry width="50" height="50" relative="1" as="geometry">'
            f'<mxPoint x="{sx}" y="{sy}" as="sourcePoint"/>'
            f'<mxPoint x="{ex}" y="{ey}" as="targetPoint"/>'
            "</mxGeometry></mxCell>"
        )

    def xml(self) -> str:
        return (
            '<mxfile host="app.diagrams.net" modified="2026-07-08T00:00:00.000Z" agent="Codex" version="24.7.17" type="device">\n'
            '  <diagram id="clip-dual-tower-enlarged-v2" name="CLIP dual tower enlarged v2">\n'
            f'    <mxGraphModel dx="{W}" dy="{H}" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{W}" pageHeight="{H}" math="0" shadow="0">\n'
            "      <root>\n"
            + "\n".join(self.cells)
            + "\n      </root>\n"
            "    </mxGraphModel>\n"
            "  </diagram>\n"
            "</mxfile>\n"
        )


def d_pill(d: Drawio, x: int, y: int, w: int, text: str, fill: str, stroke: str, color: str) -> None:
    d.rect(x, y, w, 26, text, fill, stroke, font_size=11, font_color=color, bold=True, radius=18)


def d_encoder(d: Drawio, x: int, y: int, w: int, h: int, text: str, fill: str, stroke: str) -> None:
    d.rect(x, y, w, h, text, fill, stroke, font_size=18, bold=True, radius=12)
    for i in range(6):
        d.rect(x + 16 + i * 10, y + 14, 2, h - 28, "", stroke, stroke, stroke_width=0)
    d_pill(d, x + w - 78, y + 10, 62, "frozen", "#FFFFFF", stroke, COL["subtle"])


def d_token_block(d: Drawio, x: int, y: int, mode: str) -> None:
    d.rect(x, y, 238, 86, "Patch embeddings {x_i}", "#FFFFFF", COL["line"], font_size=13, font_color=COL["muted"], bold=True, radius=12, valign="top")
    bad = {(1, 5), (2, 4), (2, 5), (1, 6)}
    stable = {(0, 1), (1, 2), (2, 2), (0, 3)}
    for r in range(3):
        for c in range(8):
            fill, stroke = COL["image_pale"], "#93C5FD"
            if mode == "direct" and (r, c) in bad:
                fill, stroke = COL["red_soft"], COL["red"]
            if mode == "wavelet" and (r, c) in stable:
                fill, stroke = COL["green_soft"], COL["green"]
            d.rect(x + 16 + c * 26, y + 34 + r * 15, 19, 11, "", fill, stroke, radius=4, stroke_width=1)
    label = "includes defect tokens" if mode == "direct" else "reliable candidates" if mode == "wavelet" else "no adaptation source"
    color = COL["red"] if mode == "direct" else COL["green"] if mode == "wavelet" else COL["subtle"]
    d.text(x + 18, y + 68, 202, 14, label, 11, color, mode != "fixed", align="center")


def d_prompt_block(d: Drawio, x: int, y: int) -> None:
    d.rect(x, y, 238, 86, "Text prompts", "#FFFFFF", COL["line"], font_size=13, font_color=COL["muted"], bold=True, radius=12, valign="top")
    d.rect(x + 16, y + 34, 206, 21, 'Normal: "normal cable"', COL["image_pale"], "#BFDBFE", font_size=11, font_color=COL["image"], bold=True, radius=7, stroke_width=1)
    d.rect(x + 16, y + 59, 206, 21, 'Abnormal: "damaged cable"', COL["red_pale"], "#FECACA", font_size=11, font_color=COL["red"], bold=True, radius=7, stroke_width=1)


def d_prototypes(d: Drawio, x: int, y: int, mode: str) -> None:
    d.rect(x, y, 238, 86, "Text prototypes", "#FFFFFF", COL["line"], font_size=13, font_color=COL["muted"], bold=True, radius=12, valign="top")
    if mode == "direct":
        d.ellipse(x + 36, y + 35, 42, 42, "p_N", "#FFFFFF", COL["line_dark"], 10, COL["subtle"])
        d.ellipse(x + 67, y + 31, 46, 46, "p_N'", COL["red_soft"], COL["red"], 12, COL["red"], True, 3)
    else:
        d.ellipse(x + 51, y + 31, 46, 46, "p_N", COL["image_soft"], COL["image"], 12, COL["image"], True, 3)
    d.ellipse(x + 153, y + 31, 46, 46, "p_A", COL["red_soft"], COL["red"], 12, COL["red"], True, 3)
    if mode == "fixed":
        d_pill(d, x + 101, y + 54, 50, "lock", "#FFFFFF", COL["line_dark"], COL["subtle"])
    if mode == "wavelet":
        d_pill(d, x + 100, y + 54, 52, "gate", COL["green_soft"], COL["green"], COL["green"])


def d_image_slot(d: Drawio, x: int, y: int, mode: str) -> None:
    if mode == "fixed":
        d.rect(x, y, 300, 86, "No update branch", "#FFFFFF", COL["line_dark"], font_size=13, font_color=COL["muted"], bold=True, radius=12, valign="top")
        d.arrow(x + 88, y + 54, x + 212, y + 54, COL["line_dark"], 3, end=False)
        d.arrow(x + 140, y + 38, x + 160, y + 70, COL["red"], 4, end=False)
        d.arrow(x + 160, y + 38, x + 140, y + 70, COL["red"], 4, end=False)
        d.text(x + 44, y + 66, 212, 15, "zero-shot matching only", 11, COL["subtle"], align="center")
    elif mode == "direct":
        d.rect(x, y, 300, 86, "Raw test-token update", COL["red_pale"], COL["red"], font_size=13, font_color=COL["red"], bold=True, radius=12, valign="top")
        for i, c in enumerate([COL["image_soft"], COL["red_soft"], COL["image_soft"], COL["red_soft"], COL["gray_soft"]]):
            d.rect(x + 44 + i * 39, y + 38, 26, 25, "", c, COL["line_dark"], radius=6, stroke_width=1)
        d.text(x + 34, y + 66, 232, 15, "abnormal patches are not filtered", 11, COL["red"], True, align="center")
    else:
        d.rect(x, y, 300, 86, "Wavelet adapter", "#FFFFFF", COL["green"], font_size=13, font_color=COL["green"], bold=True, radius=12, valign="top")
        d.rect(x + 18, y + 34, 58, 34, "DWT", COL["teal_soft"], COL["teal"], font_size=11, font_color=COL["teal"], bold=True, radius=9)
        d_pill(d, x + 96, y + 31, 88, "LL pass", COL["green_soft"], COL["green"], COL["green"])
        d_pill(d, x + 96, y + 58, 108, "HF suppress", COL["orange_soft"], COL["orange"], COL["orange"])
        d.rect(x + 218, y + 38, 62, 30, "gate", COL["green_pale"], COL["green"], font_size=11, font_color=COL["green"], bold=True, radius=9)


def d_text_slot(d: Drawio, x: int, y: int, mode: str) -> None:
    if mode == "fixed":
        d.rect(x, y, 300, 86, "p_N and p_A stay fixed\nThe text tower only supplies fixed anchors.", COL["gray_pale"], COL["line_dark"], font_size=13, font_color=COL["muted"], bold=True, radius=12)
    elif mode == "direct":
        d.rect(x, y, 300, 86, "Prototype drift inside text tower", COL["red_pale"], COL["red"], font_size=13, font_color=COL["red"], bold=True, radius=12, valign="top")
        d.ellipse(x + 52, y + 42, 36, 36, "p_N", "#FFFFFF", COL["line_dark"], 10, COL["subtle"])
        d.arrow(x + 93, y + 60, x + 157, y + 60, COL["red"], 3)
        d.ellipse(x + 164, y + 35, 52, 52, "p_N'", COL["red_soft"], COL["red"], 11, COL["red"], True, 3)
    else:
        d.rect(x, y, 300, 86, "Gated update to normal prototype\np_N <- reliable LL semantics\np_A remains the abnormal anchor", COL["green_pale"], COL["green"], font_size=13, font_color=COL["green"], bold=True, radius=12)


def d_dual_tower(d: Drawio, row_y: int, mode: str, accent: str) -> None:
    x0, y0, w, h = 310, row_y + 92, 1500, 304
    d.rect(x0, y0, w, h, "", "#FFFFFF", COL["line"], radius=18)
    d.text(x0 + 18, y0 + 10, 318, 20, "Enlarged CLIP dual tower", 13, COL["ink"], True)
    d_pill(d, x0 + 1248, y0 + 9, 104, "image tower", COL["image_soft"], "#93C5FD", COL["image"])
    d_pill(d, x0 + 1362, y0 + 9, 90, "text tower", COL["text_soft"], "#FCD34D", COL["text"])
    d.rect(x0 + 24, y0 + 42, w - 48, 112, "", COL["image_pale"], "#BFDBFE", radius=14)
    d.rect(x0 + 24, y0 + 174, w - 48, 112, "", COL["text_pale"], "#FCD34D", radius=14)
    d.text(x0 + 36, y0 + 52, 96, 36, "IMAGE\nTOWER", 13, COL["image"], True, align="center")
    d.text(x0 + 36, y0 + 184, 96, 36, "TEXT\nTOWER", 13, COL["text"], True, align="center")

    x_prompt = x0 + 126
    x_enc = x0 + 394
    x_feat = x0 + 700
    x_slot = x0 + 974
    y_img = y0 + 55
    y_txt = y0 + 187

    d.rect(x_prompt, y_img, 238, 86, "Image patches\nfrom the MVTec cable image", "#FFFFFF", COL["line"], font_size=13, font_color=COL["muted"], bold=True, radius=12)
    for r in range(3):
        for c in range(5):
            d.rect(x_prompt + 28 + c * 37, y_img + 34 + r * 15, 25, 11, "", COL["image_soft"], "#93C5FD", radius=4, stroke_width=1)
    d_encoder(d, x_enc, y_img, 260, 86, "Image Encoder", COL["image_soft"], "#60A5FA")
    d_token_block(d, x_feat, y_img, mode)
    d_image_slot(d, x_slot, y_img, mode)

    d_prompt_block(d, x_prompt, y_txt)
    d_encoder(d, x_enc, y_txt, 260, 86, "Text Encoder", COL["text_soft"], "#FBBF24")
    d_prototypes(d, x_feat, y_txt, mode)
    d_text_slot(d, x_slot, y_txt, mode)

    d.arrow(x_prompt + 238, y_img + 43, x_enc, y_img + 43, COL["image"], 3)
    d.arrow(x_enc + 260, y_img + 43, x_feat, y_img + 43, COL["image"], 3)
    d.arrow(x_feat + 238, y_img + 43, x_slot, y_img + 43, accent, 3)
    d.arrow(x_prompt + 238, y_txt + 43, x_enc, y_txt + 43, COL["text"], 3)
    d.arrow(x_enc + 260, y_txt + 43, x_feat, y_txt + 43, COL["text"], 3)
    d.arrow(x_feat + 238, y_txt + 43, x_slot, y_txt + 43, accent, 3)
    if mode == "direct":
        d.arrow(x_feat + 160, y_img + 75, x_feat + 110, y_txt + 36, COL["red"], 3, dashed=True)
        d.text(x_feat + 170, y0 + 152, 206, 18, "unfiltered cross-tower update", 11, COL["red"], True, align="center")
    elif mode == "wavelet":
        d.arrow(x_slot + 244, y_img + 58, x_slot + 245, y_txt + 30, COL["green"], 3, dashed=True)
        d.text(x_slot + 24, y0 + 152, 252, 18, "wavelet-gated cross-tower update", 11, COL["green"], True, align="center")
    else:
        d.arrow(x0 + 834, y0 + 159, x0 + 880, y0 + 159, COL["line_dark"], 3, end=False)
        d.arrow(x0 + 853, y0 + 144, x0 + 867, y0 + 174, COL["red"], 4, end=False)
        d.arrow(x0 + 867, y0 + 144, x0 + 853, y0 + 174, COL["red"], 4, end=False)
        d.text(x0 + 782, y0 + 150, 180, 18, "no adaptation bridge", 11, COL["subtle"], True, align="center")


def d_similarity(d: Drawio, x: int, y: int, mode: str) -> None:
    d.rect(x, y, 282, 210, "CLIP patch-text similarity", "#FFFFFF", COL["violet"], font_size=13, font_color=COL["violet"], bold=True, radius=12, valign="top")
    d.text(x + 80, y + 42, 64, 16, "normal", 11, COL["image"], True, align="center")
    d.text(x + 154, y + 42, 76, 16, "abnormal", 11, COL["red"], True, align="center")
    vals = {
        "fixed": ["0.72", "0.21", "0.64", "0.33", "0.57", "0.41", "0.51", "0.48"],
        "direct": ["0.56", "0.42", "0.50", "0.55", "0.44", "0.68", "0.47", "0.61"],
        "wavelet": ["0.78", "0.18", "0.73", "0.24", "0.38", "0.83", "0.70", "0.29"],
    }[mode]
    k = 0
    for r in range(4):
        d.text(x + 28, y + 68 + r * 27, 32, 15, f"x{r+1}", 10, COL["subtle"], align="center")
        for c in range(2):
            v = float(vals[k])
            fill = "#93C5FD" if c == 0 and v > 0.62 else COL["image_pale"] if c == 0 else "#F87171" if v > 0.62 else COL["red_pale"]
            stroke = "#60A5FA" if c == 0 else "#FCA5A5"
            d.rect(x + 82 + c * 78, y + 66 + r * 27, 60, 20, vals[k], fill, stroke, font_size=10, radius=4, stroke_width=1)
            k += 1
    d.text(x + 20, y + 184, 242, 16, "score = sim(x_i,p_A) - sim(x_i,p_N)", 11, COL["subtle"], align="center")


def d_output(d: Drawio, row_y: int, path: Path, note: str, accent: str) -> None:
    x, y = 2248, row_y + 132
    d.rect(x - 20, y - 42, 294, 280, "", "#FFFFFF", COL["line"], radius=12)
    d.text(x, y - 30, 150, 18, "Anomaly map", 13, COL["muted"], True, align="center")
    d.image(x, y, 158, 158, path)
    d.text(x + 174, y + 2, 78, 18, "Output", 13, accent, True, align="center")
    d.text(x + 174, y + 34, 82, 118, note, 11, COL["muted"])
    d.rect(x + 14, y + 181, 226, 34, "same input and prompts", COL["gray_pale"], COL["line"], font_size=11, font_color=COL["subtle"], radius=9, stroke_width=1)


def d_row(d: Drawio, row_y: int, title: str, subtitle: str, mode: str, accent: str, map_path: Path, note: str) -> None:
    d.rect(48, row_y, W - 96, 438, "", COL["paper"], "#D9E2EC", radius=16)
    d.rect(54, row_y + 86, 6, 300, "", accent, accent, radius=0, stroke_width=0)
    d.text(78, row_y + 19, 760, 30, title, 25, COL["ink"], True)
    d.text(80, row_y + 53, 1110, 24, subtitle, 16, COL["muted"])
    d.text(86, row_y + 104, 160, 18, "Image input", 13, COL["muted"], True, align="center")
    d.image(86, row_y + 132, 160, 148, INPUT_IMG)
    d.rect(78, row_y + 302, 176, 74, "Prompt pair", "#FFFFFF", COL["line"], font_size=13, font_color=COL["muted"], bold=True, radius=12, valign="top")
    d_pill(d, 94, row_y + 339, 132, "normal cable", COL["image_pale"], "#BFDBFE", COL["image"])
    d_pill(d, 94, row_y + 364, 132, "damaged cable", COL["red_pale"], "#FECACA", COL["red"])
    d_dual_tower(d, row_y, mode, accent)
    d_similarity(d, 1842, row_y + 136, mode)
    d_output(d, row_y, map_path, note, accent)
    d.arrow(246, row_y + 206, 310, row_y + 190, COL["image"], 3)
    d.arrow(254, row_y + 342, 310, row_y + 313, COL["text"], 3)
    d.arrow(1810, row_y + 190, 1842, row_y + 207, COL["image"], 3)
    d.arrow(1810, row_y + 324, 1842, row_y + 275, COL["text"], 3)
    d.arrow(2124, row_y + 241, 2228, row_y + 238, accent, 3)


def write_drawio() -> None:
    d = Drawio()
    d.text(60, 26, 1440, 46, "Enlarged CLIP dual-tower comparison for zero-shot anomaly detection", 38, COL["ink"], True)
    d.text(
        62,
        72,
        1540,
        28,
        "Each row keeps the same image and normal/abnormal text prompts; the difference is placed inside the dual-tower adaptation path.",
        20,
        COL["muted"],
    )
    d.rect(1656, 36, 860, 58, "", "#FFFFFF", COL["line"], radius=12)
    d_pill(d, 1680, 50, 128, "image tower", COL["image_soft"], "#93C5FD", COL["image"])
    d_pill(d, 1830, 50, 116, "text tower", COL["text_soft"], "#FCD34D", COL["text"])
    d_pill(d, 1968, 50, 118, "CLIP sim", COL["violet_soft"], "#C4B5FD", COL["violet"])
    d_pill(d, 2108, 50, 130, "drift path", COL["red_soft"], "#FCA5A5", COL["red"])
    d_pill(d, 2260, 50, 150, "wavelet gate", COL["green_soft"], "#86EFAC", COL["green"])
    d_pill(d, 2426, 50, 64, "v2", "#FFFFFF", COL["line_dark"], COL["subtle"])
    rows = [
        (
            120,
            "(a) Conventional CLIP-ZSAD",
            "Two frozen CLIP towers only perform patch-text matching with fixed normal and abnormal text prototypes.",
            "fixed",
            COL["image"],
            MAP_FIXED,
            "Fixed anchors give weak or diffuse localization when the test domain shifts.",
        ),
        (
            586,
            "(b) Direct test-time prototype adaptation",
            "Visual test tokens are fed back to the text-prototype side without filtering, so defect tokens can move p_N.",
            "direct",
            COL["red"],
            MAP_DIRECT,
            "The normal prototype drifts, producing unstable heatmaps.",
        ),
        (
            1052,
            "(c) Ours: wavelet-guided prototype adaptation",
            "A wavelet adapter gates unreliable high-frequency evidence before reliable semantics update the normal prototype.",
            "wavelet",
            COL["green"],
            MAP_FINAL,
            "p_N adapts from reliable semantics while defect cues are suppressed.",
        ),
    ]
    for args in rows:
        d_row(d, *args)
    DRAWIO_OUT.write_text(d.xml(), encoding="utf-8")


def write_audit() -> None:
    AUDIT_OUT.write_text(
        "\n".join(
            [
                "# Figure 4 enlarged CLIP dual-tower v2 audit",
                "",
                "- Canvas: 2600 x 1580 px, three stacked rows with a larger CLIP dual-tower container in every row.",
                "- Visual intent: make the image tower and text tower the dominant structure, then place the architectural difference inside their update paths.",
                "- Real raster inputs: one MVTec cable image and three anomaly-map outputs are embedded as PNG images.",
                "- Native/editable elements: row titles, prompt labels, tower lanes, encoders, tokens, prototypes, adaptation modules, similarity matrices, arrows, and legend.",
                "- Row (a): both towers are frozen; normal/abnormal text prototypes stay locked and no cross-tower adaptation bridge exists.",
                "- Row (b): abnormal visual tokens are marked in the image tower and a red dashed cross-tower update shows direct contamination of p_N into p_N'.",
                "- Row (c): a wavelet adapter is inserted inside the image tower; DWT separates LL pass and high-frequency suppression before a gated update reaches p_N in the text tower.",
                "- Prompt coverage: Normal: \"normal cable\" and Abnormal: \"damaged cable\" are visible inside every text tower.",
                "- Visual QA target: enlarged towers, readable labels, no intentional text overlap, and clear distinction between fixed, drift, and wavelet-gated structures.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    for p in (INPUT_IMG, MAP_FIXED, MAP_DIRECT, MAP_FINAL):
        if not p.exists():
            raise FileNotFoundError(p)
    render_png()
    write_drawio()
    write_audit()
    print(PNG_OUT)
    print(DRAWIO_OUT)
    print(AUDIT_OUT)


if __name__ == "__main__":
    main()
