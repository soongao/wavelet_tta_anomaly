from __future__ import annotations

import base64
import html
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


OUT_DIR = Path(__file__).resolve().parent
ASSET_DIR = OUT_DIR / "mvtec_three_row_selection_assets"

PNG_OUT = OUT_DIR / "figure4_clip_dual_tower_ppt_template_v4.png"
DRAWIO_OUT = OUT_DIR / "figure4_clip_dual_tower_ppt_template_v4.drawio"
AUDIT_OUT = OUT_DIR / "figure4_clip_dual_tower_ppt_template_v4.audit.md"

INPUT_IMG = ASSET_DIR / "mvtec_cable_input.png"
MAP_FIXED = ASSET_DIR / "mvtec_cable_fixed.png"
MAP_DIRECT = ASSET_DIR / "mvtec_cable_direct.png"
MAP_FINAL = ASSET_DIR / "mvtec_cable_final.png"

W, H = 2600, 1580


COL = {
    "bg": "#061A24",
    "bg_deep": "#001F2F",
    "band": "#082330",
    "panel": "#0B2B38",
    "panel2": "#103342",
    "ink": "#F5FBFF",
    "muted": "#B7D0DB",
    "subtle": "#7FA2AF",
    "line": "#2F5E70",
    "line2": "#40798D",
    "image": "#5B9BD4",
    "image_soft": "#123A55",
    "image_pale": "#0E2F45",
    "text": "#F4B083",
    "text_soft": "#3C2C22",
    "text_pale": "#281F19",
    "violet": "#735D94",
    "violet_soft": "#211A32",
    "red": "#C64949",
    "red_soft": "#3A1B22",
    "green": "#01AF50",
    "green_soft": "#103A2A",
    "green_pale": "#0B2F23",
    "teal": "#00A6A6",
    "teal_soft": "#08363D",
    "orange": "#F4B083",
    "orange_soft": "#3B2A1F",
    "gray": "#6E8792",
    "gray_soft": "#162A33",
    "white": "#FFFFFF",
}


def rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def rgba(hex_color: str, alpha: int) -> tuple[int, int, int, int]:
    return rgb(hex_color) + (alpha,)


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
    "title": font(39, True),
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
    out: list[str] = []
    for raw in text.split("\n"):
        words = raw.split(" ")
        cur = ""
        for word in words:
            cand = word if not cur else cur + " " + word
            if tsize(draw, cand, fnt)[0] <= width:
                cur = cand
            else:
                if cur:
                    out.append(cur)
                cur = word
        out.append(cur)
    return out


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
    radius: int = 10,
    width: int = 2,
    alpha: int = 255,
    shadow: bool = False,
) -> None:
    x, y, w, h = box
    if shadow:
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        od.rounded_rectangle((x + 5, y + 8, x + w + 5, y + h + 8), radius=radius, fill=(0, 0, 0, 68))
        img.alpha_composite(ov)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=rgba(fill, alpha), outline=outline, width=width)


def arrow_head(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str, head: int) -> None:
    sx, sy = start
    ex, ey = end
    ang = math.atan2(ey - sy, ex - sx)
    p1 = (
        ex - head * math.cos(ang) + head * 0.55 * math.sin(ang),
        ey - head * math.sin(ang) - head * 0.55 * math.cos(ang),
    )
    p2 = (
        ex - head * math.cos(ang) - head * 0.55 * math.sin(ang),
        ey - head * math.sin(ang) + head * 0.55 * math.cos(ang),
    )
    draw.polygon([end, p1, p2], fill=color)


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
    arrow_head(draw, start, end, color, head)


def poly_arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    color: str,
    width: int = 4,
    dashed: bool = False,
    head: int = 14,
) -> None:
    for a, b in zip(points[:-2], points[1:-1]):
        draw.line((a[0], a[1], b[0], b[1]), fill=color, width=width)
    arrow(draw, points[-2], points[-1], color, width=width, dashed=dashed, head=head)


def pill(img: Image.Image, box: tuple[int, int, int, int], text: str, fill: str, stroke: str, color: str) -> None:
    rr(img, box, fill, stroke, radius=max(8, box[3] // 2), width=2)
    text_box(ImageDraw.Draw(img), box, text, F["tiny_b"], color)


def paste_img(img: Image.Image, path: Path, box: tuple[int, int, int, int], radius: int = 10, border: str = COL["line2"]) -> None:
    x, y, w, h = box
    src = Image.open(path).convert("RGB")
    fitted = ImageOps.fit(src, (w, h), Image.Resampling.LANCZOS)
    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle((0, 0, w, h), radius=radius, fill=255)
    img.paste(fitted, (x, y), mask)
    ImageDraw.Draw(img).rounded_rectangle((x, y, x + w, y + h), radius=radius, outline=border, width=2)


def draw_background(img: Image.Image) -> None:
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, W, H), fill=COL["bg"])
    draw.rectangle((0, 0, W, 112), fill=COL["bg_deep"])
    for x in range(0, W, 80):
        draw.line((x, 112, x, H), fill=rgba("#0B3140", 65), width=1)
    for y in range(112, H, 80):
        draw.line((0, y, W, y), fill=rgba("#0B3140", 55), width=1)
    draw.rectangle((0, 109, W, 112), fill=COL["line2"])


def encoder_stack(img: Image.Image, box: tuple[int, int, int, int], label: str, fill: str, stroke: str, text_color: str) -> None:
    rr(img, box, fill, stroke, radius=12, width=2, shadow=True)
    x, y, w, h = box
    draw = ImageDraw.Draw(img)
    for i in range(6):
        xx = x + 16 + i * 10
        draw.line((xx, y + 14, xx, y + h - 14), fill=stroke, width=2)
    text_box(draw, (x + 70, y + 10, w - 110, h - 20), label, F["module"], text_color)
    pill(img, (x + w - 78, y + 10, 62, 24), "frozen", COL["panel"], stroke, COL["muted"])


def token_block(img: Image.Image, x: int, y: int, mode: str) -> None:
    draw = ImageDraw.Draw(img)
    rr(img, (x, y, 238, 86), COL["panel"], COL["line"], radius=10, width=2)
    text_box(draw, (x + 14, y + 8, 210, 16), "Patch embeddings {x_i}", F["small_b"], COL["muted"], align="left")
    bad = {(1, 5), (2, 4), (2, 5), (1, 6)}
    stable = {(0, 1), (1, 2), (2, 2), (0, 3)}
    for r in range(3):
        for c in range(8):
            fill, stroke = COL["image_soft"], COL["image"]
            if mode == "direct" and (r, c) in bad:
                fill, stroke = COL["red_soft"], COL["red"]
            if mode == "wavelet" and (r, c) in stable:
                fill, stroke = COL["green_soft"], COL["green"]
            draw.rounded_rectangle(
                (x + 16 + c * 26, y + 34 + r * 15, x + 35 + c * 26, y + 45 + r * 15),
                radius=4,
                fill=fill,
                outline=stroke,
                width=1,
            )
    if mode == "direct":
        text_box(draw, (x + 18, y + 68, 202, 14), "includes defect tokens", F["tiny_b"], COL["red"])
    elif mode == "wavelet":
        text_box(draw, (x + 18, y + 68, 202, 14), "reliable candidates", F["tiny_b"], COL["green"])
    else:
        text_box(draw, (x + 18, y + 68, 202, 14), "no adaptation source", F["tiny"], COL["subtle"])


def prompt_block(img: Image.Image, x: int, y: int) -> None:
    draw = ImageDraw.Draw(img)
    rr(img, (x, y, 238, 86), COL["panel"], COL["line"], radius=10, width=2)
    text_box(draw, (x + 14, y + 8, 210, 16), "Text prompts", F["small_b"], COL["muted"], align="left")
    rr(img, (x + 16, y + 34, 206, 21), COL["image_soft"], COL["image"], radius=6, width=1)
    rr(img, (x + 16, y + 59, 206, 21), COL["red_soft"], COL["red"], radius=6, width=1)
    draw.text((x + 25, y + 38), 'Normal: "normal cable"', font=F["tiny_b"], fill=COL["image"])
    draw.text((x + 25, y + 63), 'Abnormal: "damaged cable"', font=F["tiny_b"], fill=COL["red"])


def prototypes(img: Image.Image, x: int, y: int, mode: str) -> None:
    draw = ImageDraw.Draw(img)
    rr(img, (x, y, 238, 86), COL["panel"], COL["line"], radius=10, width=2)
    text_box(draw, (x + 14, y + 8, 210, 16), "Text prototypes", F["small_b"], COL["muted"], align="left")
    if mode == "direct":
        draw.ellipse((x + 36, y + 35, x + 78, y + 77), fill=COL["panel2"], outline=COL["gray"], width=2)
        text_box(draw, (x + 32, y + 48, 50, 14), "p_N", F["tiny"], COL["subtle"])
        draw.ellipse((x + 67, y + 31, x + 113, y + 77), fill=COL["red_soft"], outline=COL["red"], width=3)
        text_box(draw, (x + 67, y + 44, 46, 17), "p_N'", F["tiny_b"], COL["red"])
    else:
        draw.ellipse((x + 51, y + 31, x + 97, y + 77), fill=COL["image_soft"], outline=COL["image"], width=3)
        text_box(draw, (x + 51, y + 44, 46, 17), "p_N", F["tiny_b"], COL["image"])
    draw.ellipse((x + 153, y + 31, x + 199, y + 77), fill=COL["red_soft"], outline=COL["red"], width=3)
    text_box(draw, (x + 153, y + 44, 46, 17), "p_A", F["tiny_b"], COL["red"])
    if mode == "fixed":
        pill(img, (x + 101, y + 54, 50, 22), "lock", COL["gray_soft"], COL["gray"], COL["muted"])
    if mode == "wavelet":
        pill(img, (x + 100, y + 54, 52, 22), "gate", COL["green_soft"], COL["green"], COL["green"])


def adaptation_branch(img: Image.Image, x: int, y: int, w: int, h: int, mode: str) -> None:
    draw = ImageDraw.Draw(img)
    if mode == "fixed":
        rr(img, (x + 80, y + 2, w - 160, h - 4), COL["gray_soft"], COL["gray"], radius=10, width=2)
        text_box(draw, (x + 102, y + 8, w - 204, 16), "No test-time feedback branch", F["small_b"], COL["muted"])
        draw.line((x + w // 2 - 22, y + 29, x + w // 2 + 22, y + 29), fill=COL["gray"], width=3)
        draw.line((x + w // 2 - 10, y + 18, x + w // 2 + 10, y + 42), fill=COL["red"], width=4)
        draw.line((x + w // 2 + 10, y + 18, x + w // 2 - 10, y + 42), fill=COL["red"], width=4)
        text_box(draw, (x + 134, y + 36, w - 268, 13), "zero-shot matching only", F["tiny"], COL["subtle"])
        return
    if mode == "direct":
        rr(img, (x, y, w, h), COL["red_soft"], COL["red"], radius=10, width=2)
        text_box(draw, (x + 16, y + 8, 136, 16), "Raw TTA branch", F["small_b"], COL["red"], align="left")
        for i, c in enumerate([COL["image_soft"], COL["red_soft"], COL["image_soft"], COL["red_soft"], COL["gray_soft"]]):
            draw.rounded_rectangle((x + 168 + i * 36, y + 15, x + 192 + i * 36, y + 37), radius=5, fill=c, outline=COL["line2"], width=1)
        arrow(draw, (x + 370, y + h // 2), (x + 500, y + h // 2), COL["red"], width=3, dashed=False, head=10)
        text_box(draw, (x + 505, y + 9, 165, 15), "update p_N (unfiltered)", F["tiny_b"], COL["red"], align="center")
        text_box(draw, (x + 162, y + 39, 270, 13), "abnormal tokens can contaminate normal prototype", F["tiny_b"], COL["red"])
        return
    rr(img, (x, y, w, h), COL["panel"], COL["green"], radius=10, width=2)
    text_box(draw, (x + 16, y + 8, 166, 16), "Wavelet feedback branch", F["small_b"], COL["green"], align="left")
    rr(img, (x + 186, y + 12, 58, 30), COL["teal_soft"], COL["teal"], radius=8, width=2)
    text_box(draw, (x + 186, y + 19, 58, 12), "DWT", F["tiny_b"], COL["teal"])
    arrow(draw, (x + 250, y + 27), (x + 300, y + 27), COL["green"], width=3, head=10)
    pill(img, (x + 310, y + 9, 82, 23), "LL pass", COL["green_soft"], COL["green"], COL["green"])
    pill(img, (x + 310, y + 35, 108, 23), "HF suppress", COL["orange_soft"], COL["orange"], COL["orange"])
    rr(img, (x + 438, y + 14, 74, 30), COL["green_pale"], COL["green"], radius=8, width=2)
    text_box(draw, (x + 438, y + 21, 74, 12), "gate", F["tiny_b"], COL["green"])
    text_box(draw, (x + 56, y + 39, 180, 13), "select reliable visual semantics", F["tiny"], COL["subtle"])


def similarity_input_slot(img: Image.Image, x: int, y: int, mode: str, lane: str) -> None:
    draw = ImageDraw.Draw(img)
    if lane == "image":
        rr(img, (x, y, 300, 86), COL["panel"], COL["image"], radius=10, width=2)
        text_box(draw, (x + 18, y + 8, 264, 16), "Visual features to CLIP sim", F["small_b"], COL["image"])
        for i in range(6):
            cx = x + 50 + i * 34
            cy = y + 48 + (8 if i % 2 else 0)
            draw.ellipse((cx - 9, cy - 9, cx + 9, cy + 9), fill=COL["image_soft"], outline=COL["image"], width=2)
        text_box(draw, (x + 32, y + 68, 236, 14), "frozen image tower output V", F["tiny"], COL["subtle"])
        return

    stroke = COL["line2"]
    fill = COL["panel"]
    label = "Fixed prototype set P"
    note = "p_N and p_A are unchanged"
    color = COL["muted"]
    if mode == "direct":
        stroke, fill, label, note, color = COL["red"], COL["red_soft"], "Drifted prototype set P'", "raw TTA contaminates p_N", COL["red"]
    elif mode == "wavelet":
        stroke, fill, label, note, color = COL["green"], COL["green_soft"], "Adapted prototype set P*", "gated update preserves p_A", COL["green"]
    rr(img, (x, y, 300, 86), fill, stroke, radius=10, width=2)
    text_box(draw, (x + 18, y + 8, 264, 16), label, F["small_b"], color)
    pn_fill = COL["image_soft"] if mode != "direct" else COL["red_soft"]
    pn_label = "p_N" if mode != "direct" else "p_N'"
    draw.ellipse((x + 64, y + 34, x + 110, y + 80), fill=pn_fill, outline=stroke if mode != "fixed" else COL["image"], width=3)
    text_box(draw, (x + 64, y + 47, 46, 16), pn_label, F["tiny_b"], color if mode != "fixed" else COL["image"])
    draw.ellipse((x + 186, y + 34, x + 232, y + 80), fill=COL["red_soft"], outline=COL["red"], width=3)
    text_box(draw, (x + 186, y + 47, 46, 16), "p_A", F["tiny_b"], COL["red"])
    text_box(draw, (x + 36, y + 66, 228, 14), note, F["tiny"], color if mode != "fixed" else COL["subtle"])


def similarity_block(img: Image.Image, x: int, y: int, mode: str) -> None:
    draw = ImageDraw.Draw(img)
    rr(img, (x, y, 282, 210), COL["panel"], COL["violet"], radius=12, width=2, shadow=True)
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
            fill = COL["image"] if c == 0 and strong else COL["image_soft"] if c == 0 else COL["red"] if strong else COL["red_soft"]
            stroke = COL["image"] if c == 0 else COL["red"]
            xx = x + 82 + c * 78
            yy = y + 66 + r * 27
            draw.rounded_rectangle((xx, yy, xx + 60, yy + 20), radius=4, fill=fill, outline=stroke, width=1)
            text_box(draw, (xx, yy + 2, 60, 15), f"{val:.2f}", F["tiny_b"], COL["white"])
    text_box(draw, (x + 20, y + 184, 242, 16), "score = sim(x_i,p_A) - sim(x_i,p_N)", F["tiny"], COL["muted"])


def output_panel(img: Image.Image, row_y: int, path: Path, note: str, accent: str) -> None:
    draw = ImageDraw.Draw(img)
    x, y = 2248, row_y + 132
    rr(img, (x - 20, y - 42, 294, 280), COL["panel"], COL["line"], radius=12, width=2, shadow=True)
    text_box(draw, (x, y - 30, 150, 18), "Anomaly map", F["small_b"], COL["muted"])
    paste_img(img, path, (x, y, 158, 158), radius=10, border=accent)
    text_box(draw, (x + 174, y + 2, 78, 18), "Output", F["small_b"], accent)
    text_box(draw, (x + 174, y + 34, 82, 118), note, F["tiny"], COL["muted"], align="left")
    rr(img, (x + 14, y + 181, 226, 34), COL["gray_soft"], COL["line"], radius=8, width=1)
    text_box(draw, (x + 22, y + 189, 210, 14), "same input and prompts", F["tiny"], COL["subtle"])


def dual_tower(img: Image.Image, row_y: int, mode: str, accent: str) -> None:
    draw = ImageDraw.Draw(img)
    x0, y0, w, h = 310, row_y + 92, 1500, 304
    rr(img, (x0, y0, w, h), COL["panel"], COL["line2"], radius=16, width=2, shadow=True)
    text_box(draw, (x0 + 18, y0 + 10, 500, 20), "Enlarged CLIP dual tower with feedback path", F["small_b"], COL["ink"], align="left")
    pill(img, (x0 + 1248, y0 + 9, 104, 24), "image tower", COL["image_soft"], COL["image"], COL["image"])
    pill(img, (x0 + 1362, y0 + 9, 90, 24), "text tower", COL["text_soft"], COL["text"], COL["text"])

    img_lane = (x0 + 24, y0 + 42, w - 48, 92)
    adapt_lane = (x0 + 258, y0 + 144, 760, 58)
    txt_lane = (x0 + 24, y0 + 204, w - 48, 90)
    rr(img, img_lane, COL["image_pale"], COL["image"], radius=12, width=2)
    rr(img, txt_lane, COL["text_pale"], COL["text"], radius=12, width=2)
    text_box(draw, (img_lane[0] + 12, img_lane[1] + 8, 96, 18), "IMAGE", F["small_b"], COL["image"])
    text_box(draw, (img_lane[0] + 12, img_lane[1] + 28, 96, 18), "TOWER", F["small_b"], COL["image"])
    text_box(draw, (txt_lane[0] + 12, txt_lane[1] + 8, 96, 18), "TEXT", F["small_b"], COL["text"])
    text_box(draw, (txt_lane[0] + 12, txt_lane[1] + 28, 96, 18), "TOWER", F["small_b"], COL["text"])
    text_box(draw, (adapt_lane[0] - 122, adapt_lane[1] + 16, 102, 16), "FEEDBACK", F["tiny_b"], accent if mode != "fixed" else COL["subtle"])
    text_box(draw, (adapt_lane[0] - 122, adapt_lane[1] + 32, 102, 14), "BRANCH", F["tiny_b"], accent if mode != "fixed" else COL["subtle"])

    x_prompt = x0 + 126
    x_enc = x0 + 394
    x_feat = x0 + 700
    x_slot = x0 + 974
    y_img = img_lane[1] + 3
    y_txt = txt_lane[1] + 2

    rr(img, (x_prompt, y_img, 238, 86), COL["panel"], COL["line"], radius=10, width=2)
    text_box(draw, (x_prompt + 14, y_img + 8, 210, 16), "Image patches", F["small_b"], COL["muted"], align="left")
    for r in range(3):
        for c in range(5):
            draw.rounded_rectangle(
                (x_prompt + 28 + c * 37, y_img + 34 + r * 15, x_prompt + 53 + c * 37, y_img + 45 + r * 15),
                radius=4,
                fill=COL["image_soft"],
                outline=COL["image"],
                width=1,
            )
    text_box(draw, (x_prompt + 20, y_img + 68, 198, 14), "from the MVTec cable image", F["tiny"], COL["subtle"])

    encoder_stack(img, (x_enc, y_img, 260, 86), "Image Encoder", COL["image_soft"], COL["image"], COL["ink"])
    token_block(img, x_feat, y_img, mode)
    similarity_input_slot(img, x_slot, y_img, mode, "image")

    prompt_block(img, x_prompt, y_txt)
    encoder_stack(img, (x_enc, y_txt, 260, 86), "Text Encoder", COL["text_soft"], COL["text"], COL["ink"])
    prototypes(img, x_feat, y_txt, mode)
    similarity_input_slot(img, x_slot, y_txt, mode, "text")
    adaptation_branch(img, adapt_lane[0], adapt_lane[1], adapt_lane[2], adapt_lane[3], mode)

    arrow(draw, (x_prompt + 238, y_img + 43), (x_enc, y_img + 43), COL["image"], width=4)
    arrow(draw, (x_enc + 260, y_img + 43), (x_feat, y_img + 43), COL["image"], width=4)
    arrow(draw, (x_feat + 238, y_img + 43), (x_slot, y_img + 43), accent, width=4)
    arrow(draw, (x_prompt + 238, y_txt + 43), (x_enc, y_txt + 43), COL["text"], width=4)
    arrow(draw, (x_enc + 260, y_txt + 43), (x_feat, y_txt + 43), COL["text"], width=4)
    arrow(draw, (x_feat + 238, y_txt + 43), (x_slot, y_txt + 43), accent, width=4)

    if mode == "direct":
        arrow(draw, (x_feat + 119, y_img + 86), (adapt_lane[0] + 190, adapt_lane[1] + 16), COL["red"], width=4, dashed=True, head=12)
        poly_arrow(
            draw,
            [
                (adapt_lane[0] + 500, adapt_lane[1] + 30),
                (x_feat + 225, adapt_lane[1] + 30),
                (x_feat + 102, y_txt + 2),
            ],
            COL["red"],
            width=4,
            dashed=True,
            head=12,
        )
        text_box(draw, (adapt_lane[0] + 508, adapt_lane[1] + 36, 225, 16), "feedback to text-side prototype", F["tiny_b"], COL["red"])
    elif mode == "wavelet":
        arrow(draw, (x_feat + 119, y_img + 86), (adapt_lane[0] + 188, adapt_lane[1] + 27), COL["green"], width=4, dashed=True, head=12)
        poly_arrow(
            draw,
            [
                (adapt_lane[0] + 512, adapt_lane[1] + 29),
                (x_feat + 226, adapt_lane[1] + 29),
                (x_feat + 103, y_txt + 2),
            ],
            COL["green"],
            width=4,
            dashed=True,
            head=12,
        )
        text_box(draw, (adapt_lane[0] + 522, adapt_lane[1] + 36, 205, 16), "gated update to p_N", F["tiny_b"], COL["green"])
    else:
        draw.line((adapt_lane[0] + 372, adapt_lane[1] + 29, adapt_lane[0] + 420, adapt_lane[1] + 29), fill=COL["gray"], width=3)
        draw.line((adapt_lane[0] + 390, adapt_lane[1] + 16, adapt_lane[0] + 406, adapt_lane[1] + 42), fill=COL["red"], width=4)
        draw.line((adapt_lane[0] + 406, adapt_lane[1] + 16, adapt_lane[0] + 390, adapt_lane[1] + 42), fill=COL["red"], width=4)


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
    rr(img, (48, row_y, W - 96, 438), COL["band"], COL["line"], radius=18, width=2, alpha=245, shadow=True)
    draw.rectangle((54, row_y + 86, 60, row_y + 386), fill=accent)
    text_box(draw, (78, row_y + 19, 760, 30), title, F["row"], COL["ink"], align="left")
    text_box(draw, (80, row_y + 53, 1110, 24), subtitle, F["body"], COL["muted"], align="left")

    text_box(draw, (86, row_y + 104, 160, 18), "Image input", F["small_b"], COL["muted"])
    paste_img(img, INPUT_IMG, (86, row_y + 132, 160, 148), radius=10, border=COL["line2"])
    rr(img, (78, row_y + 302, 176, 74), COL["panel"], COL["line"], radius=10, width=2)
    text_box(draw, (92, row_y + 314, 148, 16), "Prompt pair", F["small_b"], COL["muted"])
    pill(img, (94, row_y + 339, 132, 22), "normal cable", COL["image_soft"], COL["image"], COL["image"])
    pill(img, (94, row_y + 364, 132, 22), "damaged cable", COL["red_soft"], COL["red"], COL["red"])

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
    text_box(draw, (60, 24, 1510, 45), "CLIP dual-tower architectures for zero-shot anomaly detection", F["title"], COL["ink"], align="left")
    text_box(
        draw,
        (62, 70, 1575, 28),
        "The same MVTec cable image and normal/abnormal prompts are used; only the in-tower adaptation path changes.",
        F["sub"],
        COL["muted"],
        align="left",
    )
    rr(img, (1660, 34, 846, 58), COL["panel"], COL["line"], radius=14, width=2, shadow=True)
    pill(img, (1682, 50, 128, 30), "image tower", COL["image_soft"], COL["image"], COL["image"])
    pill(img, (1832, 50, 116, 30), "text tower", COL["text_soft"], COL["text"], COL["text"])
    pill(img, (1968, 50, 118, 30), "CLIP sim", COL["violet_soft"], COL["violet"], COL["violet"])
    pill(img, (2108, 50, 130, 30), "drift path", COL["red_soft"], COL["red"], COL["red"])
    pill(img, (2260, 50, 150, 30), "wavelet gate", COL["green_soft"], COL["green"], COL["green"])
    pill(img, (2428, 50, 56, 30), "v4", COL["gray_soft"], COL["gray"], COL["muted"])


def render_png() -> None:
    img = Image.new("RGBA", (W, H), rgb(COL["bg"]) + (255,))
    draw_background(img)
    header(img)
    rows = [
        (
            120,
            "(a) Conventional CLIP-ZSAD",
            "Two frozen CLIP towers perform patch-text matching with fixed normal and abnormal text prototypes.",
            "fixed",
            COL["image"],
            MAP_FIXED,
            "Fixed anchors give weak or diffuse localization under domain shift.",
        ),
        (
            586,
            "(b) Direct test-time prototype adaptation",
            "Image patch embeddings feed back to text-side prototypes without filtering, so defect tokens can move p_N.",
            "direct",
            COL["red"],
            MAP_DIRECT,
            "The normal prototype drifts, producing unstable heatmaps.",
        ),
        (
            1052,
            "(c) Ours: wavelet-guided prototype adaptation",
            "A wavelet branch gates unreliable high-frequency evidence before reliable semantics update p_N; p_A stays anchored.",
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
        fill: str = COL["panel"],
        stroke: str = COL["line"],
        font_size: int = 13,
        font_color: str = COL["ink"],
        bold: bool = False,
        radius: int = 10,
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
        font_color: str = COL["ink"],
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
        color: str = COL["ink"],
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
            '  <diagram id="clip-dual-tower-ppt-template-v4" name="CLIP dual tower PPT template v4">\n'
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
    d.rect(x, y, w, h, text, fill, stroke, font_size=18, bold=True, radius=10)
    d_pill(d, x + w - 78, y + 10, 62, "frozen", COL["panel"], stroke, COL["muted"])


def d_tokens(d: Drawio, x: int, y: int, mode: str) -> None:
    d.rect(x, y, 238, 86, "Patch embeddings {x_i}", COL["panel"], COL["line"], font_size=13, font_color=COL["muted"], bold=True, radius=10, valign="top")
    label = "includes defect tokens" if mode == "direct" else "reliable candidates" if mode == "wavelet" else "no adaptation source"
    color = COL["red"] if mode == "direct" else COL["green"] if mode == "wavelet" else COL["subtle"]
    for r in range(3):
        for c in range(8):
            fill, stroke = COL["image_soft"], COL["image"]
            if mode == "direct" and (r, c) in {(1, 5), (2, 4), (2, 5), (1, 6)}:
                fill, stroke = COL["red_soft"], COL["red"]
            if mode == "wavelet" and (r, c) in {(0, 1), (1, 2), (2, 2), (0, 3)}:
                fill, stroke = COL["green_soft"], COL["green"]
            d.rect(x + 16 + c * 26, y + 34 + r * 15, 19, 11, "", fill, stroke, radius=4, stroke_width=1)
    d.text(x + 18, y + 68, 202, 14, label, 11, color, mode != "fixed", align="center")


def d_prototypes(d: Drawio, x: int, y: int, mode: str) -> None:
    d.rect(x, y, 238, 86, "Text prototypes", COL["panel"], COL["line"], font_size=13, font_color=COL["muted"], bold=True, radius=10, valign="top")
    if mode == "direct":
        d.ellipse(x + 36, y + 35, 42, 42, "p_N", COL["panel2"], COL["gray"], 10, COL["subtle"])
        d.ellipse(x + 67, y + 31, 46, 46, "p_N'", COL["red_soft"], COL["red"], 12, COL["red"], True, 3)
    else:
        d.ellipse(x + 51, y + 31, 46, 46, "p_N", COL["image_soft"], COL["image"], 12, COL["image"], True, 3)
    d.ellipse(x + 153, y + 31, 46, 46, "p_A", COL["red_soft"], COL["red"], 12, COL["red"], True, 3)
    if mode == "fixed":
        d_pill(d, x + 101, y + 54, 50, "lock", COL["gray_soft"], COL["gray"], COL["muted"])
    if mode == "wavelet":
        d_pill(d, x + 100, y + 54, 52, "gate", COL["green_soft"], COL["green"], COL["green"])


def d_adapt(d: Drawio, x: int, y: int, w: int, h: int, mode: str) -> None:
    if mode == "fixed":
        d.rect(x + 80, y + 2, w - 160, h - 4, "No test-time feedback branch\nzero-shot matching only", COL["gray_soft"], COL["gray"], font_size=12, font_color=COL["muted"], bold=True, radius=10)
        d.arrow(x + w // 2 - 22, y + 29, x + w // 2 + 22, y + 29, COL["gray"], 3, end=False)
        d.arrow(x + w // 2 - 10, y + 18, x + w // 2 + 10, y + 42, COL["red"], 4, end=False)
        d.arrow(x + w // 2 + 10, y + 18, x + w // 2 - 10, y + 42, COL["red"], 4, end=False)
        return
    if mode == "direct":
        d.rect(x, y, w, h, "Raw TTA branch\nabnormal tokens can contaminate p_N", COL["red_soft"], COL["red"], font_size=12, font_color=COL["red"], bold=True, radius=10, align="left")
        d.arrow(x + 370, y + h // 2, x + 500, y + h // 2, COL["red"], 3)
        d.text(x + 505, y + 9, 165, 15, "update p_N (unfiltered)", 11, COL["red"], True, align="center")
        return
    d.rect(x, y, w, h, "Wavelet feedback branch", COL["panel"], COL["green"], font_size=12, font_color=COL["green"], bold=True, radius=10, valign="top", align="left")
    d.rect(x + 186, y + 12, 58, 30, "DWT", COL["teal_soft"], COL["teal"], font_size=11, font_color=COL["teal"], bold=True, radius=8)
    d.arrow(x + 250, y + 27, x + 300, y + 27, COL["green"], 3)
    d_pill(d, x + 310, y + 9, 82, "LL pass", COL["green_soft"], COL["green"], COL["green"])
    d_pill(d, x + 310, y + 35, 108, "HF suppress", COL["orange_soft"], COL["orange"], COL["orange"])
    d.rect(x + 438, y + 14, 74, 30, "gate", COL["green_pale"], COL["green"], font_size=11, font_color=COL["green"], bold=True, radius=8)


def d_similarity_inputs(d: Drawio, x: int, y: int, mode: str, lane: str) -> None:
    if lane == "image":
        d.rect(x, y, 300, 86, "Visual features to CLIP sim\nfrozen image tower output V", COL["panel"], COL["image"], font_size=13, font_color=COL["image"], bold=True, radius=10)
        return
    stroke = COL["line2"]
    fill = COL["panel"]
    label = "Fixed prototype set P\np_N and p_A are unchanged"
    color = COL["muted"]
    if mode == "direct":
        stroke, fill, label, color = COL["red"], COL["red_soft"], "Drifted prototype set P'\nraw TTA contaminates p_N", COL["red"]
    elif mode == "wavelet":
        stroke, fill, label, color = COL["green"], COL["green_soft"], "Adapted prototype set P*\ngated update preserves p_A", COL["green"]
    d.rect(x, y, 300, 86, label, fill, stroke, font_size=13, font_color=color, bold=True, radius=10)


def d_dual_tower(d: Drawio, row_y: int, mode: str, accent: str) -> None:
    x0, y0, w, h = 310, row_y + 92, 1500, 304
    d.rect(x0, y0, w, h, "", COL["panel"], COL["line2"], radius=16)
    d.text(x0 + 18, y0 + 10, 500, 20, "Enlarged CLIP dual tower with feedback path", 13, COL["ink"], True)
    d_pill(d, x0 + 1248, y0 + 9, 104, "image tower", COL["image_soft"], COL["image"], COL["image"])
    d_pill(d, x0 + 1362, y0 + 9, 90, "text tower", COL["text_soft"], COL["text"], COL["text"])
    d.rect(x0 + 24, y0 + 42, w - 48, 92, "", COL["image_pale"], COL["image"], radius=12)
    d.rect(x0 + 24, y0 + 204, w - 48, 90, "", COL["text_pale"], COL["text"], radius=12)
    d.text(x0 + 36, y0 + 50, 96, 36, "IMAGE\nTOWER", 13, COL["image"], True, align="center")
    d.text(x0 + 36, y0 + 212, 96, 36, "TEXT\nTOWER", 13, COL["text"], True, align="center")

    x_prompt = x0 + 126
    x_enc = x0 + 394
    x_feat = x0 + 700
    x_slot = x0 + 974
    y_img = y0 + 45
    y_txt = y0 + 206
    adapt_x, adapt_y, adapt_w, adapt_h = x0 + 258, y0 + 144, 760, 58
    d.text(x0 + 160, y0 + 160, 102, 30, "FEEDBACK\nBRANCH", 11, accent if mode != "fixed" else COL["subtle"], True, align="center")

    d.rect(x_prompt, y_img, 238, 86, "Image patches\nfrom the MVTec cable image", COL["panel"], COL["line"], font_size=13, font_color=COL["muted"], bold=True, radius=10)
    d_encoder(d, x_enc, y_img, 260, 86, "Image Encoder", COL["image_soft"], COL["image"])
    d_tokens(d, x_feat, y_img, mode)
    d_similarity_inputs(d, x_slot, y_img, mode, "image")
    d.rect(x_prompt, y_txt, 238, 86, 'Text prompts\nNormal: "normal cable"\nAbnormal: "damaged cable"', COL["panel"], COL["line"], font_size=12, font_color=COL["muted"], bold=True, radius=10)
    d_encoder(d, x_enc, y_txt, 260, 86, "Text Encoder", COL["text_soft"], COL["text"])
    d_prototypes(d, x_feat, y_txt, mode)
    d_similarity_inputs(d, x_slot, y_txt, mode, "text")
    d_adapt(d, adapt_x, adapt_y, adapt_w, adapt_h, mode)

    d.arrow(x_prompt + 238, y_img + 43, x_enc, y_img + 43, COL["image"], 3)
    d.arrow(x_enc + 260, y_img + 43, x_feat, y_img + 43, COL["image"], 3)
    d.arrow(x_feat + 238, y_img + 43, x_slot, y_img + 43, accent, 3)
    d.arrow(x_prompt + 238, y_txt + 43, x_enc, y_txt + 43, COL["text"], 3)
    d.arrow(x_enc + 260, y_txt + 43, x_feat, y_txt + 43, COL["text"], 3)
    d.arrow(x_feat + 238, y_txt + 43, x_slot, y_txt + 43, accent, 3)
    if mode == "direct":
        d.arrow(x_feat + 119, y_img + 86, adapt_x + 190, adapt_y + 16, COL["red"], 3, dashed=True)
        d.arrow(adapt_x + 500, adapt_y + 30, x_feat + 103, y_txt + 2, COL["red"], 3, dashed=True)
        d.text(adapt_x + 508, adapt_y + 36, 225, 16, "feedback to text-side prototype", 11, COL["red"], True, align="center")
    elif mode == "wavelet":
        d.arrow(x_feat + 119, y_img + 86, adapt_x + 188, adapt_y + 27, COL["green"], 3, dashed=True)
        d.arrow(adapt_x + 512, adapt_y + 29, x_feat + 103, y_txt + 2, COL["green"], 3, dashed=True)
        d.text(adapt_x + 522, adapt_y + 36, 205, 16, "gated update to p_N", 11, COL["green"], True, align="center")


def d_similarity(d: Drawio, x: int, y: int, mode: str) -> None:
    d.rect(x, y, 282, 210, "CLIP patch-text similarity", COL["panel"], COL["violet"], font_size=13, font_color=COL["violet"], bold=True, radius=12, valign="top")
    vals = {
        "fixed": ["0.72", "0.21", "0.64", "0.33", "0.57", "0.41", "0.51", "0.48"],
        "direct": ["0.56", "0.42", "0.50", "0.55", "0.44", "0.68", "0.47", "0.61"],
        "wavelet": ["0.78", "0.18", "0.73", "0.24", "0.38", "0.83", "0.70", "0.29"],
    }[mode]
    d.text(x + 80, y + 42, 64, 16, "normal", 11, COL["image"], True, align="center")
    d.text(x + 154, y + 42, 76, 16, "abnormal", 11, COL["red"], True, align="center")
    k = 0
    for r in range(4):
        d.text(x + 28, y + 68 + r * 27, 32, 15, f"x{r+1}", 10, COL["subtle"], align="center")
        for c in range(2):
            v = float(vals[k])
            fill = COL["image"] if c == 0 and v > 0.62 else COL["image_soft"] if c == 0 else COL["red"] if v > 0.62 else COL["red_soft"]
            stroke = COL["image"] if c == 0 else COL["red"]
            d.rect(x + 82 + c * 78, y + 66 + r * 27, 60, 20, vals[k], fill, stroke, font_size=10, font_color=COL["white"], radius=4, stroke_width=1)
            k += 1
    d.text(x + 20, y + 184, 242, 16, "score = sim(x_i,p_A) - sim(x_i,p_N)", 11, COL["muted"], align="center")


def d_output(d: Drawio, row_y: int, path: Path, note: str, accent: str) -> None:
    x, y = 2248, row_y + 132
    d.rect(x - 20, y - 42, 294, 280, "", COL["panel"], COL["line"], radius=12)
    d.text(x, y - 30, 150, 18, "Anomaly map", 13, COL["muted"], True)
    d.image(x, y, 158, 158, path)
    d.text(x + 174, y + 2, 78, 18, "Output", 13, accent, True)
    d.text(x + 174, y + 34, 82, 118, note, 11, COL["muted"])
    d.rect(x + 14, y + 181, 226, 34, "same input and prompts", COL["gray_soft"], COL["line"], font_size=11, font_color=COL["subtle"], radius=8, stroke_width=1)


def d_row(
    d: Drawio,
    row_y: int,
    title: str,
    subtitle: str,
    mode: str,
    accent: str,
    map_path: Path,
    note: str,
) -> None:
    d.rect(48, row_y, W - 96, 438, "", COL["band"], COL["line"], radius=18)
    d.rect(54, row_y + 86, 6, 300, "", accent, accent, radius=0, stroke_width=0)
    d.text(78, row_y + 19, 760, 30, title, 25, COL["ink"], True)
    d.text(80, row_y + 53, 1110, 24, subtitle, 16, COL["muted"])
    d.text(86, row_y + 104, 160, 18, "Image input", 13, COL["muted"], True, align="center")
    d.image(86, row_y + 132, 160, 148, INPUT_IMG)
    d.rect(78, row_y + 302, 176, 74, "Prompt pair", COL["panel"], COL["line"], font_size=13, font_color=COL["muted"], bold=True, radius=10, valign="top")
    d_pill(d, 94, row_y + 339, 132, "normal cable", COL["image_soft"], COL["image"], COL["image"])
    d_pill(d, 94, row_y + 364, 132, "damaged cable", COL["red_soft"], COL["red"], COL["red"])
    d_dual_tower(d, row_y, mode, accent)
    d_similarity(d, 1842, row_y + 136, mode)
    d_output(d, row_y, map_path, note, accent)
    d.arrow(246, row_y + 206, 310, row_y + 190, COL["image"], 3)
    d.arrow(254, row_y + 342, 310, row_y + 313, COL["text"], 3)
    d.arrow(1810, row_y + 190, 1842, row_y + 207, COL["image"], 3)
    d.arrow(1810, row_y + 324, 1842, row_y + 275, COL["text"], 3)
    d.arrow(2124, row_y + 241, 2228, row_y + 238, accent, 3)


def render_drawio() -> None:
    d = Drawio()
    d.rect(0, 0, W, H, "", COL["bg"], COL["bg"], radius=0, stroke_width=0)
    d.rect(0, 0, W, 112, "", COL["bg_deep"], COL["bg_deep"], radius=0, stroke_width=0)
    d.rect(0, 109, W, 3, "", COL["line2"], COL["line2"], radius=0, stroke_width=0)
    d.text(60, 24, 1510, 45, "CLIP dual-tower architectures for zero-shot anomaly detection", 39, COL["ink"], True)
    d.text(62, 70, 1575, 28, "The same MVTec cable image and normal/abnormal prompts are used; only the in-tower adaptation path changes.", 20, COL["muted"])
    d.rect(1660, 34, 846, 58, "", COL["panel"], COL["line"], radius=14)
    d_pill(d, 1682, 50, 128, "image tower", COL["image_soft"], COL["image"], COL["image"])
    d_pill(d, 1832, 50, 116, "text tower", COL["text_soft"], COL["text"], COL["text"])
    d_pill(d, 1968, 50, 118, "CLIP sim", COL["violet_soft"], COL["violet"], COL["violet"])
    d_pill(d, 2108, 50, 130, "drift path", COL["red_soft"], COL["red"], COL["red"])
    d_pill(d, 2260, 50, 150, "wavelet gate", COL["green_soft"], COL["green"], COL["green"])
    d_pill(d, 2428, 50, 56, "v4", COL["gray_soft"], COL["gray"], COL["muted"])
    rows = [
        (
            120,
            "(a) Conventional CLIP-ZSAD",
            "Two frozen CLIP towers perform patch-text matching with fixed normal and abnormal text prototypes.",
            "fixed",
            COL["image"],
            MAP_FIXED,
            "Fixed anchors give weak or diffuse localization under domain shift.",
        ),
        (
            586,
            "(b) Direct test-time prototype adaptation",
            "Image patch embeddings feed back to text-side prototypes without filtering, so defect tokens can move p_N.",
            "direct",
            COL["red"],
            MAP_DIRECT,
            "The normal prototype drifts, producing unstable heatmaps.",
        ),
        (
            1052,
            "(c) Ours: wavelet-guided prototype adaptation",
            "A wavelet branch gates unreliable high-frequency evidence before reliable semantics update p_N; p_A stays anchored.",
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
                "# Figure 4 PPT-template v4 audit",
                "",
                "- Visual source: user-provided PPT template style tokens and preview montage.",
                "- Major style choices: dark blue-black canvas, cyan thin strokes, restrained white typography, green/red method emphasis.",
                "- Semantic gate: Direct TTA and wavelet links are drawn as feedback branches from image patch embeddings back to the text-side prototype block.",
                "- Image assets: MVTec cable input and generated cable anomaly maps are embedded in both PNG and Draw.io outputs.",
                "- Editability: Draw.io uses native boxes, labels, arrows, lanes, tokens, and embedded raster image crops for the MVTec images/maps.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    for p in [INPUT_IMG, MAP_FIXED, MAP_DIRECT, MAP_FINAL]:
        if not p.exists():
            raise FileNotFoundError(p)
    render_png()
    render_drawio()
    write_audit()
    print(PNG_OUT)
    print(DRAWIO_OUT)
    print(AUDIT_OUT)


if __name__ == "__main__":
    main()
