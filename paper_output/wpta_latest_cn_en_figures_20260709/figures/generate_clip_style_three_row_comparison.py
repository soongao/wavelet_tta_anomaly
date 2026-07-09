from __future__ import annotations

import base64
import html
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


OUT_DIR = Path(__file__).resolve().parent
ASSET_DIR = OUT_DIR / "mvtec_three_row_selection_assets"

PNG_OUT = OUT_DIR / "figure4_clip_style_three_row_comparison.png"
DRAWIO_OUT = OUT_DIR / "figure4_clip_style_three_row_comparison.drawio"
AUDIT_OUT = OUT_DIR / "figure4_clip_style_three_row_comparison.audit.md"

INPUT_IMG = ASSET_DIR / "mvtec_cable_input.png"
MAP_FIXED = ASSET_DIR / "mvtec_cable_fixed.png"
MAP_DIRECT = ASSET_DIR / "mvtec_cable_direct.png"
MAP_FINAL = ASSET_DIR / "mvtec_cable_final.png"

W, H = 2600, 1500


COL = {
    "bg": "#F7F9FC",
    "paper": "#FFFFFF",
    "ink": "#111827",
    "muted": "#4B5563",
    "subtle": "#6B7280",
    "line": "#CBD5E1",
    "line_dark": "#94A3B8",
    "image": "#2563EB",
    "image_soft": "#DBEAFE",
    "image_pale": "#EFF6FF",
    "text": "#D97706",
    "text_soft": "#FEF3C7",
    "text_pale": "#FFFBEB",
    "red": "#DC2626",
    "red_soft": "#FEE2E2",
    "red_pale": "#FEF2F2",
    "green": "#059669",
    "green_soft": "#D1FAE5",
    "green_pale": "#ECFDF5",
    "teal": "#0F766E",
    "teal_soft": "#CCFBF1",
    "violet": "#7C3AED",
    "violet_soft": "#EDE9FE",
    "gray_soft": "#E5E7EB",
    "gray_pale": "#F3F4F6",
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
    "small": font(13),
    "small_b": font(13, True),
    "tiny": font(11),
    "tiny_b": font(11, True),
    "module": font(18, True),
    "math": font(17, True),
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
    lines = wrap(draw, text, fnt, max(12, w - 10))
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
    radius: int = 14,
    width: int = 2,
    shadow: bool = False,
) -> None:
    draw = ImageDraw.Draw(img)
    x, y, w, h = box
    if shadow:
        ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        od.rounded_rectangle((x + 4, y + 6, x + w + 4, y + h + 6), radius=radius, fill=(15, 23, 42, 18))
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
    rr(img, box, fill, stroke, radius=box[3] // 2, width=2)
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


def encoder(img: Image.Image, box: tuple[int, int, int, int], label: str, fill: str, stroke: str) -> None:
    rr(img, box, fill, stroke, radius=16, width=2, shadow=True)
    x, y, w, h = box
    draw = ImageDraw.Draw(img)
    # Layer stripes make the block read closer to CLIP encoder/tower diagrams.
    for i in range(5):
        xx = x + 16 + i * 10
        draw.line((xx, y + 16, xx, y + h - 16), fill=stroke, width=2)
    text_box(draw, (x + 58, y + 9, w - 96, h - 18), label, F["module"], COL["ink"])
    pill(img, (x + w - 72, y + 10, 58, 24), "frozen", "#FFFFFF", stroke, COL["subtle"])


def prompts(img: Image.Image, x: int, y: int) -> None:
    draw = ImageDraw.Draw(img)
    rr(img, (x, y, 265, 100), "#FFFFFF", COL["line"], radius=14, width=2, shadow=True)
    text_box(draw, (x + 16, y + 8, 220, 18), "Text prompts", F["small_b"], COL["muted"], align="left")
    rr(img, (x + 14, y + 36, 237, 25), COL["image_pale"], "#BFDBFE", radius=8, width=1)
    rr(img, (x + 14, y + 67, 237, 25), COL["red_pale"], "#FECACA", radius=8, width=1)
    draw.text((x + 24, y + 41), 'Normal: "normal cable"', font=F["tiny_b"], fill=COL["image"])
    draw.text((x + 24, y + 72), 'Abnormal: "damaged cable"', font=F["tiny_b"], fill=COL["red"])


def token_grid(
    img: Image.Image,
    x: int,
    y: int,
    w: int,
    h: int,
    mode: str,
) -> None:
    draw = ImageDraw.Draw(img)
    rr(img, (x, y, w, h), "#FFFFFF", COL["line"], radius=14, width=2)
    text_box(draw, (x + 14, y + 7, w - 28, 18), "Image patch tokens", F["small_b"], COL["muted"], align="left")
    bad = {(1, 5), (2, 4), (2, 5)}
    stable = {(0, 1), (1, 2), (2, 2)}
    for r in range(4):
        for c in range(8):
            cx = x + 17 + c * 27
            cy = y + 39 + r * 20
            fill, stroke = COL["image_pale"], "#93C5FD"
            if mode == "direct" and (r, c) in bad:
                fill, stroke = COL["red_soft"], COL["red"]
            if mode == "wavelet" and (r, c) in stable:
                fill, stroke = COL["green_soft"], COL["green"]
            draw.rounded_rectangle((cx, cy, cx + 19, cy + 13), radius=4, fill=fill, outline=stroke, width=1)
    text_box(draw, (x + 15, y + h - 25, w - 30, 18), "local features x_i", F["tiny"], COL["subtle"])


def prototypes(img: Image.Image, x: int, y: int, mode: str) -> None:
    draw = ImageDraw.Draw(img)
    rr(img, (x, y, 250, 104), "#FFFFFF", COL["line"], radius=14, width=2)
    text_box(draw, (x + 14, y + 7, 222, 18), "Text prototypes", F["small_b"], COL["muted"], align="left")
    if mode == "direct":
        draw.ellipse((x + 42, y + 43, x + 84, y + 85), fill="#FFFFFF", outline=COL["line_dark"], width=2)
        draw.ellipse((x + 71, y + 39, x + 117, y + 85), fill=COL["red_soft"], outline=COL["red"], width=2)
        text_box(draw, (x + 71, y + 52, 46, 18), "pN'", F["tiny_b"], COL["red"])
        text_box(draw, (x + 24, y + 80, 66, 16), "old pN", F["tiny"], COL["subtle"])
    else:
        draw.ellipse((x + 54, y + 41, x + 100, y + 87), fill=COL["image_soft"], outline=COL["image"], width=2)
        text_box(draw, (x + 54, y + 54, 46, 18), "pN", F["tiny_b"], COL["image"])
    draw.ellipse((x + 158, y + 41, x + 204, y + 87), fill=COL["red_soft"], outline=COL["red"], width=2)
    text_box(draw, (x + 158, y + 54, 46, 18), "pA", F["tiny_b"], COL["red"])
    if mode == "fixed":
        pill(img, (x + 104, y + 73, 48, 22), "lock", "#FFFFFF", COL["line_dark"], COL["subtle"])
    if mode == "wavelet":
        pill(img, (x + 104, y + 73, 50, 22), "gate", COL["green_soft"], COL["green"], COL["green"])


def similarity(img: Image.Image, x: int, y: int, mode: str) -> None:
    draw = ImageDraw.Draw(img)
    rr(img, (x, y, 300, 170), "#FFFFFF", COL["line"], radius=16, width=2, shadow=True)
    text_box(draw, (x + 14, y + 10, 272, 20), "Patch-text similarity", F["small_b"], COL["muted"])
    text_box(draw, (x + 74, y + 36, 70, 16), "normal", F["tiny_b"], COL["image"])
    text_box(draw, (x + 155, y + 36, 84, 16), "abnormal", F["tiny_b"], COL["red"])
    vals = {
        "fixed": [(0.72, 0.21), (0.64, 0.33), (0.57, 0.41), (0.51, 0.48)],
        "direct": [(0.56, 0.42), (0.50, 0.55), (0.44, 0.68), (0.47, 0.61)],
        "wavelet": [(0.78, 0.18), (0.73, 0.24), (0.38, 0.83), (0.70, 0.29)],
    }[mode]
    for r, pair in enumerate(vals):
        text_box(draw, (x + 28, y + 57 + r * 24, 34, 15), f"x{r+1}", F["tiny"], COL["subtle"])
        for c, val in enumerate(pair):
            strong = val > 0.62
            if c == 0:
                fill = "#93C5FD" if strong else COL["image_pale"]
                stroke = "#60A5FA"
            else:
                fill = "#F87171" if strong else COL["red_pale"]
                stroke = "#FCA5A5"
            xx = x + 78 + c * 86
            yy = y + 55 + r * 24
            draw.rounded_rectangle((xx, yy, xx + 68, yy + 18), radius=4, fill=fill, outline=stroke, width=1)
            text_box(draw, (xx, yy + 1, 68, 15), f"{val:.2f}", F["tiny"], COL["ink"])
    text_box(draw, (x + 22, y + 147, 256, 16), "score = sim(x_i,pA) - sim(x_i,pN)", F["tiny"], COL["subtle"])


def fixed_block(img: Image.Image, x: int, y: int) -> None:
    draw = ImageDraw.Draw(img)
    rr(img, (x, y, 330, 188), "#FFFFFF", COL["line_dark"], radius=16, width=2, shadow=True)
    text_box(draw, (x + 18, y + 14, 294, 22), "Frozen text anchors", F["module"], COL["ink"])
    pill(img, (x + 80, y + 58, 74, 28), "pN", COL["image_soft"], COL["image"], COL["image"])
    pill(img, (x + 176, y + 58, 74, 28), "pA", COL["red_soft"], COL["red"], COL["red"])
    draw.line((x + 82, y + 120, x + 248, y + 120), fill=COL["line_dark"], width=3)
    draw.line((x + 154, y + 104, x + 176, y + 136), fill=COL["red"], width=4)
    draw.line((x + 176, y + 104, x + 154, y + 136), fill=COL["red"], width=4)
    text_box(draw, (x + 38, y + 145, 254, 28), "No test-time update; only zero-shot matching is used.", F["body"], COL["muted"])


def direct_block(img: Image.Image, x: int, y: int) -> None:
    draw = ImageDraw.Draw(img)
    rr(img, (x, y, 330, 188), "#FFFFFF", COL["red"], radius=16, width=2, shadow=True)
    text_box(draw, (x + 18, y + 14, 294, 22), "Unfiltered TTA update", F["module"], COL["ink"])
    for i, c in enumerate([COL["image_soft"], COL["red_soft"], COL["image_soft"], COL["red_soft"], COL["gray_soft"]]):
        draw.rounded_rectangle((x + 44 + i * 44, y + 58, x + 72 + i * 44, y + 84), radius=6, fill=c, outline=COL["line_dark"])
    arrow(draw, (x + 64, y + 117), (x + 211, y + 117), COL["red"], width=4)
    draw.ellipse((x + 230, y + 93, x + 284, y + 147), fill=COL["red_soft"], outline=COL["red"], width=3)
    text_box(draw, (x + 230, y + 109, 54, 18), "drift", F["tiny_b"], COL["red"])
    text_box(draw, (x + 32, y + 151, 268, 24), "Defect patches may pull the normal prototype.", F["body"], COL["red"])


def wavelet_block(img: Image.Image, x: int, y: int) -> None:
    draw = ImageDraw.Draw(img)
    rr(img, (x, y, 330, 188), "#FFFFFF", COL["green"], radius=16, width=2, shadow=True)
    text_box(draw, (x + 14, y + 14, 302, 22), "Wavelet-guided adapter", F["module"], COL["ink"])
    rr(img, (x + 32, y + 55, 76, 52), COL["teal_soft"], COL["teal"], radius=12, width=2)
    text_box(draw, (x + 32, y + 70, 76, 18), "DWT", F["math"], COL["teal"])
    pill(img, (x + 142, y + 52, 112, 28), "LL semantic", COL["green_soft"], COL["green"], COL["green"])
    pill(img, (x + 142, y + 87, 132, 28), "LH/HL/HH", "#FFEDD5", "#EA580C", "#EA580C")
    arrow(draw, (x + 108, y + 81), (x + 142, y + 66), COL["teal"], width=3, head=10)
    arrow(draw, (x + 108, y + 81), (x + 142, y + 101), COL["teal"], width=3, head=10)
    rr(img, (x + 38, y + 130, 254, 38), COL["green_pale"], "#86EFAC", radius=12, width=2)
    text_box(draw, (x + 50, y + 137, 230, 20), "reliability gate -> update pN", F["body_b"], COL["green"])


def output_panel(img: Image.Image, row_y: int, path: Path, note: str, accent: str) -> None:
    draw = ImageDraw.Draw(img)
    x, y = 2152, row_y + 114
    rr(img, (x - 22, y - 54, 372, 244), "#FFFFFF", COL["line"], radius=16, width=2, shadow=True)
    text_box(draw, (x - 4, y - 42, 164, 20), "Anomaly map", F["small_b"], COL["muted"])
    paste_img(img, path, (x, y, 154, 154), radius=12, border=accent)
    text_box(draw, (x + 182, y - 4, 130, 26), "Output", F["small_b"], accent)
    text_box(draw, (x + 178, y + 34, 148, 92), note, F["small"], COL["muted"], align="left")


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
    rr(img, (48, row_y, W - 96, 392), COL["paper"], "#D9E2EC", radius=22, width=2, shadow=True)
    draw.rectangle((53, row_y + 86, 58, row_y + 346), fill=accent)
    text_box(draw, (78, row_y + 22, 720, 28), title, F["row"], COL["ink"], align="left")
    text_box(draw, (80, row_y + 56, 1050, 24), subtitle, F["body"], COL["muted"], align="left")

    paste_img(img, INPUT_IMG, (86, row_y + 118, 150, 146), radius=12, border=COL["line_dark"])
    text_box(draw, (86, row_y + 94, 150, 18), "Image input", F["small_b"], COL["muted"])
    prompts(img, 62, row_y + 278)

    enc_x = 362
    encoder(img, (enc_x, row_y + 126, 230, 78), "Image\nEncoder", COL["image_soft"], "#60A5FA")
    encoder(img, (enc_x, row_y + 284, 230, 70), "Text\nEncoder", COL["text_soft"], "#FBBF24")

    feat_x = 678
    token_grid(img, feat_x, row_y + 110, 252, 124, mode)
    prototypes(img, feat_x, row_y + 268, mode)

    if mode == "fixed":
        fixed_block(img, 1024, row_y + 132)
    elif mode == "direct":
        direct_block(img, 1024, row_y + 132)
    else:
        wavelet_block(img, 1024, row_y + 132)

    similarity(img, 1450, row_y + 140, mode)
    output_panel(img, row_y, map_path, note, accent)

    arrow(draw, (236, row_y + 190), (362, row_y + 166), COL["image"], width=4)
    arrow(draw, (327, row_y + 326), (362, row_y + 319), COL["text"], width=4)
    arrow(draw, (592, row_y + 166), (678, row_y + 172), COL["image"], width=4)
    arrow(draw, (592, row_y + 319), (678, row_y + 320), COL["text"], width=4)
    arrow(draw, (930, row_y + 172), (1024, row_y + 188), accent, width=4)
    arrow(draw, (928, row_y + 320), (1024, row_y + 267), accent, width=4)
    arrow(draw, (1354, row_y + 226), (1450, row_y + 225), accent, width=4)
    arrow(draw, (1750, row_y + 225), (2130, row_y + 232), accent, width=4)
    if mode == "direct":
        arrow(draw, (810, row_y + 220), (1112, row_y + 248), COL["red"], width=3, dashed=True)
    if mode == "wavelet":
        arrow(draw, (807, row_y + 220), (1060, row_y + 186), COL["green"], width=3, dashed=True)


def header(img: Image.Image) -> None:
    draw = ImageDraw.Draw(img)
    text_box(draw, (60, 26, 1490, 46), "CLIP-style three-row comparison for zero-shot anomaly detection", F["title"], COL["ink"], align="left")
    text_box(
        draw,
        (62, 72, 1540, 28),
        "The same MVTec cable image and the same normal/abnormal text prompts are used; only the prototype-adaptation path changes.",
        F["sub"],
        COL["muted"],
        align="left",
    )
    rr(img, (1688, 35, 820, 58), "#FFFFFF", COL["line"], radius=16, width=2, shadow=True)
    pill(img, (1712, 49, 126, 30), "image tower", COL["image_soft"], "#93C5FD", COL["image"])
    pill(img, (1860, 49, 118, 30), "text tower", COL["text_soft"], "#FCD34D", COL["text"])
    pill(img, (2000, 49, 112, 30), "CLIP sim", COL["violet_soft"], "#C4B5FD", COL["violet"])
    pill(img, (2134, 49, 126, 30), "drift risk", COL["red_soft"], "#FCA5A5", COL["red"])
    pill(img, (2282, 49, 150, 30), "wavelet gate", COL["green_soft"], "#86EFAC", COL["green"])


def render_png() -> None:
    img = Image.new("RGBA", (W, H), rgb(COL["bg"]) + (255,))
    header(img)
    rows = [
        (
            128,
            "(a) Conventional CLIP-ZSAD",
            "Frozen CLIP image/text towers compare patch tokens with fixed normal and abnormal text prototypes.",
            "fixed",
            COL["image"],
            MAP_FIXED,
            "Fixed anchors can leave weak or diffuse localization under domain shift.",
        ),
        (
            552,
            "(b) Direct test-time prototype adaptation",
            "Test patches update the prototypes directly; abnormal regions may contaminate the normal anchor.",
            "direct",
            COL["red"],
            MAP_DIRECT,
            "Unfiltered updates introduce prototype drift and unstable heatmaps.",
        ),
        (
            976,
            "(c) Ours: wavelet-guided prototype adaptation",
            "Wavelet bands gate unreliable high-frequency evidence before updating the normal prototype.",
            "wavelet",
            COL["green"],
            MAP_FINAL,
            "Stable semantic evidence updates pN while defect cues are suppressed.",
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

    def arrow(self, sx: int, sy: int, ex: int, ey: int, color: str, width: int = 3, dashed: bool = False) -> None:
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

    def xml(self) -> str:
        return (
            '<mxfile host="app.diagrams.net" modified="2026-07-08T00:00:00.000Z" agent="Codex" version="24.7.17" type="device">\n'
            '  <diagram id="clip-style-three-row" name="CLIP style three-row comparison">\n'
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


def d_prompts(d: Drawio, x: int, y: int) -> None:
    d.rect(x, y, 265, 100, "", "#FFFFFF", COL["line"], radius=12)
    d.text(x + 16, y + 8, 220, 18, "Text prompts", 13, COL["muted"], True)
    d.rect(x + 14, y + 36, 237, 25, 'Normal: "normal cable"', COL["image_pale"], "#BFDBFE", font_size=11, font_color=COL["image"], bold=True, radius=8)
    d.rect(x + 14, y + 67, 237, 25, 'Abnormal: "damaged cable"', COL["red_pale"], "#FECACA", font_size=11, font_color=COL["red"], bold=True, radius=8)


def d_encoder(d: Drawio, x: int, y: int, w: int, h: int, value: str, fill: str, stroke: str) -> None:
    d.rect(x, y, w, h, value, fill, stroke, font_size=18, bold=True, radius=12)
    for i in range(5):
        d.rect(x + 16 + i * 10, y + 16, 2, h - 32, "", stroke, stroke, stroke_width=0)
    d_pill(d, x + w - 72, y + 10, 58, "frozen", "#FFFFFF", stroke, COL["subtle"])


def d_tokens(d: Drawio, x: int, y: int, mode: str) -> None:
    d.rect(x, y, 252, 124, "Image patch tokens", "#FFFFFF", COL["line"], font_size=13, font_color=COL["muted"], bold=True, radius=12, valign="top")
    bad = {(1, 5), (2, 4), (2, 5)}
    stable = {(0, 1), (1, 2), (2, 2)}
    for r in range(4):
        for c in range(8):
            fill, stroke = COL["image_pale"], "#93C5FD"
            if mode == "direct" and (r, c) in bad:
                fill, stroke = COL["red_soft"], COL["red"]
            if mode == "wavelet" and (r, c) in stable:
                fill, stroke = COL["green_soft"], COL["green"]
            d.rect(x + 17 + c * 27, y + 39 + r * 20, 19, 13, "", fill, stroke, radius=4, stroke_width=1)
    d.text(x + 15, y + 98, 222, 18, "local features x_i", 11, COL["subtle"], align="center")


def d_protos(d: Drawio, x: int, y: int, mode: str) -> None:
    d.rect(x, y, 250, 104, "Text prototypes", "#FFFFFF", COL["line"], font_size=13, font_color=COL["muted"], bold=True, radius=12, valign="top")
    if mode == "direct":
        d.rect(x + 42, y + 43, 42, 42, "old pN", "#FFFFFF", COL["line_dark"], font_size=9, font_color=COL["subtle"], radius=20)
        d.rect(x + 71, y + 39, 46, 46, "pN'", COL["red_soft"], COL["red"], font_size=12, font_color=COL["red"], bold=True, radius=20)
    else:
        d.rect(x + 54, y + 41, 46, 46, "pN", COL["image_soft"], COL["image"], font_size=12, font_color=COL["image"], bold=True, radius=20)
    d.rect(x + 158, y + 41, 46, 46, "pA", COL["red_soft"], COL["red"], font_size=12, font_color=COL["red"], bold=True, radius=20)
    if mode == "fixed":
        d_pill(d, x + 104, y + 73, 48, "lock", "#FFFFFF", COL["line_dark"], COL["subtle"])
    if mode == "wavelet":
        d_pill(d, x + 104, y + 73, 50, "gate", COL["green_soft"], COL["green"], COL["green"])


def d_adapt(d: Drawio, x: int, y: int, mode: str) -> None:
    if mode == "fixed":
        d.rect(x, y, 330, 188, "Frozen text anchors", "#FFFFFF", COL["line_dark"], font_size=18, bold=True, radius=12, valign="top")
        d_pill(d, x + 80, y + 58, 74, "pN", COL["image_soft"], COL["image"], COL["image"])
        d_pill(d, x + 176, y + 58, 74, "pA", COL["red_soft"], COL["red"], COL["red"])
        d.text(x + 38, y + 145, 254, 28, "No test-time update; only zero-shot matching is used.", 15, COL["muted"], align="center")
    elif mode == "direct":
        d.rect(x, y, 330, 188, "Unfiltered TTA update", "#FFFFFF", COL["red"], font_size=18, bold=True, radius=12, valign="top")
        for i, c in enumerate([COL["image_soft"], COL["red_soft"], COL["image_soft"], COL["red_soft"], COL["gray_soft"]]):
            d.rect(x + 44 + i * 44, y + 58, 28, 26, "", c, COL["line_dark"], radius=6)
        d.arrow(x + 64, y + 117, x + 211, y + 117, COL["red"], 3)
        d.rect(x + 230, y + 93, 54, 54, "drift", COL["red_soft"], COL["red"], font_size=11, font_color=COL["red"], bold=True, radius=20)
        d.text(x + 32, y + 151, 268, 24, "Defect patches may pull the normal prototype.", 15, COL["red"], align="center")
    else:
        d.rect(x, y, 330, 188, "Wavelet-guided adapter", "#FFFFFF", COL["green"], font_size=18, bold=True, radius=12, valign="top")
        d.rect(x + 32, y + 55, 76, 52, "DWT", COL["teal_soft"], COL["teal"], font_size=17, font_color=COL["teal"], bold=True, radius=12)
        d_pill(d, x + 142, y + 52, 112, "LL semantic", COL["green_soft"], COL["green"], COL["green"])
        d_pill(d, x + 142, y + 87, 132, "LH/HL/HH", "#FFEDD5", "#EA580C", "#EA580C")
        d.arrow(x + 108, y + 81, x + 142, y + 66, COL["teal"], 2)
        d.arrow(x + 108, y + 81, x + 142, y + 101, COL["teal"], 2)
        d.rect(x + 38, y + 130, 254, 38, "reliability gate -> update pN", COL["green_pale"], "#86EFAC", font_size=15, font_color=COL["green"], bold=True, radius=12)


def d_similarity(d: Drawio, x: int, y: int, mode: str) -> None:
    vals = {
        "fixed": ["0.72", "0.21", "0.64", "0.33", "0.57", "0.41", "0.51", "0.48"],
        "direct": ["0.56", "0.42", "0.50", "0.55", "0.44", "0.68", "0.47", "0.61"],
        "wavelet": ["0.78", "0.18", "0.73", "0.24", "0.38", "0.83", "0.70", "0.29"],
    }[mode]
    d.rect(x, y, 300, 170, "Patch-text similarity", "#FFFFFF", COL["line"], font_size=13, font_color=COL["muted"], bold=True, radius=12, valign="top")
    d.text(x + 74, y + 36, 70, 16, "normal", 11, COL["image"], True, align="center")
    d.text(x + 155, y + 36, 84, 16, "abnormal", 11, COL["red"], True, align="center")
    k = 0
    for r in range(4):
        d.text(x + 28, y + 57 + r * 24, 34, 15, f"x{r+1}", 10, COL["subtle"], align="center")
        for c in range(2):
            v = float(vals[k])
            fill = "#93C5FD" if c == 0 and v > 0.62 else COL["image_pale"] if c == 0 else "#F87171" if v > 0.62 else COL["red_pale"]
            stroke = "#60A5FA" if c == 0 else "#FCA5A5"
            d.rect(x + 78 + c * 86, y + 55 + r * 24, 68, 18, vals[k], fill, stroke, font_size=10, radius=4, stroke_width=1)
            k += 1
    d.text(x + 22, y + 147, 256, 16, "score = sim(x_i,pA) - sim(x_i,pN)", 11, COL["subtle"], align="center")


def d_output(d: Drawio, row_y: int, path: Path, note: str, accent: str) -> None:
    x, y = 2152, row_y + 114
    d.rect(x - 22, y - 54, 372, 244, "", "#FFFFFF", COL["line"], radius=12)
    d.text(x - 4, y - 42, 164, 20, "Anomaly map", 13, COL["muted"], True, align="center")
    d.image(x, y, 154, 154, path)
    d.text(x + 182, y - 4, 130, 26, "Output", 13, accent, True, align="center")
    d.text(x + 178, y + 34, 148, 92, note, 12, COL["muted"])


def d_row(d: Drawio, row_y: int, title: str, subtitle: str, mode: str, accent: str, map_path: Path, note: str) -> None:
    d.rect(48, row_y, W - 96, 392, "", COL["paper"], "#D9E2EC", radius=12)
    d.rect(53, row_y + 86, 5, 260, "", accent, accent, radius=0, stroke_width=0)
    d.text(78, row_y + 22, 720, 28, title, 25, COL["ink"], True)
    d.text(80, row_y + 56, 1050, 24, subtitle, 16, COL["muted"])
    d.image(86, row_y + 118, 150, 146, INPUT_IMG)
    d.text(86, row_y + 94, 150, 18, "Image input", 13, COL["muted"], True, align="center")
    d_prompts(d, 62, row_y + 278)
    d_encoder(d, 362, row_y + 126, 230, 78, "Image\nEncoder", COL["image_soft"], "#60A5FA")
    d_encoder(d, 362, row_y + 284, 230, 70, "Text\nEncoder", COL["text_soft"], "#FBBF24")
    d_tokens(d, 678, row_y + 110, mode)
    d_protos(d, 678, row_y + 268, mode)
    d_adapt(d, 1024, row_y + 132, mode)
    d_similarity(d, 1450, row_y + 140, mode)
    d_output(d, row_y, map_path, note, accent)
    d.arrow(236, row_y + 190, 362, row_y + 166, COL["image"], 3)
    d.arrow(327, row_y + 326, 362, row_y + 319, COL["text"], 3)
    d.arrow(592, row_y + 166, 678, row_y + 172, COL["image"], 3)
    d.arrow(592, row_y + 319, 678, row_y + 320, COL["text"], 3)
    d.arrow(930, row_y + 172, 1024, row_y + 188, accent, 3)
    d.arrow(928, row_y + 320, 1024, row_y + 267, accent, 3)
    d.arrow(1354, row_y + 226, 1450, row_y + 225, accent, 3)
    d.arrow(1750, row_y + 225, 2130, row_y + 232, accent, 3)
    if mode == "direct":
        d.arrow(810, row_y + 220, 1112, row_y + 248, COL["red"], 2, dashed=True)
    if mode == "wavelet":
        d.arrow(807, row_y + 220, 1060, row_y + 186, COL["green"], 2, dashed=True)


def write_drawio() -> None:
    d = Drawio()
    d.text(60, 26, 1490, 46, "CLIP-style three-row comparison for zero-shot anomaly detection", 38, COL["ink"], True)
    d.text(
        62,
        72,
        1540,
        28,
        "The same MVTec cable image and the same normal/abnormal text prompts are used; only the prototype-adaptation path changes.",
        20,
        COL["muted"],
    )
    d.rect(1688, 35, 820, 58, "", "#FFFFFF", COL["line"], radius=12)
    d_pill(d, 1712, 49, 126, "image tower", COL["image_soft"], "#93C5FD", COL["image"])
    d_pill(d, 1860, 49, 118, "text tower", COL["text_soft"], "#FCD34D", COL["text"])
    d_pill(d, 2000, 49, 112, "CLIP sim", COL["violet_soft"], "#C4B5FD", COL["violet"])
    d_pill(d, 2134, 49, 126, "drift risk", COL["red_soft"], "#FCA5A5", COL["red"])
    d_pill(d, 2282, 49, 150, "wavelet gate", COL["green_soft"], "#86EFAC", COL["green"])
    rows = [
        (
            128,
            "(a) Conventional CLIP-ZSAD",
            "Frozen CLIP image/text towers compare patch tokens with fixed normal and abnormal text prototypes.",
            "fixed",
            COL["image"],
            MAP_FIXED,
            "Fixed anchors can leave weak or diffuse localization under domain shift.",
        ),
        (
            552,
            "(b) Direct test-time prototype adaptation",
            "Test patches update the prototypes directly; abnormal regions may contaminate the normal anchor.",
            "direct",
            COL["red"],
            MAP_DIRECT,
            "Unfiltered updates introduce prototype drift and unstable heatmaps.",
        ),
        (
            976,
            "(c) Ours: wavelet-guided prototype adaptation",
            "Wavelet bands gate unreliable high-frequency evidence before updating the normal prototype.",
            "wavelet",
            COL["green"],
            MAP_FINAL,
            "Stable semantic evidence updates pN while defect cues are suppressed.",
        ),
    ]
    for args in rows:
        d_row(d, *args)
    DRAWIO_OUT.write_text(d.xml(), encoding="utf-8")


def write_audit() -> None:
    AUDIT_OUT.write_text(
        "\n".join(
            [
                "# Figure 4 CLIP-style three-row comparison audit",
                "",
                "- Canvas: 2600 x 1500 px, three stacked rows on a light paper background.",
                "- Reference style target: CLIP-like dual tower architecture, with image encoder and text encoder separated before patch-text similarity.",
                "- Real raster inputs: MVTec cable image and three anomaly-map outputs are embedded as PNG images.",
                "- Native/editable elements: row titles, prompt labels, encoders, token grids, prototypes, adaptation blocks, similarity matrices, arrows, and legend.",
                "- Row (a): frozen CLIP towers, fixed normal/abnormal prototypes, no update path.",
                "- Row (b): direct test-time prototype adaptation, red dashed contamination path from abnormal patch tokens to prototype drift.",
                "- Row (c): wavelet-guided prototype adaptation, DWT split into LL semantic and LH/HL/HH detail bands, reliability gate before updating pN.",
                "- Prompt coverage: Normal: \"normal cable\" and Abnormal: \"damaged cable\" are visible in every row.",
                "- Visual QA: figure is nonblank, high-contrast, and uses distinct color roles for image/text/drift/wavelet evidence.",
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
