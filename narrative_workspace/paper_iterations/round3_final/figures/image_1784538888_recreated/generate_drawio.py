#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import math
import random
from pathlib import Path
from urllib.parse import quote
from xml.dom import minidom
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("/Users/bytedance/Downloads/image_1784538888_recreated")
SVG_DIR = OUT_DIR / "svg"
QA_DIR = OUT_DIR / "qa"
REFERENCE = Path("/Users/bytedance/Downloads/image_1784538888.jpeg")
DRAWIO = OUT_DIR / "image_1784538888_recreated.drawio"
PNG = OUT_DIR / "image_1784538888_recreated.png"

CANVAS_W = 6784
CANVAS_H = 2496


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def make_svg_assets() -> dict[str, str]:
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    assets = {
        "lock.svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
  <rect x="56" y="96" width="144" height="116" rx="18" fill="#ffffff" stroke="#111111" stroke-width="16"/>
  <path d="M82 96V72c0-34 21-56 46-56s46 22 46 56v24" fill="none" stroke="#111111" stroke-width="16" stroke-linecap="round"/>
  <circle cx="128" cy="146" r="13" fill="#111111"/>
  <path d="M128 156v28" stroke="#111111" stroke-width="13" stroke-linecap="round"/>
</svg>
""",
        "input_texture.svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
  <defs>
    <filter id="soft" x="-20%" y="-20%" width="140%" height="140%">
      <feTurbulence type="fractalNoise" baseFrequency="0.035" numOctaves="3" seed="17"/>
      <feColorMatrix type="matrix" values="0.14 0 0 0 0.78  0 0.14 0 0 0.78  0 0 0.14 0 0.78  0 0 0 0.22 0"/>
      <feBlend in="SourceGraphic" mode="multiply"/>
    </filter>
    <radialGradient id="spot" cx="50%" cy="50%" r="58%">
      <stop offset="0" stop-color="#b00019"/>
      <stop offset="0.42" stop-color="#e12635"/>
      <stop offset="1" stop-color="#e12635" stop-opacity="0"/>
    </radialGradient>
  </defs>
  <rect width="256" height="256" fill="#d8d8d8" filter="url(#soft)"/>
  <path d="M135 114c10 6 19 15 14 28-5 12-20 10-26 1-9-13 0-36 12-29z" fill="url(#spot)"/>
  <circle cx="116" cy="128" r="8" fill="#c60024" opacity="0.82"/>
  <circle cx="146" cy="101" r="5" fill="#d30025" opacity="0.56"/>
</svg>
""",
        "gauge.svg": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">
  <circle cx="128" cy="128" r="116" fill="#e9edf1" stroke="#111111" stroke-width="8"/>
  <path d="M52 144a80 80 0 0 1 152 0" fill="none" stroke="#d9212e" stroke-width="18" stroke-linecap="round"/>
  <path d="M52 144a80 80 0 0 1 38-68" fill="none" stroke="#8bc34a" stroke-width="18" stroke-linecap="round"/>
  <path d="M90 76a80 80 0 0 1 48-11" fill="none" stroke="#f0d24d" stroke-width="18" stroke-linecap="round"/>
  <path d="M128 128l23-54" stroke="#111111" stroke-width="10" stroke-linecap="round"/>
  <circle cx="128" cy="128" r="19" fill="#ffffff" stroke="#111111" stroke-width="8"/>
  <path d="M128 153v42" stroke="#d9212e" stroke-width="16" stroke-linecap="round"/>
  <circle cx="128" cy="204" r="18" fill="#ffffff" stroke="#111111" stroke-width="7"/>
</svg>
""",
    }
    for name, text in assets.items():
        (SVG_DIR / name).write_text(text, encoding="utf-8")
    return {name: (SVG_DIR / name).read_text(encoding="utf-8") for name in assets}


class Diagram:
    def __init__(self) -> None:
        self.mxfile = ET.Element(
            "mxfile",
            {
                "host": "Electron",
                "agent": "TRAE CLI",
                "version": "24.7.17",
                "pages": "1",
            },
        )
        self.diagram = ET.SubElement(self.mxfile, "diagram", {"name": "Figure", "id": "image-1784538888"})
        self.model = ET.SubElement(
            self.diagram,
            "mxGraphModel",
            {
                "dx": "1800",
                "dy": "900",
                "grid": "1",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": str(CANVAS_W),
                "pageHeight": str(CANVAS_H),
                "math": "0",
                "shadow": "0",
            },
        )
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})
        self.counter = 0

    def next_id(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}-{self.counter:04d}"

    def vertex(self, value: str, style: str, x: float, y: float, w: float, h: float, prefix: str = "v") -> str:
        cid = self.next_id(prefix)
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": cid,
                "value": value,
                "style": style,
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {"x": f"{x:.2f}", "y": f"{y:.2f}", "width": f"{w:.2f}", "height": f"{h:.2f}", "as": "geometry"},
        )
        return cid

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        stroke: str = "#111111",
        sw: float = 5,
        rounded: bool = False,
        prefix: str = "rect",
        extra: str = "",
    ) -> str:
        style = (
            f"rounded={1 if rounded else 0};whiteSpace=wrap;html=1;"
            f"fillColor={fill};strokeColor={stroke};strokeWidth={sw};fontFamily=Arial;"
        )
        if rounded:
            style += "absoluteArcSize=1;arcSize=44;"
        style += extra
        return self.vertex("", style, x, y, w, h, prefix)

    def text(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        value: str,
        size: int,
        align: str = "center",
        valign: str = "middle",
        color: str = "#000000",
        bold: bool = False,
        italic: bool = False,
        prefix: str = "text",
    ) -> str:
        font_style = (1 if bold else 0) + (2 if italic else 0)
        style = (
            "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;rounded=0;"
            f"align={align};verticalAlign={valign};fontFamily=Arial;fontSize={size};"
            f"fontColor={color};fontStyle={font_style};spacing=0;"
        )
        return self.vertex(value, style, x, y, w, h, prefix)

    def image(self, x: float, y: float, w: float, h: float, svg_text: str, prefix: str = "img") -> str:
        uri = "data:image/svg+xml," + quote(svg_text, safe="")
        style = (
            "shape=image;html=1;imageAspect=0;aspect=fixed;strokeColor=none;fillColor=none;"
            f"image={uri};"
        )
        return self.vertex("", style, x, y, w, h, prefix)

    def edge(
        self,
        pts: list[tuple[float, float]],
        stroke: str = "#7f858a",
        sw: float = 8,
        arrow: bool = True,
        dashed: bool = False,
        prefix: str = "edge",
    ) -> str:
        cid = self.next_id(prefix)
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;"
            f"strokeColor={stroke};strokeWidth={sw};"
        )
        style += "endArrow=classic;endFill=1;" if arrow else "endArrow=none;startArrow=none;"
        if dashed:
            style += "dashed=1;dashPattern=10 10;"
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": cid,
                "value": "",
                "style": style,
                "edge": "1",
                "parent": "1",
            },
        )
        geom = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        ET.SubElement(geom, "mxPoint", {"x": f"{pts[0][0]:.2f}", "y": f"{pts[0][1]:.2f}", "as": "sourcePoint"})
        ET.SubElement(geom, "mxPoint", {"x": f"{pts[-1][0]:.2f}", "y": f"{pts[-1][1]:.2f}", "as": "targetPoint"})
        if len(pts) > 2:
            arr = ET.SubElement(geom, "Array", {"as": "points"})
            for x, y in pts[1:-1]:
                ET.SubElement(arr, "mxPoint", {"x": f"{x:.2f}", "y": f"{y:.2f}"})
        return cid

    def line(self, pts: list[tuple[float, float]], stroke: str = "#111111", sw: float = 6, dashed: bool = False) -> str:
        cid = self.next_id("line")
        style = f"rounded=0;html=1;strokeColor={stroke};strokeWidth={sw};endArrow=none;startArrow=none;"
        if dashed:
            style += "dashed=1;dashPattern=10 10;"
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": cid,
                "value": "",
                "style": style,
                "edge": "1",
                "parent": "1",
            },
        )
        geom = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        ET.SubElement(geom, "mxPoint", {"x": f"{pts[0][0]:.2f}", "y": f"{pts[0][1]:.2f}", "as": "sourcePoint"})
        ET.SubElement(geom, "mxPoint", {"x": f"{pts[-1][0]:.2f}", "y": f"{pts[-1][1]:.2f}", "as": "targetPoint"})
        if len(pts) > 2:
            arr = ET.SubElement(geom, "Array", {"as": "points"})
            for x, y in pts[1:-1]:
                ET.SubElement(arr, "mxPoint", {"x": f"{x:.2f}", "y": f"{y:.2f}"})
        return cid

    def save(self, path: Path) -> None:
        raw = ET.tostring(self.mxfile, encoding="utf-8")
        parsed = minidom.parseString(raw)
        path.write_text(parsed.toprettyxml(indent="  "), encoding="utf-8")


def color_mix(c1: str, c2: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    a = tuple(int(c1[i : i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(c2[i : i + 2], 16) for i in (1, 3, 5))
    vals = [round(a[i] + (b[i] - a[i]) * t) for i in range(3)]
    return "#" + "".join(f"{v:02x}" for v in vals)


def vector_bar(d: Diagram, x: float, y: float, w: float, h: float, colors: list[str], base: str) -> None:
    d.rect(x, y, w, h, base, "#111111", 5, False, "bar-bg")
    step = w / len(colors)
    for i, color in enumerate(colors):
        d.rect(x + i * step, y, step, h, color, color, 0, False, "bar-seg")
    d.rect(x, y, w, h, "none", "#111111", 5, False, "bar-border", "fillColor=none;")


def mini_grid(
    d: Diagram,
    x: float,
    y: float,
    cols: int,
    rows: int,
    cell: float,
    gap: float,
    colors: list[list[str]],
    stroke: str = "#111111",
    sw: float = 4,
) -> None:
    for r in range(rows):
        for c in range(cols):
            d.rect(x + c * (cell + gap), y + r * (cell + gap), cell, cell, colors[r][c], stroke, sw, False, "grid")


def heatmap(
    d: Diagram,
    x: float,
    y: float,
    w: float,
    h: float,
    rows: int,
    cols: int,
    base: str,
    hot: str,
    peaks: list[tuple[float, float, float, float]],
    stroke: str | None = None,
) -> None:
    cell_w = w / cols
    cell_h = h / rows
    for r in range(rows):
        for c in range(cols):
            val = 0.0
            for px, py, amp, sig in peaks:
                dx = c - px
                dy = r - py
                val += amp * math.exp(-(dx * dx + dy * dy) / (2 * sig * sig))
            color = color_mix(base, hot, min(1.0, val))
            d.rect(
                x + c * cell_w,
                y + r * cell_h,
                cell_w + 0.4,
                cell_h + 0.4,
                color,
                stroke or color,
                0 if stroke is None else 1,
                False,
                "heat",
            )
    d.rect(x, y, w, h, "none", "#111111", 5, False, "heat-border", "fillColor=none;")


def patch_feature_stack(d: Diagram, x: float, y: float) -> None:
    for off in [(0, 0), (36, 34), (72, 68)]:
        colors = [["#a9b8cc" for _ in range(4)] for _ in range(4)]
        mini_grid(d, x + off[0], y + off[1], 4, 4, 100, 10, colors, "#4c5663", 5)
    colors = [
        ["#9fb0c6", "#b4c0cf", "#8ca0ba", "#adbac8"],
        ["#aab8c7", "#c0cad5", "#a2b1c2", "#b7c2cf"],
        ["#96a8bf", "#a5b5c8", "#b4c0ce", "#94a7bf"],
        ["#b1bfcd", "#a7b5c6", "#98abc2", "#b9c4d0"],
    ]
    mini_grid(d, x + 108, y + 102, 4, 4, 100, 10, colors, "#111111", 6)


def selected_evidence(d: Diagram, x: float, y: float) -> None:
    colors = [
        ["#72a4e4", "#80aee9", "#d8dee9", "#eef1f6"],
        ["#7eafe9", "#72a4e0", "#eef1f6", "#d3d9e4"],
        ["#edf0f5", "#eef1f6", "#ee5738", "#f06042"],
        ["#eef1f6", "#f26245", "#f15a3f", "#f5d7ce"],
    ]
    mini_grid(d, x, y, 4, 4, 98, 18, colors, "#9aa1a9", 4)


def wavelet_map(d: Diagram, x: float, y: float, w: float, h: float) -> None:
    random.seed(7)
    rows = cols = 14
    cell_w = w / cols
    cell_h = h / rows
    for r in range(rows):
        for c in range(cols):
            val = 0.15 + 0.35 * random.random()
            val += 0.55 * math.exp(-((c - 11) ** 2 + (r - 10) ** 2) / 11)
            val += 0.35 * math.exp(-((c - 5) ** 2 + (r - 3) ** 2) / 9)
            color = color_mix("#0b7b32", "#f2fff0", min(1.0, val))
            d.rect(x + c * cell_w, y + r * cell_h, cell_w + 0.5, cell_h + 0.5, color, color, 0, False, "wave")
    d.rect(x, y, w, h, "none", "#111111", 5, False, "wave-border", "fillColor=none;")
    d.line([(x + w - 96, y + h - 36), (x + w - 8, y + h - 124)], "#b8b8b8", 10)
    d.line([(x + w - 8, y + h - 36), (x + w - 96, y + h - 124)], "#b8b8b8", 10)


def make_diagram() -> None:
    assets = make_svg_assets()
    d = Diagram()

    # Background panels.
    d.rect(0, 0, 2350, 1220, "#edf6fc", "none", 0, True, "panel")
    d.rect(0, 1275, 2350, 1065, "#f0f1f1", "none", 0, True, "panel")
    d.rect(2345, 0, 2315, 2340, "#eff7eb", "none", 0, True, "panel")
    d.rect(4670, 0, 2114, 210, "#fcecdf", "none", 0, True, "panel")
    d.rect(4410, 1030, 1470, 1310, "#fbe8dd", "none", 0, True, "panel")
    d.rect(6160, 245, 615, 1035, "#efe3f4", "none", 0, True, "panel")
    d.rect(6160, 1430, 615, 905, "#eeeeee", "none", 0, True, "panel")

    # Titles.
    d.text(450, 42, 1450, 110, "Frozen CLIP inputs", 108)
    d.text(2670, 42, 1680, 110, "Per-image evidence construction", 108)
    d.text(4940, 42, 1660, 110, "Prototype Calibration outputs", 98)
    d.text(655, 2390, 1240, 90, "Frozen CLIP inputs", 92)
    d.text(2570, 2390, 1650, 90, "Per-image evidence construction", 82)
    d.text(4685, 2390, 1320, 90, "Prototype Calibration", 90)
    d.text(6165, 2390, 600, 90, "Output heads", 80)

    # Text CLIP inputs.
    token_w, token_h, token_gap = 460, 170, 30
    token_xs = [50, 50 + token_w + token_gap, 50 + 2 * (token_w + token_gap)]
    for i, label in enumerate(["normal", "object", "[ctx]"]):
        d.rect(token_xs[i], 440, token_w, token_h, "#e5f2fb", "#111111", 6, True, "token")
        d.text(token_xs[i], 460, token_w, 130, label, 84)
    for i, label in enumerate(["abnormal", "object", "[ctx]"]):
        d.rect(token_xs[i], 770, token_w, token_h, "#ffd0d0", "#111111", 6, True, "token")
        d.text(token_xs[i], 790, token_w, 130, label, 84)
    d.line([(token_xs[-1] + token_w, 525), (1260, 525), (1260, 690)], "#8b8f94", 6)
    d.line([(token_xs[-1] + token_w, 855), (1260, 855), (1260, 690)], "#8b8f94", 6)
    d.edge([(1260, 690), (1340, 690)], "#8b8f94", 8)

    d.image(1455, 285, 95, 95, assets["lock.svg"], "lock")
    d.rect(1340, 470, 330, 500, "#ffffff", "#111111", 6, False, "encoder")
    d.rect(1390, 525, 210, 62, "#fff2c3", "#111111", 5, True, "pill")
    d.rect(1390, 645, 210, 62, "#f3d3e5", "#111111", 5, True, "pill")
    d.rect(1390, 765, 210, 62, "#fff2c3", "#111111", 5, True, "pill")
    d.edge([(1575, 505), (1505, 505), (1505, 525)], "#111111", 5)
    d.edge([(1575, 730), (1460, 730), (1460, 705)], "#111111", 5)
    d.edge([(1440, 842), (1440, 790)], "#111111", 5)
    d.edge([(1490, 842), (1490, 790)], "#111111", 5)
    d.text(1328, 982, 360, 190, "Text<br>encoder", 82)
    d.edge([(1670, 690), (1850, 690)], "#8b8f94", 8)
    d.edge([(1670, 690), (1750, 690), (1750, 535), (1850, 535)], "#8b8f94", 8)
    d.edge([(1670, 690), (1750, 690), (1750, 855), (1850, 855)], "#8b8f94", 8)
    d.text(2010, 365, 200, 95, "<i>t<sub>m</sub></i>", 92)
    d.text(2010, 690, 200, 95, "<i>t<sub>a</sub></i>", 92)
    vector_bar(d, 1850, 495, 425, 78, ["#d9e9fb", "#92b8e7", "#5c95d3", "#bad3f2", "#8fb6e5", "#467ebd"], "#d9e9fb")
    vector_bar(d, 1850, 820, 425, 78, ["#ffd6d6", "#ef8080", "#d94747", "#ffa9a9", "#f07b7b", "#c83c3c"], "#ffd6d6")
    d.text(1775, 920, 590, 185, "Text prototype<br>embedding", 78)
    d.edge([(2275, 535), (2450, 535), (2450, 775), (2550, 775)], "#8b8f94", 5, dashed=True)
    d.edge([(2275, 855), (2450, 855), (2450, 820), (2550, 820)], "#8b8f94", 5, dashed=True)

    # Visual CLIP inputs.
    d.image(75, 1545, 490, 490, assets["input_texture.svg"], "input")
    d.text(135, 2075, 360, 85, "Input", 74)
    d.edge([(565, 1790), (725, 1790)], "#8b8f94", 8)
    d.image(925, 1355, 92, 92, assets["lock.svg"], "lock")
    d.vertex(
        "Visual<br>enconwer",
        "shape=trapezoid;perimeter=trapezoidPerimeter;direction=east;whiteSpace=wrap;html=1;"
        "fillColor=#ffffff;strokeColor=#111111;strokeWidth=6;fontFamily=Arial;fontSize=82;align=center;verticalAlign=middle;",
        735,
        1475,
        490,
        620,
        "visual-encoder",
    )
    d.text(805, 2105, 360, 120, "Visual<br>encoder", 74)
    d.edge([(1225, 1790), (1390, 1790)], "#8b8f94", 8)
    patch_feature_stack(d, 1580, 1340)
    d.text(1560, 1940, 570, 85, "patch features", 74)
    d.edge([(1225, 1950), (1390, 1950), (1390, 2080), (1785, 2080)], "#8b8f94", 8)
    d.rect(1790, 2008, 145, 145, "#c8d8eb", "#111111", 6, False, "global", "shape=ellipse;")
    d.text(1965, 2035, 360, 85, "global", 72, align="left")
    d.text(1965, 2145, 360, 85, "token", 72, align="left")
    d.edge([(2070, 1600), (2400, 1755)], "#8b8f94", 8)
    d.edge([(2070, 1450), (2550, 790)], "#8b8f94", 5, dashed=True)

    d.rect(2400, 1632, 405, 405, "#ffffff", "#111111", 6, False, "haar")
    d.rect(2400, 1632, 202.5, 202.5, "#dfe7ff", "#111111", 5, False, "haar-cell")
    d.rect(2602.5, 1632, 202.5, 202.5, "#b6e790", "#111111", 5, False, "haar-cell")
    d.rect(2400, 1834.5, 202.5, 202.5, "#dfe7ff", "#111111", 5, False, "haar-cell")
    d.rect(2602.5, 1834.5, 202.5, 202.5, "#d6efb7", "#111111", 5, False, "haar-cell")
    d.text(2402, 1674, 198, 100, "LL", 70)
    d.text(2605, 1674, 198, 100, "LH", 70)
    d.text(2402, 1874, 198, 100, "HL", 70)
    d.text(2605, 1874, 198, 100, "HH", 70)
    d.text(2375, 2056, 460, 105, "Haar DWT", 76)
    d.edge([(2805, 1835), (2955, 1835)], "#8b8f94", 8)
    wavelet_map(d, 2955, 1620, 435, 435)
    d.text(2930, 2058, 520, 165, "Wavelet<br>Reliability <i>W</i>", 76)

    # Per-image evidence construction.
    d.text(2550, 655, 120, 230, "[", 180)
    d.text(2725, 655, 120, 230, "]", 180)
    for cx in [2635, 2720]:
        for cy in [690, 785, 880]:
            d.rect(cx, cy, 28, 28, "#000000", "#000000", 0, False, "dot", "shape=ellipse;")
    d.edge([(2810, 785), (2945, 785)], "#8b8f94", 8)
    heatmap(d, 2945, 545, 450, 450, 10, 10, "#edf4fb", "#0b65b1", [(2.1, 2.1, 1.0, 1.0), (6.7, 4.4, 0.9, 1.0)], "#edf4fb")
    d.text(2945, 1018, 450, 130, "Semantic<br>Cue <i>S<sub>0</sub></i>", 76)
    d.edge([(3395, 770), (3625, 770), (3625, 1280)], "#8b8f94", 8)
    d.edge([(3390, 1810), (3625, 1810), (3625, 1435)], "#8b8f94", 8)
    d.rect(3550, 1240, 160, 160, "#f7fbf5", "#111111", 7, False, "gate", "shape=ellipse;")
    d.line([(3585, 1275), (3675, 1365)], "#111111", 8)
    d.line([(3675, 1275), (3585, 1365)], "#111111", 8)
    d.text(3442, 1410, 360, 150, "Evidence<br>Gate", 72)
    d.edge([(3710, 1320), (3865, 1320)], "#8b8f94", 8)
    selected_evidence(d, 3865, 1088)
    d.text(3855, 1572, 515, 230, "Selected<br>Evidence<br>Patches", 76)

    d.edge([(4290, 1230), (4395, 1230), (4395, 975), (4440, 975)], "#8b8f94", 8)
    d.edge([(4290, 1465), (4395, 1465), (4395, 1690), (4440, 1690)], "#8b8f94", 8)
    d.text(4580, 820, 180, 80, "<i>&nu;<sub>n</sub></i>", 88)
    d.text(4580, 1500, 180, 80, "<i>&nu;<sub>a</sub></i>", 88)
    vector_bar(d, 4440, 930, 425, 82, ["#d9e9fb", "#92b8e7", "#5c95d3", "#bad3f2", "#8fb6e5", "#467ebd"], "#d9e9fb")
    vector_bar(d, 4440, 1635, 425, 82, ["#ffd0c4", "#f55f3c", "#ee7556", "#ffb5a4", "#f07b62", "#e55735"], "#ffd0c4")
    d.text(4530, 1742, 435, 265, "Visual<br>Prototype<br>embedding", 72)

    # Prototype calibration.
    d.edge([(4865, 971), (5005, 1215)], "#8b8f94", 8)
    d.edge([(4865, 1676), (5005, 1460)], "#8b8f94", 8)
    d.rect(5005, 1095, 285, 700, "#f6a5a8", "#111111", 6, True, "mixer")
    d.text(5018, 1145, 252, 115, "Mixer", 76)
    d.rect(5128, 1360, 92, 92, "#ffc3c5", "#111111", 5, False, "plus", "shape=ellipse;")
    d.text(5130, 1368, 88, 70, "+", 80)
    d.image(5168, 1570, 82, 82, assets["lock.svg"], "lock")
    d.edge([(5095, 1265), (5145, 1360)], "#111111", 5)
    d.edge([(5005, 1410), (5128, 1410)], "#8b8f94", 8)
    d.text(5125, 1770, 520, 155, "Conservative<br>Calibration", 74)
    d.text(5180, 1985, 620, 105, "<i>t &rarr; t&#771; + &nu;<sub>n</sub>,&nu;<sub>a</sub> &rarr; t&#771;<sub>n</sub>,t&#771;<sub>a</sub></i>", 62)

    d.edge([(5148, 1095), (5148, 635), (5320, 635)], "#8b8f94", 8)
    grid_colors = [["#9eabc0" for _ in range(4)] for _ in range(4)]
    mini_grid(d, 5320, 390, 4, 4, 100, 15, grid_colors, "#111111", 6)
    d.edge([(5765, 635), (5960, 635)], "#8b8f94", 8)
    d.text(5570, 975, 230, 86, "<i>t&#771;<sub>n</sub></i>", 84)
    vector_bar(d, 5425, 1130, 425, 82, ["#d9e9fb", "#92b8e7", "#5c95d3", "#bad3f2", "#8fb6e5", "#467ebd"], "#d9e9fb")
    d.text(5570, 1340, 230, 86, "<i>t&#771;<sub>a</sub></i>", 84)
    vector_bar(d, 5425, 1470, 425, 82, ["#ffd0d0", "#ef8080", "#d94747", "#ffa9a9", "#f07b7b", "#c83c3c"], "#ffd0d0")
    d.image(5772, 1340, 76, 76, assets["lock.svg"], "lock")
    d.image(5772, 1556, 76, 76, assets["lock.svg"], "lock")
    d.edge([(5290, 1290), (5425, 1171)], "#8b8f94", 8)
    d.edge([(5290, 1510), (5425, 1511)], "#8b8f94", 8)

    d.text(5880, 520, 330, 95, "grid &times; <i>t&#771;</i>", 74)
    d.rect(5960, 655, 165, 165, "#ded1ea", "#111111", 6, False, "mult", "shape=ellipse;")
    d.line([(5996, 690), (6090, 784)], "#111111", 7)
    d.line([(6090, 690), (5996, 784)], "#111111", 7)
    d.edge([(6125, 737), (6240, 737)], "#8b8f94", 8)
    d.edge([(6042, 820), (6042, 1895)], "#8b8f94", 8)
    d.edge([(5850, 1512), (5982, 1512), (5982, 1895)], "#8b8f94", 8)

    # Output heads.
    d.text(6225, 300, 470, 180, "Patch Score<br>Map", 72)
    heatmap(
        d,
        6245,
        500,
        490,
        490,
        12,
        12,
        "#3c0b62",
        "#f5d328",
        [(9.5, 5.0, 1.25, 0.75), (2.2, 1.6, 0.35, 1.5), (4.0, 2.1, 0.24, 2.0)],
        None,
    )
    d.text(6225, 1012, 500, 210, "Pixel<br>Anomaly Map", 72)
    d.rect(5970, 1855, 160, 160, "#e5e5e5", "#111111", 6, False, "score-dot", "shape=ellipse;")
    d.rect(6036, 1921, 28, 28, "#000000", "#000000", 0, False, "score-dot", "shape=ellipse;")
    d.edge([(6130, 1935), (6320, 1935)], "#8b8f94", 8)
    d.text(6335, 1545, 320, 195, "Image<br>Score", 72)
    d.image(6330, 1746, 310, 310, assets["gauge.svg"], "gauge")
    d.text(6318, 2052, 350, 210, "Image<br>Score", 72)
    d.edge([(4685, 1717), (4685, 2250), (6042, 2250), (6042, 2015)], "#8b8f94", 5, dashed=True)

    d.save(DRAWIO)
    write_docs()


def write_docs() -> None:
    (OUT_DIR / "SVG_ASSETS.md").write_text(
        """# SVG Assets

All assets are embedded into the draw.io file as data URIs and also kept here for editing.

- `lock.svg`: generic lock symbol recreated from the reference style.
- `input_texture.svg`: generic gray inspection-image placeholder with a small red anomaly spot, derived visually from the reference but not a traced crop.
- `gauge.svg`: generic image-score gauge icon recreated from the reference style.

Boxes, arrows, labels, patch grids, vector embeddings, heatmaps, and score maps are native editable draw.io cells.
""",
        encoding="utf-8",
    )
    (OUT_DIR / "LAYOUT_FINGERPRINT.md").write_text(
        """# Layout Fingerprint

- Reference: `/Users/bytedance/Downloads/image_1784538888.jpeg`
- Canvas: 6784 x 2496 px, landscape scientific workflow figure.
- Major regions:
  - Top-left Frozen CLIP text inputs: x=0, y=0, w=2350, h=1220.
  - Bottom-left Frozen CLIP visual inputs: x=0, y=1275, w=2350, h=1065.
  - Center per-image evidence construction: x=2345, y=0, w=2315, h=2340.
  - Prototype calibration: header x=4670, y=0, w=2114, h=210; main panel x=4410, y=1030, w=1470, h=1310.
  - Output heads: patch score panel x=6160, y=245, w=615, h=1035; image score panel x=6160, y=1430, w=615, h=905.
- Reading order: text tokens and visual patches feed into a central evidence gate; selected evidence creates visual prototypes; a mixer calibrates text prototypes; outputs feed patch and image score heads.
- Editable element count: approximately 260 draw.io cells, mostly native rectangles, labels, connectors, and small grid cells.
- Connector directions: left-to-right primary flow with dashed evidence links and a bottom dashed route to the image-score head.
- Known uncertainty: small heatmap values and the input texture are visually approximated; exact pixel heatmaps from the source are not embedded.
""",
        encoding="utf-8",
    )


def create_qa() -> None:
    QA_DIR.mkdir(parents=True, exist_ok=True)
    if not REFERENCE.exists() or not PNG.exists():
        raise SystemExit("Reference or exported PNG is missing; run draw.io export before QA.")
    ref = Image.open(REFERENCE).convert("RGB")
    out = Image.open(PNG).convert("RGB")
    if out.size != ref.size:
        out = out.resize(ref.size, Image.Resampling.LANCZOS)

    modules = {
        "text_clip": (0, 0, 2350, 1220),
        "visual_clip": (0, 1275, 2350, 1065),
        "evidence": (2345, 0, 2315, 2340),
        "calibration": (4410, 0, 1760, 2340),
        "outputs": (6160, 0, 624, 2340),
    }
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()

    rows = []
    for name, (x, y, w, h) in modules.items():
        ref_crop = ref.crop((x, y, x + w, y + h))
        out_crop = out.crop((x, y, x + w, y + h))
        ref_crop.save(QA_DIR / f"{name}_reference.png")
        out_crop.save(QA_DIR / f"{name}_recreated.png")
        target_w = 760
        scale = target_w / w
        target_h = max(1, int(h * scale))
        ref_s = ref_crop.resize((target_w, target_h), Image.Resampling.LANCZOS)
        out_s = out_crop.resize((target_w, target_h), Image.Resampling.LANCZOS)
        row = Image.new("RGB", (target_w * 2 + 36, target_h + 64), "white")
        draw = ImageDraw.Draw(row)
        draw.text((0, 10), f"{name} reference", fill=(0, 0, 0), font=font)
        draw.text((target_w + 36, 10), f"{name} recreated", fill=(0, 0, 0), font=font)
        row.paste(ref_s, (0, 56))
        row.paste(out_s, (target_w + 36, 56))
        rows.append(row)

    contact_w = max(r.width for r in rows)
    contact_h = sum(r.height for r in rows) + 18 * (len(rows) - 1)
    contact = Image.new("RGB", (contact_w, contact_h), "white")
    y = 0
    for row in rows:
        contact.paste(row, (0, y))
        y += row.height + 18
    contact.save(QA_DIR / "module_contact_sheet.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", action="store_true", help="Create QA crops after PNG export.")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    if args.qa:
        create_qa()
    else:
        make_diagram()


if __name__ == "__main__":
    main()
