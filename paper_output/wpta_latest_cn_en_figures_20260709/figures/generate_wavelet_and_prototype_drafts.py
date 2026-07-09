#!/usr/bin/env python3
"""Generate draft paper figures for wavelet mechanism and prototype dynamics.

The outputs are illustrative drafts for paper-figure discussion. They are not
reported experimental measurements.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from PIL import Image, ImageFilter


ROOT = Path("/Users/bytedance/Documents/Codex/2026-07-07/yon")
OUT_DIR = ROOT / "outputs/figures"
INPUT_IMAGE = OUT_DIR / "figure1_panels/panel_01_input.png"

WAVELET_PNG = OUT_DIR / "draft_figure2_wavelet_mechanism.png"
WAVELET_SVG = OUT_DIR / "draft_figure2_wavelet_mechanism.svg"
WAVELET_PDF = OUT_DIR / "draft_figure2_wavelet_mechanism.pdf"

PROTO_PNG = OUT_DIR / "draft_figure4_prototype_dynamics.png"
PROTO_SVG = OUT_DIR / "draft_figure4_prototype_dynamics.svg"
PROTO_PDF = OUT_DIR / "draft_figure4_prototype_dynamics.pdf"
CONTACT_SHEET = OUT_DIR / "draft_figures_2_and_4_preview.png"


COLORS = {
    "ink": "#252A31",
    "muted": "#667085",
    "line": "#98A2B3",
    "panel": "#F8FAFC",
    "blue": "#1F77B4",
    "blue_light": "#DCEEFF",
    "orange": "#D55E00",
    "orange_light": "#FFE8D8",
    "green": "#009E73",
    "green_light": "#DFF5EC",
    "purple": "#7B61FF",
    "purple_light": "#ECE8FF",
    "red": "#D62728",
    "red_light": "#FFE5E5",
    "yellow": "#F6C945",
    "yellow_light": "#FFF7D6",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_input_image() -> Image.Image:
    img = Image.open(INPUT_IMAGE).convert("RGB")
    side = min(img.size)
    left = (img.width - side) // 2
    top = (img.height - side) // 2
    return img.crop((left, top, left + side, top + side)).resize((256, 256))


def haar_dwt(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gray = gray[: gray.shape[0] // 2 * 2, : gray.shape[1] // 2 * 2]
    a = gray[0::2, 0::2]
    b = gray[0::2, 1::2]
    c = gray[1::2, 0::2]
    d = gray[1::2, 1::2]
    ll = (a + b + c + d) / 4.0
    lh = (a - b + c - d) / 4.0
    hl = (a + b - c - d) / 4.0
    hh = (a - b - c + d) / 4.0
    return ll, lh, hl, hh


def norm01(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    lo, hi = np.percentile(x, [2, 98])
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    label: str,
    *,
    fc: str,
    ec: str,
    lw: float = 1.8,
    fontsize: int = 10,
    weight: str = "bold",
    radius: float = 0.06,
) -> patches.FancyBboxPatch:
    box = patches.FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        linewidth=lw,
        facecolor=fc,
        edgecolor=ec,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + wh[0] / 2,
        xy[1] + wh[1] / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["ink"],
        weight=weight,
        zorder=3,
    )
    return box


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#667085",
    lw: float = 1.8,
    rad: float = 0.0,
    text: str | None = None,
    txy: tuple[float, float] | None = None,
    linestyle: str = "-",
) -> None:
    patch = patches.FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(patch)
    if text:
        tx, ty = txy if txy is not None else ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(tx, ty, text, ha="center", va="center", fontsize=8.5, color=color, zorder=4)


def image_box(
    fig: plt.Figure,
    parent_ax: plt.Axes,
    bbox: tuple[float, float, float, float],
    img: np.ndarray | Image.Image,
    title: str,
    *,
    cmap: str | None = None,
    border: str = "#98A2B3",
    title_color: str = "#252A31",
) -> plt.Axes:
    ax = fig.add_axes(bbox)
    ax.imshow(img, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(border)
        spine.set_linewidth(1.2)
    parent_ax.text(
        bbox[0] + bbox[2] / 2,
        bbox[1] + bbox[3] + 0.018,
        title,
        transform=fig.transFigure,
        ha="center",
        va="bottom",
        fontsize=9.5,
        weight="bold",
        color=title_color,
    )
    return ax


def draw_wavelet_mechanism() -> None:
    img = load_input_image()
    gray = np.asarray(img.convert("L"), dtype=float) / 255.0
    ll, lh, hl, hh = haar_dwt(gray)
    hf_energy = norm01(np.sqrt(lh**2 + hl**2 + hh**2))
    hf_energy = np.array(Image.fromarray((hf_energy * 255).astype(np.uint8)).resize((256, 256), Image.Resampling.BICUBIC))
    hf_energy = np.asarray(Image.fromarray(hf_energy).filter(ImageFilter.GaussianBlur(radius=2))) / 255.0

    fig = plt.figure(figsize=(15.5, 6.2), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.035, 0.955, "Draft Fig. 2: Wavelet-guided prototype adaptation mechanism", fontsize=16, weight="bold", color=COLORS["ink"])
    ax.text(
        0.035,
        0.915,
        "High-frequency wavelet evidence gates which visual tokens are allowed to update CLIP text-side prototypes.",
        fontsize=10.5,
        color=COLORS["muted"],
    )
    ax.text(0.86, 0.955, "Illustrative draft", fontsize=9.5, color=COLORS["muted"], ha="right")

    image_box(fig, ax, (0.045, 0.41, 0.15, 0.38), img, "Test image")
    add_box(ax, (0.24, 0.54), (0.10, 0.12), "CLIP\nimage encoder", fc=COLORS["blue_light"], ec=COLORS["blue"], fontsize=10)
    add_box(ax, (0.38, 0.54), (0.10, 0.12), "Patch\nfeatures", fc="#FFFFFF", ec=COLORS["line"], fontsize=10)
    add_box(ax, (0.52, 0.54), (0.10, 0.12), "Haar DWT\non feature map", fc=COLORS["purple_light"], ec=COLORS["purple"], fontsize=9.8)

    arrow(ax, (0.20, 0.60), (0.24, 0.60))
    arrow(ax, (0.34, 0.60), (0.38, 0.60))
    arrow(ax, (0.48, 0.60), (0.52, 0.60))

    sub_bboxes = [
        ((0.665, 0.665, 0.095, 0.18), norm01(ll), "LL\nsemantic base", "gray"),
        ((0.785, 0.665, 0.095, 0.18), norm01(np.abs(lh)), "LH\nhorizontal detail", "magma"),
        ((0.665, 0.405, 0.095, 0.18), norm01(np.abs(hl)), "HL\nvertical detail", "magma"),
        ((0.785, 0.405, 0.095, 0.18), norm01(np.abs(hh)), "HH\ncorner residual", "magma"),
    ]
    for bbox, arr, title, cmap in sub_bboxes:
        image_box(fig, ax, bbox, arr, title, cmap=cmap, border=COLORS["purple"])

    arrow(ax, (0.62, 0.60), (0.665, 0.755), color=COLORS["purple"])
    arrow(ax, (0.62, 0.60), (0.785, 0.755), color=COLORS["purple"])
    arrow(ax, (0.62, 0.60), (0.665, 0.495), color=COLORS["purple"])
    arrow(ax, (0.62, 0.60), (0.785, 0.495), color=COLORS["purple"])

    image_box(fig, ax, (0.075, 0.105, 0.14, 0.23), hf_energy, "High-frequency evidence", cmap="inferno", border=COLORS["orange"])
    add_box(ax, (0.265, 0.16), (0.13, 0.11), "Wavelet\nreliability gate", fc=COLORS["orange_light"], ec=COLORS["orange"], fontsize=10)
    add_box(ax, (0.46, 0.16), (0.13, 0.11), "Gated token\naggregation", fc=COLORS["green_light"], ec=COLORS["green"], fontsize=10)
    add_box(ax, (0.665, 0.18), (0.105, 0.095), "Normal\nprototype", fc=COLORS["blue_light"], ec=COLORS["blue"], fontsize=9.5)
    add_box(ax, (0.795, 0.18), (0.105, 0.095), "Abnormal\nprototype", fc=COLORS["red_light"], ec=COLORS["red"], fontsize=9.5)
    add_box(ax, (0.72, 0.065), (0.12, 0.07), "Similarity map", fc="#FFFFFF", ec=COLORS["line"], fontsize=9.5)

    arrow(ax, (0.145, 0.41), (0.145, 0.34), color=COLORS["orange"], text="DWT detail", txy=(0.19, 0.37), rad=-0.2)
    arrow(ax, (0.215, 0.22), (0.265, 0.215), color=COLORS["orange"])
    arrow(ax, (0.395, 0.215), (0.46, 0.215), color=COLORS["green"], text="mask unreliable patches", txy=(0.43, 0.265))
    arrow(ax, (0.59, 0.215), (0.665, 0.23), color=COLORS["green"])
    arrow(ax, (0.59, 0.215), (0.795, 0.23), color=COLORS["green"])
    arrow(ax, (0.717, 0.18), (0.755, 0.135), color=COLORS["line"])
    arrow(ax, (0.848, 0.18), (0.805, 0.135), color=COLORS["line"])

    ax.text(0.43, 0.08, "Key message: adaptation is not a serial post-processing step; it branches back to CLIP prototypes.", fontsize=10.5, color=COLORS["ink"])

    fig.savefig(WAVELET_PNG, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(WAVELET_SVG, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(WAVELET_PDF, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def sample_cluster(rng: np.random.Generator, center: tuple[float, float], n: int, scale: tuple[float, float]) -> np.ndarray:
    pts = rng.normal(0, 1, size=(n, 2))
    pts[:, 0] *= scale[0]
    pts[:, 1] *= scale[1]
    return pts + np.array(center)


def scatter_panel(ax: plt.Axes, mode: str) -> None:
    rng = np.random.default_rng(7)
    normal = sample_cluster(rng, (-0.55, 0.05), 44, (0.18, 0.13))
    abnormal = sample_cluster(rng, (0.72, 0.45), 16, (0.12, 0.11))
    boundary = sample_cluster(rng, (0.23, -0.35), 12, (0.15, 0.11))

    ax.scatter(normal[:, 0], normal[:, 1], s=28, c=COLORS["blue_light"], edgecolor=COLORS["blue"], linewidth=0.8, label="normal patches")
    ax.scatter(abnormal[:, 0], abnormal[:, 1], s=36, c=COLORS["red_light"], edgecolor=COLORS["red"], linewidth=0.9, marker="^", label="abnormal patches")
    ax.scatter(boundary[:, 0], boundary[:, 1], s=30, c=COLORS["yellow_light"], edgecolor="#C58A00", linewidth=0.8, marker="s", label="uncertain/high-frequency patches")

    p0 = np.array([-0.42, -0.42])
    if mode == "direct":
        p1 = np.array([0.13, -0.05])
        title = "Direct TTA: prototype drift"
        color = COLORS["orange"]
        note = "abnormal evidence may be absorbed"
    else:
        p1 = np.array([-0.50, -0.18])
        title = "Ours: wavelet-gated update"
        color = COLORS["green"]
        note = "unreliable high-frequency tokens suppressed"

    ax.scatter([p0[0]], [p0[1]], s=115, c="#FFFFFF", edgecolor=COLORS["ink"], linewidth=1.6, marker="D", zorder=5)
    ax.text(p0[0] - 0.03, p0[1] - 0.16, "initial\nprototype", fontsize=8, ha="center", color=COLORS["ink"])
    ax.scatter([p1[0]], [p1[1]], s=145, c=color, edgecolor=COLORS["ink"], linewidth=1.2, marker="D", zorder=5)
    ax.text(p1[0] + 0.03, p1[1] + 0.13, "adapted\nprototype", fontsize=8, ha="center", color=COLORS["ink"])
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2))

    if mode == "direct":
        ax.annotate("leakage", xy=(0.53, 0.34), xytext=(0.12, 0.64), fontsize=8.5, color=COLORS["orange"], arrowprops=dict(arrowstyle="-|>", color=COLORS["orange"], lw=1.5))
    else:
        ax.add_patch(patches.Ellipse((0.23, -0.35), 0.58, 0.38, facecolor="none", edgecolor=COLORS["green"], linewidth=1.5, linestyle="--"))
        ax.text(0.28, -0.68, "down-weighted by gate", fontsize=8.5, color=COLORS["green"], ha="center")

    ax.set_title(title, fontsize=11, weight="bold", color=COLORS["ink"], pad=8)
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8.5, color=COLORS["muted"], ha="left", va="bottom")
    ax.set_xlim(-1.05, 1.15)
    ax.set_ylim(-0.9, 0.95)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_color(COLORS["line"])
        spine.set_linewidth(1.0)


def draw_similarity_curves(ax: plt.Axes) -> None:
    steps = np.arange(0, 7)
    direct_normal = np.array([0.72, 0.70, 0.67, 0.64, 0.61, 0.59, 0.57])
    direct_abn = np.array([0.34, 0.41, 0.48, 0.55, 0.60, 0.63, 0.65])
    ours_normal = np.array([0.72, 0.73, 0.735, 0.74, 0.745, 0.748, 0.75])
    ours_abn = np.array([0.34, 0.35, 0.355, 0.36, 0.365, 0.37, 0.372])

    ax.plot(steps, direct_normal, color=COLORS["orange"], lw=2.3, marker="o", label="Direct: sim(proto, normal)")
    ax.plot(steps, direct_abn, color=COLORS["orange"], lw=2.3, marker="^", linestyle="--", label="Direct: sim(proto, abnormal)")
    ax.plot(steps, ours_normal, color=COLORS["green"], lw=2.3, marker="o", label="Ours: sim(proto, normal)")
    ax.plot(steps, ours_abn, color=COLORS["green"], lw=2.3, marker="^", linestyle="--", label="Ours: sim(proto, abnormal)")
    ax.fill_between(steps, direct_abn, direct_normal, where=direct_normal >= direct_abn, color=COLORS["orange"], alpha=0.08)
    ax.fill_between(steps, ours_abn, ours_normal, where=ours_normal >= ours_abn, color=COLORS["green"], alpha=0.08)
    ax.set_title("Expected adaptation behavior", fontsize=11, weight="bold", color=COLORS["ink"], pad=8)
    ax.set_xlabel("test-time update step", fontsize=9.5)
    ax.set_ylabel("cosine similarity", fontsize=9.5)
    ax.set_ylim(0.28, 0.80)
    ax.set_xlim(0, 6)
    ax.grid(True, axis="y", color="#EAECF0", linewidth=0.8)
    ax.tick_params(labelsize=8.5)
    ax.legend(loc="lower left", fontsize=7.7, frameon=False)
    ax.text(3.55, 0.665, "drift risk", fontsize=9, color=COLORS["orange"])
    ax.text(3.65, 0.438, "stable margin", fontsize=9, color=COLORS["green"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["line"])


def draw_prototype_dynamics() -> None:
    fig = plt.figure(figsize=(15.5, 5.4), dpi=200)
    fig.text(0.035, 0.945, "Draft Fig. 4: Prototype adaptation dynamics", fontsize=16, weight="bold", color=COLORS["ink"])
    fig.text(
        0.035,
        0.895,
        "A diagnostic view for showing why wavelet gating makes test-time prototype updates less vulnerable to semantic drift.",
        fontsize=10.5,
        color=COLORS["muted"],
    )
    fig.text(0.855, 0.945, "Illustrative draft, not measured data", fontsize=9.5, color=COLORS["muted"], ha="right")

    ax1 = fig.add_axes((0.045, 0.14, 0.27, 0.66))
    ax2 = fig.add_axes((0.36, 0.14, 0.27, 0.66))
    ax3 = fig.add_axes((0.685, 0.14, 0.27, 0.66))
    scatter_panel(ax1, "direct")
    scatter_panel(ax2, "ours")
    draw_similarity_curves(ax3)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.34, 0.035), ncol=3, frameon=False, fontsize=8.5)

    fig.savefig(PROTO_PNG, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(PROTO_SVG, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(PROTO_PDF, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def draw_contact_sheet() -> None:
    panels = [
        ("Figure 2 draft: wavelet mechanism", Image.open(WAVELET_PNG).convert("RGB")),
        ("Figure 4 draft: prototype dynamics", Image.open(PROTO_PNG).convert("RGB")),
    ]
    target_w = 1600
    gap = 36
    title_h = 64
    pad = 36

    resized = []
    for title, img in panels:
        scale = target_w / img.width
        new_h = int(img.height * scale)
        resized.append((title, img.resize((target_w, new_h), Image.Resampling.LANCZOS)))

    total_h = pad + sum(title_h + img.height for _, img in resized) + gap * (len(resized) - 1) + pad
    sheet = Image.new("RGB", (target_w + pad * 2, total_h), "#F8FAFC")

    y = pad
    for title, img in resized:
        sheet.paste(img, (pad, y + title_h))
        y += title_h + img.height + gap

    # Keep the preview as a bitmap contact sheet; source figures remain editable SVG/PDF.
    import PIL.ImageDraw
    import PIL.ImageFont

    draw = PIL.ImageDraw.Draw(sheet)
    try:
        font = PIL.ImageFont.truetype("Arial.ttf", 28)
    except OSError:
        font = PIL.ImageFont.load_default()

    y = pad + 16
    for title, img in resized:
        draw.text((pad, y), title, fill=COLORS["ink"], font=font)
        y += title_h + img.height + gap

    sheet.save(CONTACT_SHEET)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_wavelet_mechanism()
    draw_prototype_dynamics()
    draw_contact_sheet()
    print(WAVELET_PNG)
    print(PROTO_PNG)
    print(CONTACT_SHEET)


if __name__ == "__main__":
    main()
