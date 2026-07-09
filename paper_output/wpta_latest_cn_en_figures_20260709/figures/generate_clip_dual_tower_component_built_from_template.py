#!/usr/bin/env python3
"""Build a one-slide CLIP dual-tower comparison figure from template components.

The script intentionally clones native PowerPoint XML elements from the supplied
template deck, then only changes position, text, and image relationships.
"""

from __future__ import annotations

import copy
import html
import os
import shutil
import uuid
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path("/Users/bytedance/Documents/Codex/2026-07-07/yon")
TEMPLATE_PPTX = Path(
    "/Users/bytedance/Library/Containers/com.tencent.xinWeChat/Data/Documents/"
    "xwechat_files/wxid_yu3x18colyme12_9ec3/msg/file/2026-07/模板合集20260627.pptx"
)
ASSET_DIR = ROOT / "outputs/figures/mvtec_three_row_selection_assets"
OUT_DIR = ROOT / "outputs/figures"
OUT_PPTX = OUT_DIR / "figure4_clip_dual_tower_component_built_from_template.pptx"
AUDIT = OUT_DIR / "figure4_clip_dual_tower_component_built_from_template.audit.md"
WORK = Path("/private/tmp/codex-presentations/manual-component-built/component_pptx_build")

P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
A16_NS = "http://schemas.microsoft.com/office/drawing/2014/main"

ET.register_namespace("p", P_NS)
ET.register_namespace("a", A_NS)
ET.register_namespace("r", R_NS)
ET.register_namespace("a16", A16_NS)

NS = {"p": P_NS, "a": A_NS, "r": R_NS}


def q(ns: str, name: str) -> str:
    return f"{{{ns}}}{name}"


def parse_xml(path: Path) -> ET.ElementTree:
    return ET.parse(path)


def find_by_cnv_id(root: ET.Element, cnv_id: str) -> ET.Element:
    sp_tree = root.find(".//p:spTree", NS)
    if sp_tree is None:
        raise ValueError("slide has no spTree")
    for elem in sp_tree:
        cnv = elem.find(".//p:cNvPr", NS)
        if cnv is not None and cnv.attrib.get("id") == str(cnv_id):
            return elem
    raise KeyError(f"cannot find cNvPr id={cnv_id}")


def get_sp_tree(root: ET.Element) -> ET.Element:
    sp_tree = root.find(".//p:spTree", NS)
    if sp_tree is None:
        raise ValueError("slide has no spTree")
    return sp_tree


def clear_slide(root: ET.Element) -> ET.Element:
    sp_tree = get_sp_tree(root)
    for child in list(sp_tree)[2:]:
        sp_tree.remove(child)
    return sp_tree


def update_creation_id(elem: ET.Element) -> None:
    for creation in elem.findall(".//a16:creationId", {"a16": A16_NS}):
        creation.set("id", "{" + str(uuid.uuid4()).upper() + "}")


def set_cnv(elem: ET.Element, shape_id: int, name: str) -> None:
    cnv = elem.find(".//p:cNvPr", NS)
    if cnv is None:
        raise ValueError("element has no cNvPr")
    cnv.set("id", str(shape_id))
    cnv.set("name", name)
    update_creation_id(elem)


def set_xfrm(elem: ET.Element, x: int, y: int, w: int, h: int, keep_rotation: bool = False) -> None:
    xfrm = elem.find(".//a:xfrm", NS)
    if xfrm is None:
        sp_pr = elem.find(".//p:spPr", NS)
        if sp_pr is None:
            raise ValueError("element has no spPr")
        xfrm = ET.Element(q(A_NS, "xfrm"))
        sp_pr.insert(0, xfrm)
    if not keep_rotation:
        for attr in ["rot", "flipH", "flipV"]:
            xfrm.attrib.pop(attr, None)
    off = xfrm.find("a:off", NS)
    if off is None:
        off = ET.SubElement(xfrm, q(A_NS, "off"))
    ext = xfrm.find("a:ext", NS)
    if ext is None:
        ext = ET.SubElement(xfrm, q(A_NS, "ext"))
    off.set("x", str(int(x)))
    off.set("y", str(int(y)))
    ext.set("cx", str(int(w)))
    ext.set("cy", str(int(h)))


def set_solid_fill(elem: ET.Element, color: str | None) -> None:
    if color is None:
        return
    color = color.strip("#").upper()
    sp_pr = elem.find(".//p:spPr", NS)
    if sp_pr is None:
        return
    for child in list(sp_pr):
        if child.tag in {q(A_NS, "solidFill"), q(A_NS, "noFill")}:
            sp_pr.remove(child)
    geom = sp_pr.find("a:prstGeom", NS)
    insert_at = list(sp_pr).index(geom) + 1 if geom is not None and geom in list(sp_pr) else len(list(sp_pr))
    solid = ET.Element(q(A_NS, "solidFill"))
    srgb = ET.SubElement(solid, q(A_NS, "srgbClr"))
    srgb.set("val", color)
    sp_pr.insert(insert_at, solid)


def set_line_color(elem: ET.Element, color: str | None, width: int | None = None) -> None:
    if color is None and width is None:
        return
    sp_pr = elem.find(".//p:spPr", NS)
    if sp_pr is None:
        return
    ln = sp_pr.find("a:ln", NS)
    if ln is None:
        ln = ET.SubElement(sp_pr, q(A_NS, "ln"))
    if width is not None:
        ln.set("w", str(int(width)))
    if color is not None:
        color = color.strip("#").upper()
        for child in list(ln):
            if child.tag in {q(A_NS, "solidFill"), q(A_NS, "noFill")}:
                ln.remove(child)
        solid = ET.Element(q(A_NS, "solidFill"))
        srgb = ET.SubElement(solid, q(A_NS, "srgbClr"))
        srgb.set("val", color)
        ln.insert(0, solid)


def set_text(
    elem: ET.Element,
    text: str | None,
    *,
    size: int = 1900,
    bold: bool = False,
    color: str = "222222",
    align: str = "ctr",
    font: str = "Arial",
    anchor: str = "ctr",
) -> None:
    if text is None:
        return
    tx_body = elem.find("p:txBody", NS)
    if tx_body is None:
        return
    body_pr = tx_body.find("a:bodyPr", NS)
    if body_pr is None:
        body_pr = ET.Element(q(A_NS, "bodyPr"))
        tx_body.insert(0, body_pr)
    body_pr.set("anchor", anchor)
    body_pr.set("rtlCol", "0")
    body_pr.set("wrap", "square")
    lst = tx_body.find("a:lstStyle", NS)
    for child in list(tx_body):
        if child.tag == q(A_NS, "p"):
            tx_body.remove(child)
    if lst is None:
        lst = ET.SubElement(tx_body, q(A_NS, "lstStyle"))
    lines = text.split("\n")
    for line in lines:
        p = ET.SubElement(tx_body, q(A_NS, "p"))
        p_pr = ET.SubElement(p, q(A_NS, "pPr"))
        p_pr.set("algn", align)
        r = ET.SubElement(p, q(A_NS, "r"))
        r_pr = ET.SubElement(r, q(A_NS, "rPr"))
        r_pr.set("lang", "en-US")
        r_pr.set("altLang", "zh-CN")
        r_pr.set("sz", str(size))
        if bold:
            r_pr.set("b", "1")
        solid = ET.SubElement(r_pr, q(A_NS, "solidFill"))
        srgb = ET.SubElement(solid, q(A_NS, "srgbClr"))
        srgb.set("val", color.strip("#").upper())
        ET.SubElement(r_pr, q(A_NS, "latin")).set("typeface", font)
        ET.SubElement(r_pr, q(A_NS, "ea")).set("typeface", "Microsoft YaHei")
        ET.SubElement(r_pr, q(A_NS, "cs")).set("typeface", font)
        t = ET.SubElement(r, q(A_NS, "t"))
        t.text = line


def clone_shape(
    sp_tree: ET.Element,
    template: ET.Element,
    next_id: list[int],
    name: str,
    x: int,
    y: int,
    w: int,
    h: int,
    text: str | None = None,
    *,
    fill: str | None = None,
    line: str | None = None,
    line_w: int | None = None,
    font_size: int = 1900,
    bold: bool = False,
    text_color: str = "222222",
    align: str = "ctr",
    anchor: str = "ctr",
) -> ET.Element:
    elem = copy.deepcopy(template)
    set_cnv(elem, next_id[0], name)
    next_id[0] += 1
    set_xfrm(elem, x, y, w, h)
    set_solid_fill(elem, fill)
    set_line_color(elem, line, line_w)
    set_text(elem, text, size=font_size, bold=bold, color=text_color, align=align, anchor=anchor)
    sp_tree.append(elem)
    return elem


def clone_pic(
    sp_tree: ET.Element,
    template: ET.Element,
    next_id: list[int],
    name: str,
    rel_id: str,
    x: int,
    y: int,
    w: int,
    h: int,
) -> ET.Element:
    elem = copy.deepcopy(template)
    set_cnv(elem, next_id[0], name)
    next_id[0] += 1
    set_xfrm(elem, x, y, w, h)
    blip = elem.find(".//a:blip", NS)
    if blip is None:
        raise ValueError("picture has no blip")
    blip.set(q(R_NS, "embed"), rel_id)
    sp_tree.append(elem)
    return elem


def clone_connector(
    sp_tree: ET.Element,
    template: ET.Element,
    next_id: list[int],
    name: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    color: str | None = None,
    width: int | None = 28575,
    dashed: bool = False,
) -> ET.Element:
    elem = copy.deepcopy(template)
    set_cnv(elem, next_id[0], name)
    next_id[0] += 1
    x = min(x1, x2)
    y = min(y1, y2)
    w = max(abs(x2 - x1), 1)
    h = max(abs(y2 - y1), 1)
    xfrm = elem.find(".//a:xfrm", NS)
    if xfrm is None:
        raise ValueError("connector has no xfrm")
    for attr in ["rot", "flipH", "flipV"]:
        xfrm.attrib.pop(attr, None)
    if x2 < x1:
        xfrm.set("flipH", "1")
    if y2 < y1:
        xfrm.set("flipV", "1")
    off = xfrm.find("a:off", NS)
    ext = xfrm.find("a:ext", NS)
    if off is None or ext is None:
        raise ValueError("connector xfrm incomplete")
    off.set("x", str(x))
    off.set("y", str(y))
    ext.set("cx", str(w))
    ext.set("cy", str(h))

    # Remove stale endpoint bindings from cloned connectors.
    c_nv = elem.find(".//p:cNvCxnSpPr", NS)
    if c_nv is not None:
        for child in list(c_nv):
            if child.tag in {q(A_NS, "stCxn"), q(A_NS, "endCxn")}:
                c_nv.remove(child)
    ln = elem.find(".//a:ln", NS)
    if ln is not None:
        if width is not None:
            ln.set("w", str(width))
        if color is not None:
            for child in list(ln):
                if child.tag in {q(A_NS, "solidFill"), q(A_NS, "noFill")}:
                    ln.remove(child)
            solid = ET.Element(q(A_NS, "solidFill"))
            srgb = ET.SubElement(solid, q(A_NS, "srgbClr"))
            srgb.set("val", color.strip("#").upper())
            ln.insert(0, solid)
        if dashed:
            ET.SubElement(ln, q(A_NS, "prstDash")).set("val", "dash")
    sp_tree.append(elem)
    return elem


def center(x: int, y: int, w: int, h: int) -> tuple[int, int]:
    return x + w // 2, y + h // 2


def build_relationships() -> ET.ElementTree:
    rels = ET.Element("Relationships")
    rels.set("xmlns", PKG_REL_NS)
    entries = [
        ("rId1", "../slideLayouts/slideLayout7.xml", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout"),
        ("rId2", "../media/mvtec_cable_input.png", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"),
        ("rId3", "../media/mvtec_cable_fixed.png", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"),
        ("rId4", "../media/mvtec_cable_direct.png", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"),
        ("rId5", "../media/mvtec_cable_final.png", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"),
    ]
    for rid, target, typ in entries:
        rel = ET.SubElement(rels, "Relationship")
        rel.set("Id", rid)
        rel.set("Target", target)
        rel.set("Type", typ)
    return ET.ElementTree(rels)


def restrict_presentation_to_slide1(pkg: Path) -> None:
    pres_path = pkg / "ppt/presentation.xml"
    pres = parse_xml(pres_path)
    root = pres.getroot()
    sld_id_lst = root.find("p:sldIdLst", NS)
    if sld_id_lst is None:
        raise ValueError("presentation has no sldIdLst")
    for child in list(sld_id_lst):
        sld_id_lst.remove(child)
    slide = ET.SubElement(sld_id_lst, q(P_NS, "sldId"))
    slide.set("id", "256")
    slide.set(q(R_NS, "id"), "rId2")
    pres.write(pres_path, encoding="UTF-8", xml_declaration=True)

    rels_path = pkg / "ppt/_rels/presentation.xml.rels"
    tree = parse_xml(rels_path)
    rels = tree.getroot()
    for rel in list(rels):
        if rel.attrib.get("Type") == "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" and rel.attrib.get("Id") != "rId2":
            rels.remove(rel)
    for rel in rels:
        if rel.attrib.get("Id") == "rId2":
            rel.set("Target", "slides/slide1.xml")
    tree.write(rels_path, encoding="UTF-8", xml_declaration=True)


def add_token_grid(
    sp_tree: ET.Element,
    templates: dict[str, ET.Element],
    next_id: list[int],
    x: int,
    y: int,
    *,
    cols: int = 4,
    rows: int = 3,
    cell: int = 190000,
    gap: int = 65000,
) -> None:
    colors = ["FBE3D6", "DEEBF7", "C1F1C8", "FFD965", "A5CAEC", "F2CFEE"]
    for r in range(rows):
        for c in range(cols):
            clone_shape(
                sp_tree,
                templates["small_rect"],
                next_id,
                "template patch-token cell",
                x + c * (cell + gap),
                y + r * (cell + gap),
                cell,
                cell,
                "",
                fill=colors[(r * cols + c) % len(colors)],
                line="666B6A",
                line_w=15000,
                font_size=1000,
            )


def add_prototype_cells(
    sp_tree: ET.Element,
    templates: dict[str, ET.Element],
    next_id: list[int],
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    adapted: str,
) -> None:
    clone_shape(
        sp_tree,
        templates["tag_green"],
        next_id,
        "normal prototype",
        x + 180000,
        y + 560000,
        w - 360000,
        360000,
        "normal",
        fill="DDF2D3",
        line="7A9E6B",
        line_w=19050,
        font_size=1350,
        bold=True,
    )
    clone_shape(
        sp_tree,
        templates["node_pink"],
        next_id,
        "abnormal prototype",
        x + 180000,
        y + 990000,
        w - 360000,
        360000,
        "abnormal",
        fill="F8D4DA",
        line="B46E7D",
        line_w=19050,
        font_size=1350,
        bold=True,
    )
    clone_shape(
        sp_tree,
        templates["small_label"],
        next_id,
        "prototype state tag",
        x + 280000,
        y + h - 420000,
        w - 560000,
        300000,
        adapted,
        fill="FFFFFF",
        line="A9A9A9",
        line_w=12000,
        font_size=1150,
        text_color="555555",
    )


def add_row(
    sp_tree: ET.Element,
    templates: dict[str, ET.Element],
    next_id: list[int],
    y: int,
    title: str,
    subtitle: str,
    output_rel: str,
    mode: str,
    row_fill: str,
) -> None:
    row_x, row_w, row_h = 540000, 16900000, 7150000

    # Background and structural separators use cloned template shapes.
    clone_shape(
        sp_tree,
        templates["row_bg"],
        next_id,
        f"{title} row background",
        row_x,
        y,
        row_w,
        row_h,
        "",
        fill=row_fill,
        line="D6CABE",
        line_w=19050,
        font_size=1000,
    )
    clone_shape(
        sp_tree,
        templates["title_tag"],
        next_id,
        f"{title} method tag",
        row_x + 320000,
        y + 250000,
        4450000,
        640000,
        title,
        fill="FFD965",
        line="3E4044",
        line_w=24000,
        font_size=2150,
        bold=True,
        align="l",
    )
    clone_shape(
        sp_tree,
        templates["small_label"],
        next_id,
        f"{title} subtitle",
        row_x + 5000000,
        y + 290000,
        7500000,
        540000,
        subtitle,
        fill="FFFFFF",
        line="D1C9BC",
        line_w=12000,
        font_size=1350,
        text_color="555555",
        align="l",
    )
    clone_shape(
        sp_tree,
        templates["lane_label"],
        next_id,
        f"{title} image lane label",
        row_x + 320000,
        y + 1250000,
        1500000,
        430000,
        "Image tower",
        fill="DEEBF7",
        line="7A9EC4",
        line_w=16000,
        font_size=1250,
        bold=True,
    )
    clone_shape(
        sp_tree,
        templates["lane_label"],
        next_id,
        f"{title} text lane label",
        row_x + 320000,
        y + 4380000,
        1500000,
        430000,
        "Text tower",
        fill="F8D4DA",
        line="B46E7D",
        line_w=16000,
        font_size=1250,
        bold=True,
    )

    # Geometry.
    inp = (row_x + 520000, y + 1830000, 1300000, 1300000)
    img_enc = (row_x + 2350000, y + 1620000, 2050000, 1740000)
    patch = (row_x + 4750000, y + 1480000, 2100000, 1960000)
    prompt = (row_x + 520000, y + 4930000, 2600000, 1220000)
    txt_enc = (row_x + 3600000, y + 4780000, 2050000, 1550000)
    proto = (row_x + 6100000, y + 4620000, 2350000, 1780000)
    sim = (row_x + 9100000, y + 3060000, 2150000, 1550000)
    out_img = (row_x + 11780000, y + 2830000, 1280000, 1280000)
    out_box = (row_x + 13320000, y + 2590000, 2850000, 1780000)

    # Common arrows are placed before nodes so that template nodes remain on top.
    sx, sy = center(*inp)
    ex, ey = center(*img_enc)
    clone_connector(sp_tree, templates["arrow"], next_id, "input to image encoder", sx + inp[2] // 2, sy, img_enc[0], ey, color="555555")
    sx, sy = center(*img_enc)
    ex, ey = center(*patch)
    clone_connector(sp_tree, templates["arrow"], next_id, "image encoder to patch tokens", sx + img_enc[2] // 2, sy, patch[0], ey, color="555555")
    sx, sy = center(*prompt)
    ex, ey = center(*txt_enc)
    clone_connector(sp_tree, templates["arrow"], next_id, "prompt to text encoder", sx + prompt[2] // 2, sy, txt_enc[0], ey, color="555555")
    sx, sy = center(*txt_enc)
    ex, ey = center(*proto)
    clone_connector(sp_tree, templates["arrow"], next_id, "text encoder to prototypes", sx + txt_enc[2] // 2, sy, proto[0], ey, color="555555")
    clone_connector(sp_tree, templates["arrow"], next_id, "patch tokens to similarity", patch[0] + patch[2], patch[1] + patch[3] // 2, sim[0], sim[1] + 500000, color="555555")
    clone_connector(sp_tree, templates["arrow"], next_id, "prototypes to similarity", proto[0] + proto[2], proto[1] + proto[3] // 2, sim[0], sim[1] + 1050000, color="555555")
    clone_connector(sp_tree, templates["arrow"], next_id, "similarity to anomaly map", sim[0] + sim[2], sim[1] + sim[3] // 2, out_img[0], out_img[1] + out_img[3] // 2, color="555555")
    clone_connector(sp_tree, templates["arrow"], next_id, "map to decision", out_img[0] + out_img[2], out_img[1] + out_img[3] // 2, out_box[0], out_box[1] + out_box[3] // 2, color="555555")

    if mode == "direct":
        upd = (row_x + 6650000, y + 1740000, 2200000, 920000)
        clone_connector(sp_tree, templates["arrow"], next_id, "direct token update branch", patch[0] + patch[2], patch[1] + 660000, upd[0], upd[1] + upd[3] // 2, color="C25A20", width=35000)
        clone_connector(sp_tree, templates["arrow"], next_id, "direct update to prototypes", upd[0] + upd[2] // 2, upd[1] + upd[3], proto[0] + proto[2] // 2, proto[1], color="C25A20", width=35000)
    elif mode == "wavelet":
        dwt = (row_x + 6350000, y + 1380000, 1450000, 760000)
        gate = (row_x + 8150000, y + 1380000, 1650000, 760000)
        upd = (row_x + 7250000, y + 2620000, 1900000, 820000)
        clone_connector(sp_tree, templates["arrow"], next_id, "patch tokens to DWT", patch[0] + patch[2], patch[1] + 530000, dwt[0], dwt[1] + dwt[3] // 2, color="196285", width=35000)
        clone_connector(sp_tree, templates["arrow"], next_id, "DWT to wavelet gate", dwt[0] + dwt[2], dwt[1] + dwt[3] // 2, gate[0], gate[1] + gate[3] // 2, color="196285", width=35000)
        clone_connector(sp_tree, templates["arrow"], next_id, "wavelet gate to update", gate[0] + gate[2] // 2, gate[1] + gate[3], upd[0] + upd[2] // 2, upd[1], color="196285", width=35000)
        clone_connector(sp_tree, templates["arrow"], next_id, "gated update to prototypes", upd[0] + upd[2] // 2, upd[1] + upd[3], proto[0] + proto[2] // 2, proto[1], color="196285", width=35000)

    # Common nodes.
    clone_pic(sp_tree, templates["pic"], next_id, "MVTec cable input image", "rId2", *inp)
    clone_shape(
        sp_tree,
        templates["tower_blue"],
        next_id,
        "image encoder tower",
        *img_enc,
        "Image\nEncoder",
        fill="DEEBF7",
        line="8D8989",
        line_w=30000,
        font_size=1800,
        bold=True,
    )
    clone_shape(
        sp_tree,
        templates["tower_green"],
        next_id,
        "patch token container",
        *patch,
        "Patch / image tokens",
        fill="E8F4E9",
        line="8D8989",
        line_w=30000,
        font_size=1450,
        bold=True,
        anchor="t",
    )
    add_token_grid(sp_tree, templates, next_id, patch[0] + 450000, patch[1] + 650000)
    clone_shape(
        sp_tree,
        templates["prompt_box"],
        next_id,
        "text prompts",
        *prompt,
        "Text prompts\n\"normal cable\"\n\"abnormal cable\"",
        fill="FFFFFF",
        line="8D8989",
        line_w=26000,
        font_size=1420,
        bold=False,
    )
    clone_shape(
        sp_tree,
        templates["tower_pink"],
        next_id,
        "text encoder tower",
        *txt_enc,
        "Text\nEncoder",
        fill="FCF0EE",
        line="8D8989",
        line_w=30000,
        font_size=1800,
        bold=True,
    )
    proto_state = {"frozen": "frozen", "direct": "adapted", "wavelet": "gated adapted"}[mode]
    clone_shape(
        sp_tree,
        templates["proto_container"],
        next_id,
        "text prototype memory",
        *proto,
        "Text-side\nprototypes",
        fill="FFF9E8",
        line="A89B75",
        line_w=26000,
        font_size=1450,
        bold=True,
        anchor="t",
    )
    add_prototype_cells(sp_tree, templates, next_id, *proto, adapted=proto_state)

    clone_shape(
        sp_tree,
        templates["sim_box"],
        next_id,
        "similarity scoring",
        *sim,
        "Similarity\nscoring",
        fill="F2F2F2",
        line="7C7B7B",
        line_w=26000,
        font_size=1650,
        bold=True,
    )
    clone_pic(sp_tree, templates["pic"], next_id, f"{title} anomaly map", output_rel, *out_img)
    decision_text = {
        "frozen": "Fixed prototypes\nno test-time update",
        "direct": "Direct prototype update\nabnormal patches can drift",
        "wavelet": "Wavelet-gated update\nlow-frequency reliable cues",
    }[mode]
    decision_fill = {"frozen": "DEEBF7", "direct": "FAE2D5", "wavelet": "E8F4E9"}[mode]
    clone_shape(
        sp_tree,
        templates["decision_box"],
        next_id,
        "method effect box",
        *out_box,
        decision_text,
        fill=decision_fill,
        line="8D8989",
        line_w=26000,
        font_size=1450,
        bold=True,
    )

    if mode == "frozen":
        clone_shape(
            sp_tree,
            templates["small_label"],
            next_id,
            "frozen clip note",
            row_x + 6940000,
            y + 2350000,
            1800000,
            520000,
            "no feedback",
            fill="F2F2F2",
            line="999999",
            line_w=14000,
            font_size=1250,
            text_color="666666",
        )
    elif mode == "direct":
        upd = (row_x + 6650000, y + 1740000, 2200000, 920000)
        clone_shape(
            sp_tree,
            templates["node_orange"],
            next_id,
            "direct prototype update module",
            *upd,
            "Direct\nprototype update",
            fill="FAE2D5",
            line="C25A20",
            line_w=26000,
            font_size=1350,
            bold=True,
        )
        clone_shape(
            sp_tree,
            templates["small_label"],
            next_id,
            "direct risk label",
            row_x + 8900000,
            y + 1760000,
            1700000,
            520000,
            "unfiltered\nimage cues",
            fill="FFFFFF",
            line="C25A20",
            line_w=13000,
            font_size=1150,
            text_color="8A3A14",
        )
    elif mode == "wavelet":
        dwt = (row_x + 6350000, y + 1380000, 1450000, 760000)
        gate = (row_x + 8150000, y + 1380000, 1650000, 760000)
        upd = (row_x + 7250000, y + 2620000, 1900000, 820000)
        clone_shape(
            sp_tree,
            templates["node_blue"],
            next_id,
            "DWT module",
            *dwt,
            "DWT\nLL / HF",
            fill="DEEBF7",
            line="196285",
            line_w=26000,
            font_size=1350,
            bold=True,
        )
        clone_shape(
            sp_tree,
            templates["node_green"],
            next_id,
            "wavelet gate module",
            *gate,
            "Wavelet\ngate",
            fill="E8F4E9",
            line="7A9E6B",
            line_w=26000,
            font_size=1350,
            bold=True,
        )
        clone_shape(
            sp_tree,
            templates["node_purple"],
            next_id,
            "gated prototype update module",
            *upd,
            "Gated\nprototype update",
            fill="E6E0F0",
            line="7E6A9C",
            line_w=26000,
            font_size=1300,
            bold=True,
        )
        clone_shape(
            sp_tree,
            templates["small_label"],
            next_id,
            "wavelet reliability label",
            row_x + 9950000,
            y + 1420000,
            1900000,
            620000,
            "suppress\nhigh-frequency noise",
            fill="FFFFFF",
            line="196285",
            line_w=13000,
            font_size=1120,
            text_color="174E68",
        )


def build_deck() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    with zipfile.ZipFile(TEMPLATE_PPTX) as zf:
        zf.extractall(WORK)

    slide26 = parse_xml(WORK / "ppt/slides/slide26.xml").getroot()
    slide22 = parse_xml(WORK / "ppt/slides/slide22.xml").getroot()
    base_tree = parse_xml(WORK / "ppt/slides/slide1.xml")
    base_root = base_tree.getroot()
    sp_tree = clear_slide(base_root)

    templates = {
        "pic": find_by_cnv_id(slide26, "2"),
        "row_bg": find_by_cnv_id(slide22, "341"),
        "title_tag": find_by_cnv_id(slide26, "313"),
        "small_label": find_by_cnv_id(slide26, "369"),
        "lane_label": find_by_cnv_id(slide26, "372"),
        "tower_blue": find_by_cnv_id(slide26, "272"),
        "tower_green": find_by_cnv_id(slide26, "270"),
        "tower_pink": find_by_cnv_id(slide26, "271"),
        "prompt_box": find_by_cnv_id(slide26, "397"),
        "proto_container": find_by_cnv_id(slide22, "480"),
        "sim_box": find_by_cnv_id(slide26, "397"),
        "decision_box": find_by_cnv_id(slide26, "272"),
        "node_orange": find_by_cnv_id(slide26, "280"),
        "node_blue": find_by_cnv_id(slide26, "387"),
        "node_green": find_by_cnv_id(slide26, "390"),
        "node_purple": find_by_cnv_id(slide22, "438"),
        "tag_green": find_by_cnv_id(slide26, "390"),
        "node_pink": find_by_cnv_id(slide22, "429"),
        "small_rect": find_by_cnv_id(slide26, "411"),
        "arrow": find_by_cnv_id(slide26, "278"),
    }

    next_id = [10]

    # Figure title.
    clone_shape(
        sp_tree,
        templates["title_tag"],
        next_id,
        "figure title",
        650000,
        360000,
        8350000,
        690000,
        "CLIP dual-tower adaptation paths for zero-shot anomaly detection",
        fill="FFD965",
        line="3E4044",
        line_w=26000,
        font_size=2050,
        bold=True,
        align="l",
    )
    clone_shape(
        sp_tree,
        templates["small_label"],
        next_id,
        "figure note",
        9350000,
        390000,
        7850000,
        620000,
        "All nodes/arrows are cloned native components from the supplied PPT template.",
        fill="FFFFFF",
        line="D1C9BC",
        line_w=12000,
        font_size=1320,
        text_color="555555",
        align="l",
    )

    rows = [
        (
            1350000,
            "Frozen CLIP-ZSAD",
            "Image and text encoders remain frozen; text prototypes are fixed during inference.",
            "rId3",
            "frozen",
            "FCF3EA",
        ),
        (
            8750000,
            "Direct Test-Time Prototype Adaptation",
            "Image-side tokens directly update text-side prototypes at test time.",
            "rId4",
            "direct",
            "FFF4EC",
        ),
        (
            16150000,
            "Wavelet-Guided Prototype Adaptation (Ours)",
            "Image tokens first pass through DWT and a wavelet gate before prototype feedback.",
            "rId5",
            "wavelet",
            "F3FFEA",
        ),
    ]
    for row in rows:
        add_row(sp_tree, templates, next_id, *row)

    # Save slide and relationships.
    base_tree.write(WORK / "ppt/slides/slide1.xml", encoding="UTF-8", xml_declaration=True)
    build_relationships().write(WORK / "ppt/slides/_rels/slide1.xml.rels", encoding="UTF-8", xml_declaration=True)

    media = WORK / "ppt/media"
    shutil.copyfile(ASSET_DIR / "mvtec_cable_input.png", media / "mvtec_cable_input.png")
    shutil.copyfile(ASSET_DIR / "mvtec_cable_fixed.png", media / "mvtec_cable_fixed.png")
    shutil.copyfile(ASSET_DIR / "mvtec_cable_direct.png", media / "mvtec_cable_direct.png")
    shutil.copyfile(ASSET_DIR / "mvtec_cable_final.png", media / "mvtec_cable_final.png")

    restrict_presentation_to_slide1(WORK)

    if OUT_PPTX.exists():
        OUT_PPTX.unlink()
    with zipfile.ZipFile(OUT_PPTX, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(WORK.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(WORK).as_posix())

    AUDIT.write_text(
        "\n".join(
            [
                "# Component-built Figure Audit",
                "",
                f"- Source template: `{TEMPLATE_PPTX}`",
                "- Output is a one-slide PPTX whose visible diagram elements are cloned from native PowerPoint XML components in the template deck.",
                "- Cloned component classes: row rounded panels, method tags, lane labels, encoder blocks, prompt/prototype boxes, small token cells, straight connectors, and image frames.",
                "- Real MVTec cable image is inserted as the input; fixed/direct/final cable maps are inserted as output examples.",
                "- Architecture intent: row 2 and row 3 place adaptation as a feedback branch into text-side prototypes, not as a serial post-processing block after the anomaly map.",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    build_deck()
    print(OUT_PPTX)
