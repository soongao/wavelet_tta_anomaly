from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F


Box = Tuple[int, int, int, int]


def build_multicrop_boxes(
    image_size: int,
    grid: int = 2,
    crop_ratio: float = 0.75,
) -> List[Box]:
    """Build deterministic overlapping crop boxes on the resized image grid."""
    if grid < 1:
        raise ValueError(f"grid must be >= 1, got {grid}")
    image_size = int(image_size)
    crop_size = int(round(float(crop_ratio) * image_size))
    crop_size = max(1, min(crop_size, image_size))
    if grid == 1 or crop_size == image_size:
        return [(0, 0, image_size, image_size)]

    max_start = image_size - crop_size
    starts = [
        int(round(max_start * idx / float(grid - 1)))
        for idx in range(grid)
    ]
    boxes = []
    for y0 in starts:
        for x0 in starts:
            x1 = min(x0 + crop_size, image_size)
            y1 = min(y0 + crop_size, image_size)
            boxes.append((x0, y0, x1, y1))
    return boxes


def output_box_to_pil_box(
    box: Box,
    output_size: int,
    image_width: int,
    image_height: int,
) -> Box:
    """Map a box from resized square output coordinates to original PIL pixels."""
    x0, y0, x1, y1 = box
    scale_x = float(image_width) / float(output_size)
    scale_y = float(image_height) / float(output_size)
    pil_box = (
        int(round(x0 * scale_x)),
        int(round(y0 * scale_y)),
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
    )
    left, top, right, bottom = pil_box
    left = max(0, min(left, image_width - 1))
    top = max(0, min(top, image_height - 1))
    right = max(left + 1, min(right, image_width))
    bottom = max(top + 1, min(bottom, image_height))
    return left, top, right, bottom


def stitch_crop_maps(
    crop_maps: Sequence[torch.Tensor],
    boxes: Sequence[Box],
    image_size: int,
) -> torch.Tensor:
    """Paste crop anomaly maps back to the full image grid and average overlap."""
    if len(crop_maps) != len(boxes):
        raise ValueError("crop_maps and boxes must have the same length")
    if len(crop_maps) == 0:
        raise ValueError("at least one crop map is required")

    first = crop_maps[0]
    if first.dim() == 2:
        first = first.unsqueeze(0)
    if first.dim() != 3:
        raise ValueError(f"crop map must be [B, H, W] or [H, W], got {tuple(first.shape)}")
    batch = first.size(0)
    dtype = first.dtype
    device = first.device
    canvas = torch.zeros(batch, image_size, image_size, dtype=dtype, device=device)
    weights = torch.zeros_like(canvas)

    for crop_map, box in zip(crop_maps, boxes):
        if crop_map.dim() == 2:
            crop_map = crop_map.unsqueeze(0)
        crop_map = crop_map.to(device=device, dtype=dtype)
        x0, y0, x1, y1 = box
        height = max(1, y1 - y0)
        width = max(1, x1 - x0)
        resized = F.interpolate(
            crop_map.unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        canvas[:, y0:y1, x0:x1] += resized
        weights[:, y0:y1, x0:x1] += 1.0

    return canvas / weights.clamp_min(1.0)
