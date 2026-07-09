import cv2
import os
import numpy as np


def _normalize_scoremap(pred):
    pred = np.asarray(pred, dtype=np.float32)
    min_value = float(np.nanmin(pred))
    max_value = float(np.nanmax(pred))
    denom = max(max_value - min_value, 1e-6)
    return (pred - min_value) / denom


def _resolve_save_file(path, save_path, cls_name, data_root=None, preserve_structure=False):
    filename = path.split('/')[-1]
    if preserve_structure:
        if data_root is None:
            raise ValueError("data_root is required when preserve_structure=True")
        rel_path = os.path.relpath(os.path.abspath(path), os.path.abspath(str(data_root)))
        if rel_path == os.pardir or rel_path.startswith(os.pardir + os.sep):
            raise ValueError(f"image path is outside data_root: {path}")
        return os.path.join(save_path, rel_path)

    cls = path.split('/')[-2]
    return os.path.join(save_path, 'imgs', cls_name, cls, filename)


def visualizer(pathes, anomaly_map, img_size, save_path, cls_name, data_root=None, preserve_structure=False):
    for idx, path in enumerate(pathes):
        source = cv2.imread(path)
        if source is None:
            raise FileNotFoundError(f"failed to read image: {path}")
        vis = cv2.cvtColor(cv2.resize(source, (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
        mask = _normalize_scoremap(anomaly_map[idx])
        mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
        mask = np.clip(mask, 0.0, 1.0)
        vis = apply_ad_scoremap(vis, mask)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
        save_file = _resolve_save_file(
            path,
            save_path,
            cls_name[idx],
            data_root=data_root,
            preserve_structure=preserve_structure,
        )
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        cv2.imwrite(save_file, vis)

def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = np.nan_to_num(scoremap, nan=0.0, posinf=1.0, neginf=0.0)
    scoremap = np.clip(scoremap, 0.0, 1.0)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)
