from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
import numpy as np
from skimage import measure


def _max_f1_score(gt, pr):
    precision, recall, _ = precision_recall_curve(gt, pr)
    denom = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denom,
        out=np.zeros_like(precision, dtype=np.float64),
        where=denom > 0,
    )
    if f1.size == 0:
        return float("nan")
    return float(np.nanmax(f1))


def _as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _pixel_rank_metrics(results, obj):
    cache_key = "_pixel_rank_metrics"
    if cache_key in results[obj]:
        return results[obj][cache_key]

    gt = _as_numpy(results[obj]["imgs_masks"]).astype(bool).ravel()
    pr = _as_numpy(results[obj]["anomaly_maps"]).ravel()
    positive_count = int(gt.sum())
    negative_count = int(gt.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        metrics = {
            "pixel-auroc": float("nan"),
            "pixel-ap": float("nan"),
            "pixel-f1-max": float("nan"),
        }
        results[obj][cache_key] = metrics
        return metrics

    order = np.argsort(pr, kind="mergesort")[::-1]
    sorted_scores = pr[order]
    sorted_gt = gt[order]
    distinct_value_indices = np.where(np.diff(sorted_scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, sorted_gt.size - 1]

    tps = np.cumsum(sorted_gt, dtype=np.float64)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / positive_count

    fpr = np.r_[0, fps / negative_count]
    tpr = np.r_[0, recall]
    pixel_auroc = auc(fpr, tpr)
    pixel_ap = np.sum(np.diff(np.r_[0, recall]) * precision)
    denom = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denom,
        out=np.zeros_like(precision, dtype=np.float64),
        where=denom > 0,
    )

    metrics = {
        "pixel-auroc": float(pixel_auroc),
        "pixel-ap": float(pixel_ap),
        "pixel-f1-max": float(np.nanmax(f1)) if f1.size else float("nan"),
    }
    results[obj][cache_key] = metrics
    return metrics

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    masks = np.asarray(masks).astype(bool)
    amaps = np.asarray(amaps)
    min_th, max_th = amaps.min(), amaps.max()
    if max_th <= min_th:
        return float("nan")

    delta = (max_th - min_th) / max_step
    thresholds = np.arange(min_th, max_th, delta)

    region_scores = []
    for mask, amap in zip(masks, amaps):
        label_img = measure.label(mask)
        for region in measure.regionprops(label_img):
            coords = region.coords
            region_scores.append(amap[coords[:, 0], coords[:, 1]])

    if len(region_scores) == 0:
        return float("nan")

    background_scores = amaps[~masks]
    if background_scores.size == 0:
        return float("nan")

    sorted_background_scores = np.sort(background_scores)
    background_counts = background_scores.size - np.searchsorted(
        sorted_background_scores,
        thresholds,
        side="right",
    )
    fprs = background_counts / background_scores.size

    region_pros = []
    for scores in region_scores:
        sorted_scores = np.sort(scores)
        region_counts = scores.size - np.searchsorted(
            sorted_scores,
            thresholds,
            side="right",
        )
        region_pros.append(region_counts / scores.size)
    pros = np.stack(region_pros, axis=0).mean(axis=0)

    idxes = fprs < expect_fpr
    if idxes.sum() < 2:
        return float("nan")
    fprs = fprs[idxes]
    if fprs.max() <= fprs.min():
        return float("nan")
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc


def image_level_metrics(results, obj, metric):
    gt = results[obj]['gt_sp']
    pr = results[obj]['pr_sp']
    gt = np.array(gt)
    pr = np.array(pr)
    if metric == 'image-auroc':
        performance = roc_auc_score(gt, pr)
    elif metric == 'image-ap':
        performance = average_precision_score(gt, pr)
    elif metric == 'image-f1-max':
        performance = _max_f1_score(gt, pr)
    else:
        raise ValueError(f"unsupported image metric: {metric}")

    return performance
    # table.append(str(np.round(performance * 100, decimals=1)))


def pixel_level_metrics(results, obj, metric, aupro_steps=200):
    if metric == 'pixel-auroc':
        performance = _pixel_rank_metrics(results, obj)[metric]
    elif metric == 'pixel-ap':
        performance = _pixel_rank_metrics(results, obj)[metric]
    elif metric == 'pixel-f1-max':
        performance = _pixel_rank_metrics(results, obj)[metric]
    elif metric == 'pixel-aupro':
        gt = _as_numpy(results[obj]['imgs_masks'])
        pr = _as_numpy(results[obj]['anomaly_maps'])
        if len(gt.shape) == 4:
            gt = gt.squeeze(1)
        if len(pr.shape) == 4:
            pr = pr.squeeze(1)
        performance = cal_pro_score(gt, pr, max_step=aupro_steps)
    else:
        raise ValueError(f"unsupported pixel metric: {metric}")
    return performance
    
