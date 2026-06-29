from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
import numpy as np
from skimage import measure

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

    pros, fprs, ths = [], [], []
    for th in thresholds:
        pro = [(scores > th).sum() / scores.size for scores in region_scores]
        fpr = (background_scores > th).sum() / background_scores.size
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
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

    return performance
    # table.append(str(np.round(performance * 100, decimals=1)))


def pixel_level_metrics(results, obj, metric, aupro_steps=200):
    gt = results[obj]['imgs_masks']
    pr = results[obj]['anomaly_maps']
    gt = np.array(gt)
    pr = np.array(pr)
    if metric == 'pixel-auroc':
        performance = roc_auc_score(gt.ravel(), pr.ravel())
    elif metric == 'pixel-aupro':
        if len(gt.shape) == 4:
            gt = gt.squeeze(1)
        if len(pr.shape) == 4:
            pr = pr.squeeze(1)
        performance = cal_pro_score(gt, pr, max_step=aupro_steps)
    return performance
    
