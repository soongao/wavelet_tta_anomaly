import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset
from logger import get_logger, log_run_context
from tqdm import tqdm

import os
import random
import numpy as np
from tabulate import tabulate
from utils import get_transform
from multicrop_utils import build_multicrop_boxes, output_box_to_pil_box, stitch_crop_maps
from PIL import Image
from wavelet_calibration import (
    apply_layer_consistency_calibration,
    apply_low_rank_residual_calibration,
    apply_wavelet_calibration,
    apply_wavelet_residual_map_calibration,
    apply_structure_texture_calibration,
    compute_texture_reliability,
    compute_wavelet_reliability,
    fuse_image_score_with_pixel_score,
    global_image_confidence_gate,
    fuse_wavelet_gates,
    structure_texture_gates_from_patch_features,
    topk_pixel_score,
    wavelet_gate_from_patch_features,
)
from test_time_rectification import rectify_text_features_with_multi_layer_anchors

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from visualization import visualizer

from metrics import image_level_metrics, pixel_level_metrics
from tqdm import tqdm
from scipy.ndimage import gaussian_filter


def build_anomaly_maps_from_patch_features(patch_features, text_features, args):
    anomaly_map_list = []
    selected_patch_features = []
    for idx, patch_feature in enumerate(patch_features):
        if idx >= args.feature_map_layer[0]:
            patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
            similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features)
            similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
            anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
            anomaly_map_list.append(anomaly_map)
            selected_patch_features.append(patch_feature)

    if len(anomaly_map_list) == 0:
        raise ValueError("No patch feature layer was selected. Check --feature_map_layer.")

    stacked_maps = torch.stack(anomaly_map_list)
    if args.layer_weighting == "sum":
        anomaly_map = stacked_maps.sum(dim=0)
    elif args.layer_weighting == "dynamic_wavelet":
        gate_list = [
            wavelet_gate_from_patch_features(patch_feature, output_size=args.image_size)
            for patch_feature in selected_patch_features
        ]
        reliability_list = [
            compute_wavelet_reliability(anomaly_map, gate).view(-1)
            for anomaly_map, gate in zip(anomaly_map_list, gate_list)
        ]
        weights = torch.stack(reliability_list, dim=0)
        temperature = max(float(args.layer_weight_temperature), 1e-6)
        weights = torch.softmax(weights / temperature, dim=0).view(
            len(anomaly_map_list),
            -1,
            1,
            1,
        )
        anomaly_map = (stacked_maps * weights).sum(dim=0) * len(anomaly_map_list)
    else:
        raise ValueError(f"unsupported layer weighting mode: {args.layer_weighting}")

    if getattr(args, "use_layer_consistency", False):
        return anomaly_map, selected_patch_features, anomaly_map_list
    return anomaly_map, selected_patch_features


def build_online_multicrop_map(image_path, model, text_features, preprocess, args, device):
    image = Image.open(image_path).convert("RGB")
    boxes = build_multicrop_boxes(
        image_size=args.image_size,
        grid=args.multicrop_grid,
        crop_ratio=args.multicrop_crop_ratio,
    )
    crop_maps = []
    for box in boxes:
        pil_box = output_box_to_pil_box(
            box,
            output_size=args.image_size,
            image_width=image.width,
            image_height=image.height,
        )
        crop = image.crop(pil_box)
        crop_tensor = preprocess(crop).unsqueeze(0).to(device)
        crop_features = model.encode_image(crop_tensor, args.features_list, DPAM_layer=20)[1]
        crop_map, _ = build_anomaly_maps_from_patch_features(
            crop_features,
            text_features,
            args,
        )
        crop_maps.append(crop_map.detach().cpu().float())
    return stitch_crop_maps(crop_maps, boxes, args.image_size)


def test(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset

    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_run_context(
        logger,
        args,
        title="AnomalyCLIP full evaluation",
        extra_info={"device": device},
    )

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
    model.eval()

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list


    results = {}
    metrics = {}
    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        results[obj]['imgs_masks'] = []
        results[obj]['anomaly_maps'] = []
        metrics[obj] = {}
        metrics[obj]['pixel-auroc'] = 0
        metrics[obj]['pixel-aupro'] = 0
        metrics[obj]['image-auroc'] = 0
        metrics[obj]['image-ap'] = 0

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)
    text_features = text_features/text_features.norm(dim=-1, keepdim=True)


    model.to(device)
    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        cls_name = items['cls_name']
        cls_id = items['cls_id']
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())

        with torch.no_grad():
            image_features, patch_features = model.encode_image(image, features_list, DPAM_layer = 20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ text_features.permute(0, 2, 1)
            text_probs = (text_probs/0.07).softmax(-1)
            text_probs = text_probs[:, 0, 1]
            text_features_for_map = text_features[0]
            map_outputs = build_anomaly_maps_from_patch_features(
                patch_features,
                text_features_for_map,
                args,
            )
            if args.use_layer_consistency:
                anomaly_map, selected_patch_features, layer_maps = map_outputs
            else:
                anomaly_map, selected_patch_features = map_outputs
                layer_maps = None

            wavelet_gate_list = []
            texture_gate_list = []
            if args.use_wavelet or args.use_tta_rectification or args.use_wavelet_residual:
                for patch_feature in selected_patch_features:
                    wavelet_gate = wavelet_gate_from_patch_features(
                        patch_feature,
                        output_size=args.image_size,
                        levels=args.wavelet_levels,
                        level_fusion=args.wavelet_level_fusion,
                    )
                    wavelet_gate_list.append(wavelet_gate)
                    texture_gate = structure_texture_gates_from_patch_features(
                        patch_feature,
                        output_size=args.image_size,
                        edge_power=args.texture_edge_power,
                        levels=args.wavelet_levels,
                        level_fusion=args.wavelet_level_fusion,
                    )[0]
                    texture_gate_list.append(texture_gate)
                wavelet_gate = fuse_wavelet_gates(wavelet_gate_list, mode=args.wavelet_fusion)
                texture_gate = fuse_wavelet_gates(texture_gate_list, mode=args.wavelet_fusion)
                if args.wavelet_mode == "dual_route":
                    wavelet_reliability = compute_texture_reliability(
                        anomaly_map,
                        texture_gate,
                        topk_ratio=args.wavelet_reliability_topk_ratio,
                    )
                else:
                    wavelet_reliability = compute_wavelet_reliability(
                        anomaly_map,
                        wavelet_gate,
                        topk_ratio=args.wavelet_reliability_topk_ratio,
                    )
            else:
                wavelet_gate = None
                texture_gate = None
                wavelet_reliability = None

            if args.use_tta_rectification:
                tta_gate = texture_gate if args.wavelet_mode == "dual_route" else wavelet_gate
                rectified_text_features, _ = rectify_text_features_with_multi_layer_anchors(
                    text_features_for_map,
                    selected_patch_features,
                    anomaly_map,
                    wavelet_gate=tta_gate,
                    reliability=wavelet_reliability if not args.disable_adaptive_wavelet else None,
                    mode=args.tta_mode,
                    alpha=args.tta_alpha,
                    topk_ratio=args.tta_topk_ratio,
                    update_abnormal=args.tta_update_abnormal,
                    min_confidence=args.tta_min_confidence,
                    min_confidence_margin=args.tta_min_confidence_margin,
                    repulsion_weight=args.tta_repulsion_weight,
                    abnormal_alpha_scale=args.tta_abnormal_alpha_scale,
                    fusion=args.tta_anchor_layers,
                )
                text_features_for_map = rectified_text_features[0]
                map_outputs = build_anomaly_maps_from_patch_features(
                    patch_features,
                    text_features_for_map,
                    args,
                )
                if args.use_layer_consistency:
                    anomaly_map, _, layer_maps = map_outputs
                else:
                    anomaly_map, _ = map_outputs
                    layer_maps = None
                text_probs = image_features @ text_features_for_map.unsqueeze(0).permute(0, 2, 1)
                text_probs = (text_probs / 0.07).softmax(-1)
                text_probs = text_probs[:, 0, 1]

            if args.use_wavelet:
                image_confidence_gate = None
                if args.use_image_to_pixel_gate:
                    image_confidence_gate = global_image_confidence_gate(
                        text_probs,
                        power=args.image_to_pixel_power,
                        min_gate=args.image_to_pixel_min_gate,
                        max_gate=args.image_to_pixel_max_gate,
                    )
                if args.wavelet_mode == "dual_route":
                    anomaly_map, _ = apply_structure_texture_calibration(
                        anomaly_map,
                        texture_gate,
                        beta=args.wavelet_beta,
                        condition_power=args.wavelet_condition_power,
                        suppress_beta=args.wavelet_suppress_beta,
                        texture_max_delta_ratio=args.texture_max_delta_ratio,
                        texture_suppression_weight=args.texture_suppression_weight,
                        texture_local_contrast_kernel=args.texture_local_contrast_kernel,
                        texture_local_contrast_weight=args.texture_local_contrast_weight,
                        rank_preserve_topk_ratio=args.rank_preserve_topk_ratio,
                        rank_gate_mode=args.rank_gate_mode,
                        rank_gate_temperature=args.rank_gate_temperature,
                        use_wavelet_confidence=args.use_wavelet_confidence,
                        wavelet_confidence_power=args.wavelet_confidence_power,
                        adaptive=not args.disable_adaptive_wavelet,
                        reliability=wavelet_reliability,
                        reliability_power=args.wavelet_reliability_power,
                        reliability_topk_ratio=args.wavelet_reliability_topk_ratio,
                        min_reliability=args.wavelet_min_reliability,
                        texture_delta_reliability_power=args.texture_delta_reliability_power,
                        image_confidence_gate=image_confidence_gate,
                        image_confidence_weight=args.image_to_pixel_weight,
                    )
                else:
                    anomaly_map, _ = apply_wavelet_calibration(
                        anomaly_map,
                        wavelet_gate,
                        beta=args.wavelet_beta,
                        condition_power=args.wavelet_condition_power,
                        suppress_beta=args.wavelet_suppress_beta,
                        adaptive=not args.disable_adaptive_wavelet,
                        reliability=wavelet_reliability,
                        reliability_power=args.wavelet_reliability_power,
                        reliability_topk_ratio=args.wavelet_reliability_topk_ratio,
                        image_confidence_gate=image_confidence_gate,
                        image_confidence_weight=args.image_to_pixel_weight,
                    )
            if args.use_layer_consistency:
                anomaly_map, _ = apply_layer_consistency_calibration(
                    anomaly_map,
                    layer_maps,
                    topk_ratio=args.layer_consistency_topk_ratio,
                    min_layers=args.layer_consistency_min_layers,
                    boost_weight=args.layer_consistency_boost,
                    suppress_weight=args.layer_consistency_suppress,
                    condition_power=args.layer_consistency_condition_power,
                )
            if args.use_wavelet_residual:
                residual_gate = texture_gate if args.wavelet_mode == "dual_route" else wavelet_gate
                anomaly_map, _ = apply_wavelet_residual_map_calibration(
                    anomaly_map,
                    residual_gate,
                    weight=args.wavelet_residual_weight,
                    topk_ratio=args.wavelet_residual_topk_ratio,
                    condition_power=args.wavelet_residual_condition_power,
                    max_delta_ratio=args.wavelet_residual_max_delta_ratio,
                    local_contrast_kernel=args.wavelet_residual_local_contrast_kernel,
                    local_contrast_weight=args.wavelet_residual_local_contrast_weight,
                    rank_gate_mode=args.wavelet_residual_rank_gate_mode,
                    rank_gate_temperature=args.wavelet_residual_rank_gate_temperature,
                    confidence_power=args.wavelet_residual_confidence_power,
                )
            if args.use_low_rank_residual:
                anomaly_map, _ = apply_low_rank_residual_calibration(
                    anomaly_map,
                    selected_patch_features,
                    output_size=args.image_size,
                    weight=args.low_rank_weight,
                    rank_ratio=args.low_rank_rank_ratio,
                    max_rank=args.low_rank_max_rank,
                    sample_patches=args.low_rank_sample_patches,
                    fusion=args.low_rank_fusion,
                    condition_power=args.low_rank_condition_power,
                    max_delta_ratio=args.low_rank_max_delta_ratio,
                    clip_topk_ratio=args.low_rank_clip_topk_ratio,
                    clip_gate_mode=args.low_rank_clip_gate_mode,
                    clip_gate_temperature=args.low_rank_clip_gate_temperature,
                    center=not args.low_rank_no_center,
                    layer_selection=args.low_rank_layer_selection,
                )
            if args.use_multicrop_fusion:
                crop_map = build_online_multicrop_map(
                    items['img_path'][0] if isinstance(items['img_path'], (list, tuple)) else items['img_path'],
                    model,
                    text_features_for_map,
                    preprocess,
                    args,
                    device,
                )
                anomaly_map = (
                    (1.0 - args.multicrop_weight) * anomaly_map.detach().cpu().float()
                    + args.multicrop_weight * crop_map.float()
                ).clamp_min(0.0)
            anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in anomaly_map.detach().cpu()], dim = 0 )
            if args.use_pixel_to_image_fusion:
                pixel_score = topk_pixel_score(
                    anomaly_map,
                    topk_ratio=args.pixel_to_image_topk_ratio,
                    normalize=args.pixel_to_image_normalize,
                )
                text_probs = fuse_image_score_with_pixel_score(
                    text_probs.detach().cpu(),
                    pixel_score,
                    weight=args.pixel_to_image_weight,
                )
            results[cls_name[0]]['pr_sp'].extend(text_probs.detach().cpu())
            results[cls_name[0]]['anomaly_maps'].append(anomaly_map)
            # visualizer(items['img_path'], anomaly_map.detach().cpu().numpy(), args.image_size, args.save_path, cls_name)

    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []
    for obj in obj_list:
        table = []
        table.append(obj)
        results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
        results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
        if args.metrics == 'image-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap) 
        elif args.metrics == 'pixel-level':
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc", aupro_steps=args.aupro_steps)
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro", aupro_steps=args.aupro_steps)
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        elif args.metrics == 'image-pixel-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc", aupro_steps=args.aupro_steps)
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro", aupro_steps=args.aupro_steps)
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(pixel_aupro * 100, decimals=1)))
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap) 
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        table_ls.append(table)

    if args.metrics == 'image-level':
        # logger
        table_ls.append(['mean', 
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'image_auroc', 'image_ap'], tablefmt="pipe")
    elif args.metrics == 'pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1))
                       ])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro'], tablefmt="pipe")
    elif args.metrics == 'image-pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)), 
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro', 'image_auroc', 'image_ap'], tablefmt="pipe")
    logger.info("\n%s", results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="/Users/bytedance/Downloads/mvtec_anomaly_detection", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./my_results_mvtec/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='/Users/bytedance/code/AnomalyCLIP/checkpoints/9_12_4_multiscale/epoch_15.pth', help='path to checkpoint')
    # model
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int,  nargs="+", default=[1, 2, 3], help="zero shot")
    parser.add_argument("--layer_weighting", type=str, default="sum", choices=["sum", "dynamic_wavelet"], help="multi-layer anomaly-map fusion strategy")
    parser.add_argument("--layer_weight_temperature", type=float, default=1.0, help="temperature for dynamic_wavelet layer weights")
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--aupro_steps", type=int, default=200, help="number of thresholds used by pixel AUPRO")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=float, default=5, help="zero shot")
    parser.add_argument("--use_wavelet", action="store_true", help="enable training-free wavelet feature calibration")
    parser.add_argument("--wavelet_beta", type=float, default=0.5, help="strength of wavelet calibration")
    parser.add_argument("--wavelet_condition_power", type=float, default=1.0, help="power for CLIP-conditioned wavelet gate")
    parser.add_argument("--wavelet_suppress_beta", type=float, default=0.3, help="strength for suppressing high-frequency normal texture")
    parser.add_argument("--wavelet_fusion", type=str, default="mean", choices=["mean", "sum"], help="multi-layer wavelet gate fusion")
    parser.add_argument("--wavelet_levels", type=int, default=1, help="number of repeated Haar decomposition levels per patch-feature layer")
    parser.add_argument("--wavelet_level_fusion", type=str, default="mean", choices=["mean", "sum"], help="fusion strategy for multi-scale Haar levels")
    parser.add_argument("--wavelet_mode", type=str, default="dual_route", choices=["dual_route", "adaptive"], help="wavelet calibration mode")
    parser.add_argument("--texture_edge_power", type=float, default=1.0, help="strength for removing structure edges from texture route")
    parser.add_argument("--texture_max_delta_ratio", type=float, default=0.05, help="maximum texture-route additive change relative to each map score range")
    parser.add_argument("--texture_suppression_weight", type=float, default=0.0, help="optional weight for suppressing non-semantic texture responses")
    parser.add_argument("--texture_local_contrast_kernel", type=int, default=0, help="local window for texture residual; 0 disables it")
    parser.add_argument("--texture_local_contrast_weight", type=float, default=0.0, help="mixing weight for local texture residual")
    parser.add_argument("--rank_preserve_topk_ratio", type=float, default=0.0, help="restrict texture bonus to top CLIP anomaly pixels; 0 disables it")
    parser.add_argument("--rank_gate_mode", type=str, default="hard", choices=["hard", "soft"], help="hard or soft CLIP-rank gate for the texture route")
    parser.add_argument("--rank_gate_temperature", type=float, default=0.05, help="sigmoid temperature for soft rank gate")
    parser.add_argument("--use_wavelet_confidence", action="store_true", help="use local CLIP-wavelet agreement as confidence")
    parser.add_argument("--wavelet_confidence_power", type=float, default=1.0, help="power for local CLIP-wavelet confidence")
    parser.add_argument("--disable_adaptive_wavelet", action="store_true", help="disable per-image CLIP-wavelet agreement gating")
    parser.add_argument("--wavelet_reliability_power", type=float, default=1.0, help="power applied to per-image wavelet reliability")
    parser.add_argument("--wavelet_reliability_topk_ratio", type=float, default=0.05, help="top-k area used for CLIP-wavelet agreement")
    parser.add_argument("--wavelet_min_reliability", type=float, default=0.0, help="minimum per-image wavelet reliability required to apply texture calibration")
    parser.add_argument("--texture_delta_reliability_power", type=float, default=0.0, help="scale the max texture delta by per-image reliability")
    parser.add_argument("--use_layer_consistency", action="store_true", help="refine pixel map with cross-layer anomaly agreement")
    parser.add_argument("--layer_consistency_topk_ratio", type=float, default=0.10, help="per-layer top-k area used for cross-layer consistency")
    parser.add_argument("--layer_consistency_min_layers", type=int, default=2, help="minimum agreeing layers needed for consistency boost")
    parser.add_argument("--layer_consistency_boost", type=float, default=0.03, help="strength of cross-layer consistency boost")
    parser.add_argument("--layer_consistency_suppress", type=float, default=0.0, help="strength of inconsistent-response suppression")
    parser.add_argument("--layer_consistency_condition_power", type=float, default=1.0, help="CLIP prior power for layer consistency calibration")
    parser.add_argument("--use_wavelet_residual", action="store_true", help="add a bounded wavelet residual directly to the pixel map")
    parser.add_argument("--wavelet_residual_weight", type=float, default=0.03, help="strength of direct wavelet residual map boost")
    parser.add_argument("--wavelet_residual_topk_ratio", type=float, default=0.35, help="CLIP top-k area allowed to receive wavelet residual")
    parser.add_argument("--wavelet_residual_condition_power", type=float, default=2.0, help="CLIP prior power for direct wavelet residual")
    parser.add_argument("--wavelet_residual_max_delta_ratio", type=float, default=0.03, help="maximum direct wavelet residual change relative to map score range")
    parser.add_argument("--wavelet_residual_local_contrast_kernel", type=int, default=17, help="local window for wavelet residual contrast")
    parser.add_argument("--wavelet_residual_local_contrast_weight", type=float, default=0.5, help="mixing weight for local wavelet residual contrast")
    parser.add_argument("--wavelet_residual_rank_gate_mode", type=str, default="hard", choices=["hard", "soft"], help="hard or soft CLIP-rank gate for direct wavelet residual")
    parser.add_argument("--wavelet_residual_rank_gate_temperature", type=float, default=0.05, help="sigmoid temperature for soft wavelet residual rank gate")
    parser.add_argument("--wavelet_residual_confidence_power", type=float, default=1.0, help="local CLIP-wavelet agreement power for direct wavelet residual")
    parser.add_argument("--use_low_rank_residual", action="store_true", help="add low-rank patch-feature reconstruction residual to the pixel map")
    parser.add_argument("--low_rank_weight", type=float, default=0.03, help="strength of low-rank residual map boost")
    parser.add_argument("--low_rank_rank_ratio", type=float, default=0.15, help="fraction of sampled patch-feature rank retained")
    parser.add_argument("--low_rank_max_rank", type=int, default=32, help="maximum retained low-rank basis size")
    parser.add_argument("--low_rank_sample_patches", type=int, default=128, help="deterministic patch samples used to fit the low-rank basis")
    parser.add_argument("--low_rank_fusion", type=str, default="mean", choices=["mean", "sum"], help="fusion strategy for layer-wise low-rank residual maps")
    parser.add_argument("--low_rank_condition_power", type=float, default=2.0, help="CLIP prior power for low-rank residual calibration")
    parser.add_argument("--low_rank_max_delta_ratio", type=float, default=0.03, help="maximum low-rank residual change relative to map score range")
    parser.add_argument("--low_rank_clip_topk_ratio", type=float, default=0.35, help="CLIP top-k area allowed to receive low-rank residual")
    parser.add_argument("--low_rank_clip_gate_mode", type=str, default="hard", choices=["hard", "soft"], help="hard or soft CLIP-rank gate for low-rank residual")
    parser.add_argument("--low_rank_clip_gate_temperature", type=float, default=0.05, help="sigmoid temperature for soft low-rank residual rank gate")
    parser.add_argument("--low_rank_no_center", action="store_true", help="disable centering before fitting the low-rank basis")
    parser.add_argument("--low_rank_layer_selection", type=str, default="all", choices=["all", "last"], help="selected patch-feature layers used by low-rank residual")
    parser.add_argument("--use_multicrop_fusion", action="store_true", help="fuse online multi-crop stitched anomaly map into the pixel map")
    parser.add_argument("--multicrop_weight", type=float, default=0.25, help="weight for multi-crop stitched map fusion")
    parser.add_argument("--multicrop_grid", type=int, default=2, help="multi-crop grid per spatial dimension")
    parser.add_argument("--multicrop_crop_ratio", type=float, default=0.75, help="crop size as a ratio of resized image size")
    parser.add_argument("--use_image_to_pixel_gate", action="store_true", help="use image anomaly confidence to gate pixel-level wavelet calibration")
    parser.add_argument("--image_to_pixel_weight", type=float, default=0.0, help="strength of image-to-pixel confidence gating")
    parser.add_argument("--image_to_pixel_power", type=float, default=1.0, help="power for image-to-pixel confidence gate")
    parser.add_argument("--image_to_pixel_min_gate", type=float, default=0.0, help="minimum image-to-pixel gate value")
    parser.add_argument("--image_to_pixel_max_gate", type=float, default=1.0, help="maximum image-to-pixel gate value")
    parser.add_argument("--use_pixel_to_image_fusion", action="store_true", help="fuse top-k pixel anomaly response back into image score")
    parser.add_argument("--pixel_to_image_weight", type=float, default=0.0, help="weight for pixel-to-image score fusion")
    parser.add_argument("--pixel_to_image_topk_ratio", type=float, default=0.01, help="top-k pixel ratio used for pixel-to-image fusion")
    parser.add_argument("--pixel_to_image_normalize", action="store_true", help="normalize each pixel map before pixel-to-image fusion")
    parser.add_argument("--use_tta_rectification", action="store_true", help="enable wavelet-guided test-time text feature rectification")
    parser.add_argument("--tta_mode", type=str, default="legacy", choices=["legacy", "wavelet_guided"], help="test-time text rectification strategy")
    parser.add_argument("--tta_alpha", type=float, default=0.2, help="text feature interpolation strength for test-time rectification")
    parser.add_argument("--tta_topk_ratio", type=float, default=0.05, help="fraction of reliable patches used as visual anchors")
    parser.add_argument("--tta_min_confidence", type=float, default=0.0, help="minimum anchor confidence required for text rectification")
    parser.add_argument("--tta_min_confidence_margin", type=float, default=0.0, help="required confidence gap between selected and opposite TTA anchors")
    parser.add_argument("--tta_anchor_layers", type=str, default="last", choices=["last", "mean"], help="use last selected layer or average anchors from all selected layers")
    parser.add_argument("--tta_update_abnormal", action="store_true", help="also rectify the abnormal text feature; default rectifies only normal")
    parser.add_argument("--tta_repulsion_weight", type=float, default=0.25, help="normal-text repulsion strength from abnormal visual anchors in wavelet_guided TTA")
    parser.add_argument("--tta_abnormal_alpha_scale", type=float, default=1.0, help="relative abnormal-text update strength in wavelet_guided TTA")
    
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)
