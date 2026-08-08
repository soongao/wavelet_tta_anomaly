import hashlib
import math
import os
import urllib
import warnings
from typing import Union, List
from pkg_resources import packaging

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm
import numpy as np

from .build_model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from torchvision.transforms import InterpolationMode

if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


__all__ = [
    "available_models",
    "load",
    "get_similarity_map",
    "compute_similarity",
    "compute_transport_anomaly",
    "feature_to_poincare",
    "poincare_distance",
]
_tokenizer = _Tokenizer()

_MODELS = {
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(
        url: str,
        cache_dir: Union[str, None] = None,
):

    if not cache_dir:
        # cache_dir = os.path.expanduser("~/.cache/clip")
        cache_dir = os.path.expanduser("/remote-home/iot_zhouqihang/root/.cache/clip")
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)

    if 'openaipublic' in url:
        expected_sha256 = url.split("/")[-2]
    elif 'mlfoundations' in url:
        expected_sha256 = os.path.splitext(filename)[0].split("-")[-1]
    else:
        expected_sha256 = ''

    download_target = os.path.join(cache_dir, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if expected_sha256:
            if hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(expected_sha256):
                return download_target
            else:
                warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")
        else:
            return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.headers.get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if expected_sha256 and not hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(expected_sha256):
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize((n_px, n_px), interpolation=InterpolationMode.BICUBIC),
        #CenterCrop(n_px), # rm center crop to explain whole image
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict

def load_checkpoint(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)
    # detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)
    resize_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys

def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", design_details = None, jit: bool = False, download_root: str = None):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    print("name", name)
    if name in _MODELS:
        # model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("/Users/bytedance/code/env/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        model = build_model(name, state_dict or model.state_dict(), design_details).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())


def get_similarity_map(sm, shape):
    side = int(sm.shape[1] ** 0.5)
    sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)
    sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear')
    sm = sm.permute(0, 2, 3, 1)
    return sm


def compute_similarity(image_features, text_features, t=2):
    prob_1 = image_features[:, :1, :] @ text_features.t()
    b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], image_features.shape[1], image_features.shape[2]
    feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
    similarity = feats.sum(-1)
    return (similarity/0.07).softmax(-1), prob_1


def _as_curvature(curvature, reference, eps=1e-8):
    return torch.as_tensor(curvature, dtype=reference.dtype, device=reference.device).clamp_min(eps)


def _artanh(x, eps=1e-5):
    x = x.clamp(min=-1 + eps, max=1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _project_to_ball(x, curvature, eps=1e-5):
    c = _as_curvature(curvature, x, eps)
    sqrt_c = torch.sqrt(c)
    max_norm = (1.0 - eps) / sqrt_c
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return x * scale


def _mobius_add(x, y, curvature, eps=1e-5):
    c = _as_curvature(curvature, x, eps)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denominator = (1 + 2 * c * xy + c * c * x2 * y2).clamp_min(eps)
    return _project_to_ball(numerator / denominator, c, eps)


def feature_to_poincare(features, curvature=1.0, radius_scale=0.1, eps=1e-5):
    """Lift raw CLIP features to the Poincare ball while preserving bounded norm information."""
    features = features.float()
    c = _as_curvature(curvature, features, eps)
    sqrt_c = torch.sqrt(c)
    norm = features.norm(dim=-1, keepdim=True)
    direction = features / norm.clamp_min(eps)
    tangent = direction * torch.log1p(norm).mul(radius_scale)
    tangent_norm = tangent.norm(dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.tanh(sqrt_c * tangent_norm) / (sqrt_c * tangent_norm)
    return _project_to_ball(scale * tangent, c, eps)


def poincare_distance(x, y, curvature=1.0, eps=1e-5):
    c = _as_curvature(curvature, x, eps)
    sqrt_c = torch.sqrt(c)
    delta = _mobius_add(-x, y, c, eps)
    delta_norm = delta.norm(dim=-1).clamp_min(eps)
    return 2.0 / sqrt_c * _artanh(sqrt_c * delta_norm, eps)


def _logmap(x, y, curvature=1.0, eps=1e-5):
    c = _as_curvature(curvature, x, eps)
    sqrt_c = torch.sqrt(c)
    delta = _mobius_add(-x, y, c, eps)
    delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(eps)
    lambda_x = 2.0 / (1.0 - c * (x * x).sum(dim=-1, keepdim=True)).clamp_min(eps)
    scale = 2.0 / (sqrt_c * lambda_x) * _artanh(sqrt_c * delta_norm, eps) / delta_norm
    return scale * delta


def _cone_violation(parent, child, curvature=1.0, cone_aperture=0.1, eps=1e-5):
    c = _as_curvature(curvature, parent, eps)
    sqrt_c = torch.sqrt(c)
    origin = torch.zeros_like(parent)
    outward_axis = -_logmap(parent, origin, c, eps)
    child_direction = _logmap(parent, child, c, eps)
    dot = (outward_axis * child_direction).sum(dim=-1)
    axis_norm = outward_axis.norm(dim=-1).clamp_min(eps)
    child_norm = child_direction.norm(dim=-1).clamp_min(eps)
    angle = torch.acos((dot / (axis_norm * child_norm)).clamp(-1 + eps, 1 - eps))

    parent_norm = parent.norm(dim=-1).clamp_min(eps)
    aperture_arg = cone_aperture * (1.0 - c * parent_norm.pow(2)) / (sqrt_c * parent_norm)
    aperture = torch.asin(aperture_arg.clamp(0.0, 1.0 - eps))
    return F.relu(angle - aperture) / math.pi


def _energy_to_logits(energy, margin=0.2, temperature=1.0):
    logits = torch.stack([margin - energy, energy - margin], dim=-1)
    return logits / temperature


def _transport_anchors(text_features, anchor_mode="normal"):
    if text_features.dim() == 3:
        text_features = text_features[0]
    if anchor_mode == "normal":
        return text_features[:1]
    if anchor_mode in {"both", "normal_anomaly"}:
        return text_features[:2]
    if anchor_mode == "anomaly":
        return text_features[1:2]
    raise ValueError(f"unknown ot_anchor_mode: {anchor_mode}")


def _transport_cost_matrix(
    patch_features,
    text_features,
    cost_type="hyperbolic_cone",
    anchor_mode="normal",
    curvature=1.0,
    radius_scale=0.1,
    cone_aperture=0.1,
    eps=1e-5,
):
    anchors = _transport_anchors(text_features, anchor_mode).float()
    patches = patch_features.float()

    if cost_type == "cosine":
        patch_point = F.normalize(patches, dim=-1, eps=eps)
        anchor_point = F.normalize(anchors, dim=-1, eps=eps)
        return 1.0 - torch.einsum("bnd,md->bnm", patch_point, anchor_point)

    if cost_type == "euclidean":
        patch_point = F.normalize(patches, dim=-1, eps=eps)
        anchor_point = F.normalize(anchors, dim=-1, eps=eps)
        return (patch_point.unsqueeze(2) - anchor_point.view(1, 1, -1, patch_point.shape[-1])).norm(dim=-1)

    patch_point = feature_to_poincare(patches, curvature, radius_scale, eps)
    anchor_point = feature_to_poincare(anchors, curvature, radius_scale, eps)
    if cost_type == "hyperbolic_distance":
        return poincare_distance(
            patch_point.unsqueeze(2),
            anchor_point.view(1, 1, -1, patch_point.shape[-1]),
            curvature,
            eps,
        )
    if cost_type == "hyperbolic_cone":
        return _cone_violation(
            anchor_point.view(1, 1, -1, patch_point.shape[-1]),
            patch_point.unsqueeze(2),
            curvature,
            cone_aperture,
            eps,
        )
    raise ValueError(f"unknown ot_cost: {cost_type}")


def _balanced_sinkhorn(cost, a, b, epsilon=0.05, iterations=50, eps=1e-8):
    kernel = torch.exp(-cost / max(float(epsilon), eps)).clamp_min(eps)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(iterations):
        kv = torch.matmul(kernel, v.unsqueeze(-1)).squeeze(-1).clamp_min(eps)
        u = a / kv
        ktu = torch.matmul(kernel.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1).clamp_min(eps)
        v = b / ktu
    return u.unsqueeze(-1) * kernel * v.unsqueeze(1)


def _unbalanced_sinkhorn(
    cost,
    a,
    b,
    epsilon=0.05,
    tau_patch=0.5,
    tau_anchor=0.5,
    iterations=50,
    eps=1e-8,
):
    kernel = torch.exp(-cost / max(float(epsilon), eps)).clamp_min(eps)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    patch_power = float(tau_patch) / (float(tau_patch) + float(epsilon) + eps)
    anchor_power = float(tau_anchor) / (float(tau_anchor) + float(epsilon) + eps)
    for _ in range(iterations):
        kv = torch.matmul(kernel, v.unsqueeze(-1)).squeeze(-1).clamp_min(eps)
        u = (a / kv).clamp_min(eps).pow(patch_power)
        ktu = torch.matmul(kernel.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1).clamp_min(eps)
        v = (b / ktu).clamp_min(eps).pow(anchor_power)
    return u.unsqueeze(-1) * kernel * v.unsqueeze(1)


def _partial_transport(cost, a, b, partial_mass=0.9, epsilon=0.05, iterations=50, eps=1e-8):
    batch, patch_count, anchor_count = cost.shape
    keep_count = max(1, min(patch_count, int(math.ceil(patch_count * float(partial_mass)))))
    min_cost = cost.min(dim=-1).values
    keep_indices = min_cost.topk(keep_count, largest=False, dim=1).indices

    gamma = torch.zeros_like(cost)
    for batch_idx in range(batch):
        sub_cost = cost[batch_idx:batch_idx + 1, keep_indices[batch_idx], :]
        sub_a = torch.full(
            (1, keep_count),
            float(partial_mass) / keep_count,
            device=cost.device,
            dtype=cost.dtype,
        )
        sub_b = torch.full(
            (1, anchor_count),
            float(partial_mass) / anchor_count,
            device=cost.device,
            dtype=cost.dtype,
        )
        sub_gamma = _balanced_sinkhorn(sub_cost, sub_a, sub_b, epsilon, iterations, eps)[0]
        gamma[batch_idx, keep_indices[batch_idx], :] = sub_gamma
    return gamma


def _transport_plan(
    cost,
    mode="unbalanced",
    epsilon=0.05,
    tau_patch=0.5,
    tau_anchor=0.5,
    partial_mass=0.9,
    iterations=50,
    eps=1e-8,
):
    batch, patch_count, anchor_count = cost.shape
    a = torch.full((batch, patch_count), 1.0 / patch_count, device=cost.device, dtype=cost.dtype)
    b = torch.full((batch, anchor_count), 1.0 / anchor_count, device=cost.device, dtype=cost.dtype)

    if mode == "balanced":
        gamma = _balanced_sinkhorn(cost, a, b, epsilon, iterations, eps)
    elif mode == "partial":
        gamma = _partial_transport(cost, a, b, partial_mass, epsilon, iterations, eps)
    elif mode == "unbalanced":
        gamma = _unbalanced_sinkhorn(cost, a, b, epsilon, tau_patch, tau_anchor, iterations, eps)
    else:
        raise ValueError(f"unknown ot_mode: {mode}")
    return gamma, a, b


def compute_transport_anomaly(
    patch_features,
    text_features,
    ot_mode="unbalanced",
    ot_cost="hyperbolic_cone",
    ot_anchor_mode="normal",
    ot_epsilon=0.05,
    ot_tau_patch=0.5,
    ot_tau_anchor=0.5,
    ot_partial_mass=0.9,
    ot_iterations=50,
    ot_score="combined",
    ot_alpha=1.0,
    ot_beta=1.0,
    curvature=1.0,
    temperature=1.0,
    radius_scale=0.1,
    cone_aperture=0.1,
    margin=0.2,
    eps=1e-5,
):
    """Score patches with balanced, partial, or unbalanced transport to text anchors."""
    cost = _transport_cost_matrix(
        patch_features,
        text_features,
        cost_type=ot_cost,
        anchor_mode=ot_anchor_mode,
        curvature=curvature,
        radius_scale=radius_scale,
        cone_aperture=cone_aperture,
        eps=eps,
    ).clamp_min(0.0)
    gamma, source_mass, _ = _transport_plan(
        cost,
        mode=ot_mode,
        epsilon=ot_epsilon,
        tau_patch=ot_tau_patch,
        tau_anchor=ot_tau_anchor,
        partial_mass=ot_partial_mass,
        iterations=ot_iterations,
        eps=eps,
    )

    transported_mass = gamma.sum(dim=-1)
    unmatched_mass = (source_mass - transported_mass).clamp_min(0.0)
    unmatched_ratio = unmatched_mass / source_mass.clamp_min(eps)
    matched_cost = (gamma * cost).sum(dim=-1) / transported_mass.clamp_min(eps)
    transport_entropy = -(gamma.clamp_min(eps) * gamma.clamp_min(eps).log()).sum(dim=-1)

    if ot_score == "unmatched":
        score = unmatched_ratio
    elif ot_score == "cost":
        score = matched_cost
    elif ot_score == "combined":
        score = float(ot_alpha) * unmatched_ratio + float(ot_beta) * matched_cost
    else:
        raise ValueError(f"unknown ot_score: {ot_score}")

    similarity = _energy_to_logits(score, margin, temperature).softmax(dim=-1)
    components = {
        "transport_score": score,
        "unmatched_mass": unmatched_mass,
        "unmatched_ratio": unmatched_ratio,
        "matched_cost": matched_cost,
        "transported_mass": transported_mass,
        "transport_entropy": transport_entropy,
        "min_transport_cost": cost.min(dim=-1).values,
    }
    return similarity, score, components
