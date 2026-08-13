import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from constants import CONFIG, KNOWN_ACTIONS, ACTION_NAMES

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Functions"))

from Functions.ntu_normalize import normalize_skeleton_batch, get_valid_mask


def pad_or_cut_to_300(clip):
    """將骨架視窗調整為 300 幀"""
    C, T, V = clip.shape
    target_frames = CONFIG["max_frames"]

    output = np.zeros((C, target_frames, V), dtype=np.float32)

    if T >= target_frames:
        output = clip[:, :target_frames, :]
    else:
        output[:, :T, :] = clip

    return output


def compute_combined_score(mse_score, dist_score, normalizer, dist_weight, mse_weight):
    """計算融合異常分數"""
    mse_log = np.log1p(mse_score)

    norm_dist = (
        (dist_score - normalizer["dist_min"]) /
        (normalizer["dist_max"] - normalizer["dist_min"] + 1e-8)
    )

    norm_mse = (
        (mse_log - normalizer["mse_log_min"]) /
        (normalizer["mse_log_max"] - normalizer["mse_log_min"] + 1e-8)
    )

    combined_score = norm_dist * dist_weight + norm_mse * mse_weight

    return float(combined_score), float(norm_dist), float(norm_mse), float(mse_log)


def predict_one_clip(
    model,
    clip_np,
    device,
    centroids_norm,
    normalizer,
    threshold,
    dist_weight,
    mse_weight
):
    """對單一骨架視窗進行推論"""
    clip_tensor = torch.tensor(clip_np, dtype=torch.float32).unsqueeze(0).to(device)

    valid_mask = get_valid_mask(clip_tensor)

    clip_tensor = normalize_skeleton_batch(
        clip_tensor,
        center_joint_idx=CONFIG["center_joint_idx"]
    ).contiguous()

    with torch.no_grad():
        outputs = model(clip_tensor)
        logits, recon_x, z, _ = outputs

        # 1. 已知動作分類
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

        pred_action_id = KNOWN_ACTIONS[pred_idx]
        pred_action_name = ACTION_NAMES.get(pred_action_id, "unknown")

        # 2. Masked MSE
        squared_diff = (recon_x - clip_tensor) ** 2
        masked_diff = squared_diff * valid_mask
        mse_score = (
            torch.sum(masked_diff) /
            (torch.sum(valid_mask) + 1e-6)
        ).item()

        # 3. Cosine Distance
        z_norm = F.normalize(z, p=2, dim=1)
        cos_sim = torch.mm(z_norm, centroids_norm.t())

        max_sim, nearest_class_idx = torch.max(cos_sim, dim=1)
        dist_score = (1.0 - max_sim).item()

        nearest_action_id = KNOWN_ACTIONS[nearest_class_idx.item()]
        nearest_action_name = ACTION_NAMES.get(nearest_action_id, "unknown")

        # 4. Combined Fusion
        combined_score, norm_dist, norm_mse, mse_log = compute_combined_score(
            mse_score,
            dist_score,
            normalizer,
            dist_weight,
            mse_weight
        )

        is_unknown = combined_score >= threshold

    return {
        "action_id": pred_action_id,
        "action_name": pred_action_name,
        "confidence": confidence,
        "nearest_action_id": nearest_action_id,
        "nearest_action_name": nearest_action_name,
        "combined_score": combined_score,
        "is_unknown": is_unknown,
        "mse_score": mse_score,
        "dist_score": dist_score,
        "norm_dist": norm_dist,
        "norm_mse": norm_mse,
        "mse_log": mse_log,
    }
