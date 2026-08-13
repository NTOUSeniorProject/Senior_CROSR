import os
import sys
import torch
from constants import CONFIG, KNOWN_ACTIONS, DEFAULT_KNOWN_ACTIONS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Functions"))

from Functions.ST_CROSR import ST_CROSR


def load_radar_meta_params(device):
    """加載雷達校正邊界與全域閾值"""
    global KNOWN_ACTIONS

    if not os.path.exists(CONFIG["radar_meta_path"]):
        raise FileNotFoundError(f"找不到 radar_meta_params 檔案：{CONFIG['radar_meta_path']}")

    meta = torch.load(CONFIG["radar_meta_path"], map_location=device)
    centroids_norm = meta["centroids_norm"].to(device).float()

    normalizer = {
        "dist_min": float(meta["dist_min"]),
        "dist_max": float(meta["dist_max"]),
        "mse_log_min": float(meta["mse_min"]),
        "mse_log_max": float(meta["mse_max"]),
    }

    threshold = float(meta["threshold"]) if CONFIG["use_saved_threshold"] else float(CONFIG["manual_threshold"])
    dist_weight = float(meta.get("dist_weight", 0.4))
    mse_weight = float(meta.get("mse_weight", 0.6))

    if "known_actions" in meta:
        KNOWN_ACTIONS = list(map(int, meta["known_actions"]))

    print("✅ 已成功加載雷達校正邊界與全域閾值")
    return centroids_norm, normalizer, threshold, dist_weight, mse_weight


def load_st_crosr_model(device):
    """加載 ST-CROSR 深度學習模型"""
    checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=device)

    num_classes = len(KNOWN_ACTIONS)
    model = ST_CROSR(
        num_known_classes=num_classes,
        num_nodes=CONFIG["num_nodes"],
        target_frames=CONFIG["max_frames"]
    ).to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("✅ ST-CROSR 模型神經網路載入完成")
    return model
