import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO

from Functions.ST_CROSR import ST_CROSR
from Functions.ntu_normalize import normalize_skeleton_batch, get_valid_mask


# ============================================================
# 設定區
# ============================================================
CONFIG = {
    # 實際測試影片
    "video_path": r"C:\CROSR\downloads\摔倒参考.mp4",

    # YOLO pose 權重
    "yolo_model_path": r"C:\CROSR\yolo26x-pose.pt",

    # ST-CROSR checkpoint
    "checkpoint_path": r"C:\CROSR\checkpoints_20260601_2301\best_val.pth",

    # 由 test_ntu_dataset.py 事先產生好的雷達校正參數
    "radar_meta_path": r"C:\CROSR\radar_meta_params.pth",

    # 跟訓練時一致
    "max_frames": 300,
    "num_nodes": 17,
    "center_joint_idx": 11,

    # 滑動視窗設定
    # 30 FPS 時：
    # window_size = 120 約 4 秒
    # stride = 30 約 1 秒
    "window_size": 120,
    "stride": 30,

    # 是否使用 radar_meta_params.pth 裡面的 threshold
    # True：使用 radar_meta_params.pth 內的 threshold
    # False：使用 manual_threshold
    "use_saved_threshold": True,

    # 手動 threshold
    # 只有 use_saved_threshold=False 時才會使用
    "manual_threshold": 0.1471,

    # 如果連續偵測異常，幾秒內不要重複警報
    "alert_cooldown_sec": 4,

    # 是否顯示 YOLO 處理畫面
    "show_yolo_window": True,
}


# ============================================================
# 預設 known actions
# 如果 checkpoint 或 radar_meta_params 裡有 known_actions，會自動覆蓋
# ============================================================
DEFAULT_KNOWN_ACTIONS = [
    1, 2, 3, 4, 5, 6,
    8, 9, 11, 12,
    14, 15, 16, 17, 18, 19, 20, 21,
    23, 25,
    28, 29, 30, 32, 33, 34, 37,
    41, 42, 43, 44, 45, 46, 47, 48, 49
]

KNOWN_ACTIONS = DEFAULT_KNOWN_ACTIONS.copy()


# ============================================================
# NTU60 動作名稱
# ============================================================
ACTION_NAMES = {
    1: "drink water",
    2: "eat meal/snack",
    3: "brushing teeth",
    4: "brushing hair",
    5: "drop",
    6: "pickup",
    7: "throw",
    8: "sitting down",
    9: "standing up",
    10: "clapping",
    11: "reading",
    12: "writing",
    13: "tear up paper",
    14: "wear jacket",
    15: "take off jacket",
    16: "wear shoe",
    17: "take off shoe",
    18: "wear glasses",
    19: "take off glasses",
    20: "put on hat/cap",
    21: "take off hat/cap",
    22: "cheer up",
    23: "hand waving",
    24: "kicking something",
    25: "reach into pocket",
    26: "hopping",
    27: "jump up",
    28: "make a phone call",
    29: "playing with phone/tablet",
    30: "typing on keyboard",
    31: "pointing to something",
    32: "taking a selfie",
    33: "check time",
    34: "rub two hands",
    35: "nod head/bow",
    36: "shake head",
    37: "wipe face",
    38: "salute",
    39: "put palms together",
    40: "cross hands in front",
    41: "sneeze/cough",
    42: "staggering",
    43: "falling",
    44: "touch head",
    45: "touch chest",
    46: "touch back",
    47: "touch neck",
    48: "nausea/vomiting",
    49: "use a fan",
    50: "punching/slapping",
    51: "kicking",
    52: "pushing",
    53: "pat on back",
    54: "point finger",
    55: "hugging",
    56: "giving object",
    57: "touch pocket",
    58: "shaking hands",
    59: "walking towards",
    60: "walking apart",
}


# ============================================================
# 載入雷達校正參數 radar_meta_params.pth
# ============================================================
def load_radar_meta_params(device):
    global KNOWN_ACTIONS

    if not os.path.exists(CONFIG["radar_meta_path"]):
        raise FileNotFoundError(
            f"找不到 radar_meta_params 檔案：{CONFIG['radar_meta_path']}\n"
            f"請先執行 test_ntu_dataset.py 產生 radar_meta_params.pth"
        )

    meta = torch.load(CONFIG["radar_meta_path"], map_location=device)

    required_keys = [
        "centroids_norm",
        "dist_min",
        "dist_max",
        "mse_min",
        "mse_max",
        "threshold",
    ]

    for key in required_keys:
        if key not in meta:
            raise KeyError(
                f"radar_meta_params.pth 缺少欄位：{key}"
            )

    centroids_norm = meta["centroids_norm"].to(device).float()

    normalizer = {
        "dist_min": float(meta["dist_min"]),
        "dist_max": float(meta["dist_max"]),

        # 注意：
        # radar_meta_params.pth 裡叫 mse_min / mse_max
        # 但它實際上應該是 log1p 後的 MSE min/max
        "mse_log_min": float(meta["mse_min"]),
        "mse_log_max": float(meta["mse_max"]),
    }

    if CONFIG["use_saved_threshold"]:
        threshold = float(meta["threshold"])
    else:
        threshold = float(CONFIG["manual_threshold"])

    dist_weight = float(meta.get("dist_weight", 0.4))
    mse_weight = float(meta.get("mse_weight", 0.6))

    if "known_actions" in meta:
        KNOWN_ACTIONS = list(map(int, meta["known_actions"]))
    else:
        print("⚠️ radar_meta_params.pth 沒有 known_actions，使用 DEFAULT_KNOWN_ACTIONS")

    print("✅ 已載入 radar_meta_params.pth")
    print(f"centroids_norm shape: {tuple(centroids_norm.shape)}")
    print(f"dist_min / dist_max: {normalizer['dist_min']:.6f} / {normalizer['dist_max']:.6f}")
    print(f"mse_log_min / mse_log_max: {normalizer['mse_log_min']:.6f} / {normalizer['mse_log_max']:.6f}")
    print(f"dist_weight: {dist_weight}")
    print(f"mse_weight: {mse_weight}")
    print(f"Combined threshold: {threshold:.6f}")
    print("known actions:")
    print(KNOWN_ACTIONS)

    return centroids_norm, normalizer, threshold, dist_weight, mse_weight


# ============================================================
# 載入 ST-CROSR 模型
# ============================================================
def load_st_crosr_model(device):
    global KNOWN_ACTIONS

    checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=device)

    if isinstance(checkpoint, dict) and "known_actions" in checkpoint:
        checkpoint_known_actions = checkpoint["known_actions"]
        print("✅ checkpoint 內有 known_actions")
        print(checkpoint_known_actions)

        if list(checkpoint_known_actions) != list(KNOWN_ACTIONS):
            print("⚠️ 注意：checkpoint known_actions 和 radar_meta_params known_actions 不完全一致")
            print("checkpoint known_actions:", checkpoint_known_actions)
            print("radar_meta_params known_actions:", KNOWN_ACTIONS)
            print("這可能造成分類 index 對應錯誤，請確認兩者來自同一次訓練。")
    else:
        print("⚠️ checkpoint 沒有 known_actions，使用 radar_meta_params / DEFAULT_KNOWN_ACTIONS")

    num_classes = len(KNOWN_ACTIONS)

    model = ST_CROSR(
        num_known_classes=num_classes,
        num_nodes=CONFIG["num_nodes"],
        target_frames=CONFIG["max_frames"]
    ).to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

        if "best_val_accuracy" in checkpoint:
            print(f"best val accuracy: {checkpoint['best_val_accuracy']:.2f}%")
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("✅ ST-CROSR 模型載入完成")

    return model


# ============================================================
# 從影片抽整段 YOLO 17 點骨架
# 輸出 shape: [2, T, 17]
# ============================================================
def extract_full_skeleton_from_video(video_path, yolo_model):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30

    print(f"影片 FPS: {fps:.2f}")
    print(f"影片總幀數: {total_frames}")

    skeleton_list = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        one_frame_skeleton = np.zeros((2, 17), dtype=np.float32)

        results = yolo_model(frame, verbose=False)

        if len(results) > 0 and results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy

            if keypoints is not None and len(keypoints) > 0:
                # 只取第一個人
                person_kpts = keypoints[0].cpu().numpy()

                if person_kpts.shape[0] >= 17:
                    one_frame_skeleton[0, :] = person_kpts[:17, 0]
                    one_frame_skeleton[1, :] = person_kpts[:17, 1]

        skeleton_list.append(one_frame_skeleton)

        if CONFIG["show_yolo_window"]:
            cv2.imshow("video", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"已處理 {frame_idx}/{total_frames} frames")

    cap.release()
    cv2.destroyAllWindows()

    if len(skeleton_list) == 0:
        raise RuntimeError("影片沒有成功讀到任何 frame")

    # [T, 2, 17] -> [2, T, 17]
    full_skeleton = np.stack(skeleton_list, axis=0)
    full_skeleton = np.transpose(full_skeleton, (1, 0, 2))

    print("✅ 骨架抽取完成")
    print("full_skeleton shape:", full_skeleton.shape)

    return full_skeleton, fps


# ============================================================
# 補成 300 幀
# input: [2, T, 17]
# output: [2, 300, 17]
# ============================================================
def pad_or_cut_to_300(clip):
    C, T, V = clip.shape
    target_frames = CONFIG["max_frames"]

    output = np.zeros((C, target_frames, V), dtype=np.float32)

    if T >= target_frames:
        output = clip[:, :target_frames, :]
    else:
        output[:, :T, :] = clip

    return output


# ============================================================
# 計算 Combined Score
# 跟 test_ntu_dataset.py 的方法一致：
# mse_log = log1p(mse)
# norm_dist = minmax(dist)
# norm_mse = minmax(mse_log)
# combined = norm_dist * 0.4 + norm_mse * 0.6
# ============================================================
def compute_combined_score(
    mse_score,
    dist_score,
    normalizer,
    dist_weight,
    mse_weight
):
    mse_log = np.log1p(mse_score)

    norm_dist = (
        (dist_score - normalizer["dist_min"])
        / (normalizer["dist_max"] - normalizer["dist_min"] + 1e-8)
    )

    norm_mse = (
        (mse_log - normalizer["mse_log_min"])
        / (normalizer["mse_log_max"] - normalizer["mse_log_min"] + 1e-8)
    )

    combined_score = norm_dist * dist_weight + norm_mse * mse_weight

    return float(combined_score), float(norm_dist), float(norm_mse), float(mse_log)


# ============================================================
# 單一視窗推論
# clip shape: [2, 300, 17]
# ============================================================
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
    clip_tensor = torch.tensor(
        clip_np,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    # valid_mask 要在 normalize 前取
    valid_mask = get_valid_mask(clip_tensor)

    clip_tensor = normalize_skeleton_batch(
        clip_tensor,
        center_joint_idx=CONFIG["center_joint_idx"]
    )

    clip_tensor = clip_tensor.contiguous()

    with torch.no_grad():
        outputs = model(clip_tensor)

        if len(outputs) == 4:
            logits, recon_x, z, w_hat = outputs
        elif len(outputs) == 3:
            logits, recon_x, z = outputs
            w_hat = None
        elif len(outputs) == 2:
            logits, recon_x = outputs
            z = None
            w_hat = None
        else:
            raise RuntimeError(
                f"模型 forward 輸出數量不正確，目前是 {len(outputs)} 個"
            )

        if z is None:
            raise RuntimeError("Combined Score 需要 z 特徵，但模型沒有回傳 z")

        # ----------------------------------------------------
        # 1. 已知動作分類結果
        # ----------------------------------------------------
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

        pred_action_id = KNOWN_ACTIONS[pred_idx]
        pred_action_name = ACTION_NAMES.get(
            pred_action_id,
            "unknown action name"
        )

        # ----------------------------------------------------
        # 2. Masked MSE
        # ----------------------------------------------------
        squared_diff = (recon_x - clip_tensor) ** 2
        masked_diff = squared_diff * valid_mask

        mse_score = torch.sum(masked_diff) / (torch.sum(valid_mask) + 1e-6)
        mse_score = mse_score.item()

        # ----------------------------------------------------
        # 3. Cosine Distance
        # 跟 threshold 腳本一致：
        # z 跟所有已知類別 centroids 比，取最大 similarity
        # dist = 1 - max_similarity
        # ----------------------------------------------------
        z_norm = F.normalize(z, p=2, dim=1)
        cos_sim = torch.mm(z_norm, centroids_norm.t())
        max_sim, nearest_class_idx = torch.max(cos_sim, dim=1)

        dist_score = 1.0 - max_sim
        dist_score = dist_score.item()
        nearest_class_idx = nearest_class_idx.item()

        nearest_action_id = KNOWN_ACTIONS[nearest_class_idx]
        nearest_action_name = ACTION_NAMES.get(
            nearest_action_id,
            "unknown action name"
        )

        # ----------------------------------------------------
        # 4. Combined Score
        # ----------------------------------------------------
        combined_score, norm_dist, norm_mse, mse_log = compute_combined_score(
            mse_score=mse_score,
            dist_score=dist_score,
            normalizer=normalizer,
            dist_weight=dist_weight,
            mse_weight=mse_weight
        )

        is_unknown = combined_score >= threshold

    return {
        "pred_idx": pred_idx,
        "action_id": pred_action_id,
        "action_name": pred_action_name,
        "confidence": confidence,

        "nearest_class_idx": nearest_class_idx,
        "nearest_action_id": nearest_action_id,
        "nearest_action_name": nearest_action_name,

        "mse_score": mse_score,
        "mse_log": mse_log,
        "dist_score": dist_score,
        "norm_dist": norm_dist,
        "norm_mse": norm_mse,
        "combined_score": combined_score,
        "is_unknown": is_unknown,
    }


# ============================================================
# 滑動視窗測整部影片
# ============================================================
def sliding_window_predict(
    model,
    full_skeleton,
    fps,
    device,
    centroids_norm,
    normalizer,
    threshold,
    dist_weight,
    mse_weight
):
    C, T, V = full_skeleton.shape

    window_size = CONFIG["window_size"]
    stride = CONFIG["stride"]

    results = []
    last_alert_time = -999999

    print("\n==============================")
    print("開始滑動視窗推論")
    print("==============================")
    print(f"full video shape: {full_skeleton.shape}")
    print(f"window_size: {window_size} frames")
    print(f"stride: {stride} frames")
    print(f"Combined threshold: {threshold:.6f}")

    if T < window_size:
        starts = [0]
    else:
        starts = list(range(0, T - window_size + 1, stride))

        last_start = T - window_size

        if starts[-1] != last_start:
            starts.append(last_start)

    for i, start in enumerate(starts):
        end = min(start + window_size, T)

        clip = full_skeleton[:, start:end, :]
        clip = pad_or_cut_to_300(clip)

        result = predict_one_clip(
            model=model,
            clip_np=clip,
            device=device,
            centroids_norm=centroids_norm,
            normalizer=normalizer,
            threshold=threshold,
            dist_weight=dist_weight,
            mse_weight=mse_weight
        )

        start_sec = start / fps
        end_sec = end / fps

        result["window_index"] = i + 1
        result["start_frame"] = start
        result["end_frame"] = end
        result["start_sec"] = start_sec
        result["end_sec"] = end_sec

        results.append(result)

        status = "未知/異常" if result["is_unknown"] else "已知"

        if result["is_unknown"]:
            print(
                f"[{i+1:03d}] "
                f"{start_sec:6.2f}s ~ {end_sec:6.2f}s | "
                f"未知/異常動作 | "
                f"combined={result['combined_score']:.4f} | "
                f"threshold={threshold:.4f}"
            )
        else:
            print(
                f"[{i+1:03d}] "
                f"{start_sec:6.2f}s ~ {end_sec:6.2f}s | "
                f"已知動作 | "
                f"A{result['action_id']:03d} {result['action_name']} | "
                f"conf={result['confidence']:.4f} | "
                f"combined={result['combined_score']:.4f}"
            )

        if result["is_unknown"]:
            current_time = start_sec

            if current_time - last_alert_time >= CONFIG["alert_cooldown_sec"]:
                print("🚨 發出警報：Combined Score 超過 Threshold")
                last_alert_time = current_time
            else:
                print("⚠️ 異常仍存在，但冷卻時間內不重複警報")

    return results


# ============================================================
# 統整結果
# ============================================================
def summarize_results(results, threshold):
    print("\n==============================")
    print("影片測試總結")
    print("==============================")

    total_windows = len(results)
    known_windows = [r for r in results if not r["is_unknown"]]
    abnormal_windows = [r for r in results if r["is_unknown"]]

    print(f"Combined threshold: {threshold:.6f}")
    print(f"總共判斷視窗數: {total_windows}")
    print(f"已知視窗數: {len(known_windows)}")
    print(f"未知 / 異常視窗數: {len(abnormal_windows)}")

    if total_windows == 0:
        print("沒有產生任何滑動視窗，請檢查影片長度、window_size、stride")
        return

    action_count = {}

    for r in results:
        action_id = r["action_id"]
        action_name = r["action_name"]
        key = f"A{action_id:03d} {action_name}"

        if key not in action_count:
            action_count[key] = 0

        action_count[key] += 1

    print("\n模型分類結果統計：")

    for action, count in sorted(
        action_count.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"- {action}: {count} 次")

    falling_count = sum(1 for r in results if r["action_id"] == 43)
    staggering_count = sum(1 for r in results if r["action_id"] == 42)
    vomiting_count = sum(1 for r in results if r["action_id"] == 48)

    print(f"\nA043 falling 視窗數: {falling_count}")
    print(f"A042 staggering 視窗數: {staggering_count}")
    print(f"A048 nausea/vomiting 視窗數: {vomiting_count}")

    # 危險動作：已知危險類別 + unknown
    danger_windows = []

    for r in results:
        if r["action_id"] in [42, 43, 48] or r["is_unknown"]:
            danger_windows.append(r)

    print("\n==============================")
    print("危險行為判斷")
    print("==============================")

    if len(danger_windows) > 0:
        print("✅ 判斷：影片中有偵測到可能危險行為")

        for r in danger_windows:
            danger_type = "未知/異常" if r["is_unknown"] else "已知危險動作"

            print(
                f"- {r['start_sec']:.2f}s ~ {r['end_sec']:.2f}s | "
                f"{danger_type} | "
                f"Pred: A{r['action_id']:03d} {r['action_name']} | "
                f"Nearest: A{r['nearest_action_id']:03d} {r['nearest_action_name']} | "
                f"conf={r['confidence']:.4f} | "
                f"combined={r['combined_score']:.4f}"
            )
    else:
        print("⚠️ 判斷：影片中沒有明顯危險行為")

    if len(abnormal_windows) > 0:
        print("\n偵測到未知 / 異常的時間段：")

        for r in abnormal_windows:
            print(
                f"- {r['start_sec']:.2f}s ~ {r['end_sec']:.2f}s | "
                f"Pred: A{r['action_id']:03d} {r['action_name']} | "
                f"Nearest: A{r['nearest_action_id']:03d} {r['nearest_action_name']} | "
                f"conf={r['confidence']:.4f} | "
                f"mse={r['mse_score']:.4f} | "
                f"dist={r['dist_score']:.4f} | "
                f"combined={r['combined_score']:.4f}"
            )
    else:
        print("\n沒有偵測到未知 / 異常動作。")
        print("注意：這不代表沒有偵測到動作，只代表 Combined Score 沒有超過 threshold。")


# ============================================================
# 主程式
# ============================================================
def main():
    print("\n==============================")
    print("ST-CROSR 實際影片測試")
    print("使用已儲存的 Radar Meta Params")
    print("==============================")

    if not os.path.exists(CONFIG["video_path"]):
        raise FileNotFoundError(f"找不到影片：{CONFIG['video_path']}")

    if not os.path.exists(CONFIG["checkpoint_path"]):
        raise FileNotFoundError(f"找不到 checkpoint：{CONFIG['checkpoint_path']}")

    if not os.path.exists(CONFIG["yolo_model_path"]):
        raise FileNotFoundError(f"找不到 YOLO 權重：{CONFIG['yolo_model_path']}")

    if not os.path.exists(CONFIG["radar_meta_path"]):
        raise FileNotFoundError(f"找不到 radar_meta_params：{CONFIG['radar_meta_path']}")

    print("\n目前設定：")
    print(f"影片路徑: {CONFIG['video_path']}")
    print(f"YOLO 權重: {CONFIG['yolo_model_path']}")
    print(f"Checkpoint: {CONFIG['checkpoint_path']}")
    print(f"Radar Meta Params: {CONFIG['radar_meta_path']}")
    print(f"use_saved_threshold: {CONFIG['use_saved_threshold']}")
    print(f"manual_threshold: {CONFIG['manual_threshold']}")
    print(f"center_joint_idx: {CONFIG['center_joint_idx']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n使用裝置:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # 先載入 Radar Meta Params，取得 KNOWN_ACTIONS 與 threshold
    centroids_norm, normalizer, threshold, dist_weight, mse_weight = load_radar_meta_params(device)

    # 再依照 KNOWN_ACTIONS 建立模型
    model = load_st_crosr_model(device)

    yolo_model = YOLO(CONFIG["yolo_model_path"])

    full_skeleton, fps = extract_full_skeleton_from_video(
        CONFIG["video_path"],
        yolo_model
    )

    print("\n==============================")
    print("骨架資料檢查")
    print("==============================")
    print("full_skeleton shape:", full_skeleton.shape)
    print("骨架最大值:", full_skeleton.max())
    print("骨架最小值:", full_skeleton.min())
    print("骨架非零比例:", np.count_nonzero(full_skeleton) / full_skeleton.size)

    results = sliding_window_predict(
        model=model,
        full_skeleton=full_skeleton,
        fps=fps,
        device=device,
        centroids_norm=centroids_norm,
        normalizer=normalizer,
        threshold=threshold,
        dist_weight=dist_weight,
        mse_weight=mse_weight
    )

    summarize_results(results, threshold)


if __name__ == "__main__":
    main()