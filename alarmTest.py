import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO

from ST_CROSR import ST_CROSR
from ntu_normalize import normalize_skeleton_batch, get_valid_mask


# ============================================================
# 設定區
# ============================================================
CONFIG = {
    # 實際測試影片
    # "video_path": r"C:\CROSR\downloads\摔倒参考.mp4",
    "video_path": r"C:\CROSR\IMG_2033.mov",

    # YOLO pose 權重
    "yolo_model_path": r"C:\CROSR\yolo26x-pose.pt",

    # ST-CROSR checkpoint
    "checkpoint_path": r"checkpoints_20260602_2237\best_val.pth",

    # 由 test_ntu_dataset.py 事先產生好的雷達校正參數
    "radar_meta_path": r"C:\CROSR\radar_meta_params.pth",

    # 跟訓練時一致
    "max_frames": 300,
    "num_nodes": 17,
    "center_joint_idx": 11,

    # 滑動視窗設定 (30 FPS 時：window_size=120 約 4 秒，stride=30 約 1 秒更新一次雷達)
    "window_size": 120,
    "stride": 30,

    # 是否使用 radar_meta_params.pth 裡面的 threshold
    "use_saved_threshold": True,

    # 手動 threshold (只有 use_saved_threshold=False 時才會使用)
    "manual_threshold": 0.1471,

    # 如果連續偵測異常，幾秒內不要重複警報
    "alert_cooldown_sec": 4,

    # 是否顯示即時監控顯示畫面
    "show_yolo_window": True,
}


DEFAULT_KNOWN_ACTIONS = [
    1, 2, 3, 4, 5, 6,
    8, 9, 11, 12,
    14, 15, 16, 17, 18, 19, 20, 21,
    23, 25,
    28, 29, 30, 32, 33, 34, 37,
    41, 44, 45, 46, 47, 49
]

KNOWN_ACTIONS = DEFAULT_KNOWN_ACTIONS.copy()

ACTION_NAMES = {
    1: "drink water", 2: "eat meal/snack", 3: "brushing teeth", 4: "brushing hair",
    5: "drop", 6: "pickup", 7: "throw", 8: "sitting down", 9: "standing up",
    10: "clapping", 11: "reading", 12: "writing", 13: "tear up paper",
    14: "wear jacket", 15: "take off jacket", 16: "wear shoe", 17: "take off shoe",
    18: "wear glasses", 19: "take off glasses", 20: "put on hat/cap", 21: "take off hat/cap",
    22: "cheer up", 23: "hand waving", 24: "kicking something", 25: "reach into pocket",
    26: "hopping", 27: "jump up", 28: "make a phone call", 29: "playing with phone/tablet",
    30: "typing on keyboard", 31: "pointing to something", 32: "taking a selfie",
    33: "check time", 34: "rub two hands", 35: "nod head/bow", 36: "shake head",
    37: "wipe face", 38: "salute", 39: "put palms together", 40: "cross hands in front",
    41: "sneeze/cough", 42: "staggering", 43: "falling", 44: "touch head",
    45: "touch chest", 46: "touch back", 47: "touch neck", 48: "nausea/vomiting",
    49: "use a fan", 50: "punching/slapping", 51: "kicking", 52: "pushing",
    53: "pat on back", 54: "point finger", 55: "hugging", 56: "giving object",
    57: "touch pocket", 58: "shaking hands", 59: "walking towards", 60: "walking apart",
}


def load_radar_meta_params(device):
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
    checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=device)
    num_classes = len(KNOWN_ACTIONS)
    model = ST_CROSR(num_known_classes=num_classes, num_nodes=CONFIG["num_nodes"], target_frames=CONFIG["max_frames"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint)
    model.eval()
    print("✅ ST-CROSR 模型神經網路載入完成")
    return model


def pad_or_cut_to_300(clip):
    C, T, V = clip.shape
    target_frames = CONFIG["max_frames"]
    output = np.zeros((C, target_frames, V), dtype=np.float32)
    if T >= target_frames:
        output = clip[:, :target_frames, :]
    else:
        output[:, :T, :] = clip
    return output


def compute_combined_score(mse_score, dist_score, normalizer, dist_weight, mse_weight):
    mse_log = np.log1p(mse_score)
    norm_dist = (dist_score - normalizer["dist_min"]) / (normalizer["dist_max"] - normalizer["dist_min"] + 1e-8)
    norm_mse = (mse_log - normalizer["mse_log_min"]) / (normalizer["mse_log_max"] - normalizer["mse_log_min"] + 1e-8)
    combined_score = norm_dist * dist_weight + norm_mse * mse_weight
    return float(combined_score), float(norm_dist), float(norm_mse), float(mse_log)


def predict_one_clip(model, clip_np, device, centroids_norm, normalizer, threshold, dist_weight, mse_weight):
    clip_tensor = torch.tensor(clip_np, dtype=torch.float32).unsqueeze(0).to(device)
    valid_mask = get_valid_mask(clip_tensor)
    clip_tensor = normalize_skeleton_batch(clip_tensor, center_joint_idx=CONFIG["center_joint_idx"]).contiguous()
    
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
        mse_score = (torch.sum(masked_diff) / (torch.sum(valid_mask) + 1e-6)).item()
        
        # 3. Cosine Distance
        z_norm = F.normalize(z, p=2, dim=1)
        cos_sim = torch.mm(z_norm, centroids_norm.t())
        max_sim, nearest_class_idx = torch.max(cos_sim, dim=1)
        dist_score = (1.0 - max_sim).item()
        
        nearest_action_id = KNOWN_ACTIONS[nearest_class_idx.item()]
        nearest_action_name = ACTION_NAMES.get(nearest_action_id, "unknown")
        
        # 4. Combined Fusion
        combined_score, norm_dist, norm_mse, mse_log = compute_combined_score(
            mse_score, dist_score, normalizer, dist_weight, mse_weight
        )
        is_unknown = combined_score >= threshold

    return {
        "action_id": pred_action_id, "action_name": pred_action_name, "confidence": confidence,
        "nearest_action_id": nearest_action_id, "nearest_action_name": nearest_action_name,
        "combined_score": combined_score, "is_unknown": is_unknown, "mse_score": mse_score, "dist_score": dist_score
    }


# ============================================================
# 核心重構：即時視訊串流播放與排雷監控一體化化
# ============================================================
def play_and_live_inference(video_path, yolo_model, model, device, centroids_norm, normalizer, threshold, dist_weight, mse_weight):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0: fps = 30

    print("\n============================================================")
    print("🚀 啟動即時序列串流推論監控系統...")
    print(f"影片預估總長度: {total_frames / fps:.2f} 秒 (共 {total_frames} 幀)")
    print("============================================================\n")

    skeleton_buffer = []
    frame_idx = 0
    last_alert_time = -9999
    
    # 儲存最新的雷達探測結果，用來即時渲染在播放畫面上
    current_radar_res = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: 抽取當前單幀的 YOLO 17點骨架
        one_frame_skeleton = np.zeros((2, 17), dtype=np.float32)
        results = yolo_model(frame, verbose=False)

        if len(results) > 0 and results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy
            if keypoints is not None and len(keypoints) > 0:
                person_kpts = keypoints[0].cpu().numpy()
                if person_kpts.shape[0] >= 17:
                    one_frame_skeleton[0, :] = person_kpts[:17, 0]
                    one_frame_skeleton[1, :] = person_kpts[:17, 1]

        # 將當前幀的骨架打入右側的動態滑動視窗快取
        skeleton_buffer.append(one_frame_skeleton)
        current_sec = frame_idx / fps

        # Step 2: 判定是否觸發滑動視窗評估
        # 當累積的幀數滿足一個 window_size 且每隔 stride 幀，立刻進行 ST-CROSR 計算
        if len(skeleton_buffer) >= CONFIG["window_size"] and (frame_idx % CONFIG["stride"] == 0):
            # 取出最近的 N 幀骨架快取
            clip = np.stack(skeleton_buffer[-CONFIG["window_size"]:], axis=0)
            clip = np.transpose(clip, (1, 0, 2))  # [W, 2, 17] -> [2, W, 17]
            clip_padded = pad_or_cut_to_300(clip)

            # 丟進大腦模型做開集辨識
            current_radar_res = predict_one_clip(
                model, clip_padded, device, centroids_norm, normalizer, threshold, dist_weight, mse_weight
            )

            # 【核心功能】如果發現異常，當場在控制台輸出「精準到秒」的紅色強效警告！
            if current_radar_res["is_unknown"]:
                if current_sec - last_alert_time >= CONFIG["alert_cooldown_sec"]:
                    print(f"🚨 【異常爆警!!】影片撥放到 [ {current_sec:6.2f} 秒 ] 🔴 綜合異常分：{current_radar_res['combined_score']:.4f} 超過安全閾值 ({threshold:.4f})！")
                    print(f"    -> 系統判定：此動作為未知類別 (最接近的正常動作範本為: {current_radar_res['nearest_action_name']})")
                    last_alert_time = current_sec
                else:
                    print(f"⚠️ [持續異常偵測中] 影片時間: {current_sec:6.2f} 秒 | 處於冷卻時間內，跳過重複報警。")

        # Step 3: 即時渲染畫面的視覺化 UI
        if CONFIG["show_yolo_window"]:
            display_frame = frame.copy()
            
            # 1. 頂部繪製動態時間軸畫布
            cv2.rectangle(display_frame, (10, 10), (320, 50), (0, 0, 0), -1)
            cv2.putText(display_frame, f"Time: {current_sec:.2f}s / {total_frames/fps:.2f}s", (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # 2. 如果雷達引擎有算好的結果，直接大膽地畫到畫面上
            if current_radar_res is not None:
                if current_radar_res["is_unknown"]:
                    # 如果有異常，在影片最上方橫切出一整條極其顯眼的紅色震撼警報大字！
                    cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 60), (0, 0, 255), -1)
                    cv2.putText(display_frame, f"🚨 ALARM: UNKNOWN ANOMALY DETECTED AT {current_sec:.2f}s! 🚨", (20, 38),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # 畫面下方顯示數值面板
                    cv2.putText(display_frame, f"Score: {current_radar_res['combined_score']:.4f} (Thresh: {threshold:.4f})", (20, 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    # 如果正常，顯示綠色放行字樣與當前的分類預測結果
                    cv2.putText(display_frame, f"STATUS: NORMAL ({current_radar_res['combined_score']:.4f})", (20, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(display_frame, f"ACT: {current_radar_res['action_name']} ({current_radar_res['confidence']*100:.1f}%)", (20, 115),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("ST-CROSR Live Real-Time Radar Monitor", display_frame)
            
            # 按 'q' 鍵可隨時優雅退場
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("🛑 使用者手動中斷串流播放。")
                break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("\n🏁 影片串流即時掃描圓滿結束。")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加載雷達統計配置
    centroids_norm, normalizer, threshold, dist_weight, mse_weight = load_radar_meta_params(device)
    
    # 2. 依據類別數量組裝模型
    model = load_st_crosr_model(device)
    
    # 3. 實體化 YOLO 姿態追踪引擎
    yolo_model = YOLO(CONFIG["yolo_model_path"])

    # 4. 啟動一體化即時推論播放器
    play_and_live_inference(
        CONFIG["video_path"], yolo_model, model, device,
        centroids_norm, normalizer, threshold, dist_weight, mse_weight
    )


if __name__ == "__main__":
    main()