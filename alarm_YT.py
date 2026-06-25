import os
import re
import cv2
import torch
import shutil
import subprocess
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO

from ST_CROSR import ST_CROSR
from ntu_normalize import normalize_skeleton_batch, get_valid_mask


# ============================================================
# 設定區
# ============================================================
CONFIG = {
    # ========================================================
    # 影片來源設定
    # 可以放：
    # 1. 本機影片路徑
    # 2. YouTube 一般影片連結
    # 3. YouTube 直播連結
    # 4. m3u8 / http / rtsp 串流
    # ========================================================

    # 本機影片範例：
    # "video_path": r"C:\CROSR\IMG_2033.mov",

    # YouTube 影片 / 直播範例：
    "video_path": r"https://youtu.be/kD0RBvXA1q4?si=ZJnV3lV45Yifloay",

    "yolo_model_path": r"C:\CROSR\yolo26x-pose.pt",
    "checkpoint_path": r"checkpoints_20260602_2237\best_val.pth",
    "radar_meta_path": r"C:\CROSR\radar_meta_params.pth",

    "max_frames": 300,
    "num_nodes": 17,
    "center_joint_idx": 11,

    "window_size": 120,
    "stride": 30,

    # ── 閾值設定 ──────────────────────────────────────────────
    # True  → 使用 radar_meta_params.pth 裡面儲存的閾值
    # False → 使用下方 manual_threshold
    "use_saved_threshold": False,
    "manual_threshold": 0.4,

    # ── 連續異常報警設定 ──────────────────────────────────────
    # True  → 需連續偵測異常達 consecutive_alert_sec 秒才觸發警報
    # False → 單次超過閾值就立即報警
    "use_consecutive_alert": True,
    "consecutive_alert_sec": 4,

    # 報警後冷卻時間（秒），冷卻期間不重複警報
    "alert_cooldown_sec": 4,

    "show_yolo_window": True,

    # ========================================================
    # YouTube / 串流設定
    # ========================================================
    # True  → 如果 video_path 是 YouTube 連結，使用 yt-dlp 解析串流網址
    # False → 完全照原本方式丟給 OpenCV
    "enable_youtube_url": True,

    # YouTube 解析格式
    # 直播通常會拿到 m3u8
    # 一般影片會盡量拿 OpenCV 比較容易讀的 mp4 / hls
    "youtube_format": "best[protocol^=m3u8][height<=720]/best[protocol^=m3u8]/best[height<=720]/best",

    # 如果 YouTube 需要登入 / 年齡驗證 / 私人影片，可以改成 True
    # 會嘗試讀取瀏覽器 cookies
    "use_browser_cookies": False,

    # 瀏覽器名稱，可用 chrome / edge / firefox
    "cookies_browser": "chrome",

    # OpenCV 讀串流時是否指定 FFMPEG backend
    "use_ffmpeg_backend": True,
    "is_live_stream": False,
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


# ============================================================
# 新增：影片來源解析
# ============================================================
def is_url(source):
    if not isinstance(source, str):
        return False
    return source.startswith(("http://", "https://", "rtsp://", "rtmp://"))


def is_youtube_url(source):
    if not isinstance(source, str):
        return False

    youtube_patterns = [
        r"youtube\.com",
        r"youtu\.be",
        r"youtube\.com/live",
        r"youtube\.com/watch",
        r"youtube\.com/shorts",
    ]

    return any(re.search(pattern, source, re.IGNORECASE) for pattern in youtube_patterns)


def is_direct_stream_url(source):
    if not isinstance(source, str):
        return False

    lower_source = source.lower()

    direct_keywords = [
        ".m3u8",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".flv",
        ".webm",
        "rtsp://",
        "rtmp://",
    ]

    return any(keyword in lower_source for keyword in direct_keywords)


def check_ytdlp_installed():
    ytdlp_path = shutil.which("yt-dlp")
    if ytdlp_path is None:
        raise RuntimeError(
            "找不到 yt-dlp。\n"
            "請先在你的環境安裝：\n"
            "pip install yt-dlp\n"
            "或使用：\n"
            "python -m pip install yt-dlp"
        )
    return ytdlp_path


def resolve_youtube_stream_url(youtube_url):
    """
    使用 yt-dlp 把 YouTube 連結解析成真正可讀的串流 URL。

    注意：
    - 一般影片可能會回傳 mp4 / webm 直連
    - 直播通常會回傳 m3u8
    - 如果 yt-dlp 回傳多行，優先選 m3u8，其次選第一個 URL
    """

    check_ytdlp_installed()

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f", CONFIG["youtube_format"],
        "-g",
        youtube_url
    ]

    if CONFIG["use_browser_cookies"]:
        cmd.extend([
            "--cookies-from-browser",
            CONFIG["cookies_browser"]
        ])

    print("\n============================================================")
    print("🔗 偵測到 YouTube 連結，正在解析串流網址...")
    print(f"YouTube URL: {youtube_url}")
    print("============================================================")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="ignore"
        )

        urls = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip().startswith(("http://", "https://"))
        ]

        if len(urls) == 0:
            raise RuntimeError(
                "yt-dlp 沒有輸出可用串流網址。\n"
                f"stderr:\n{result.stderr}"
            )

        # 優先選 m3u8，直播比較常見
        for url in urls:
            if ".m3u8" in url.lower():
                print("✅ 已取得 YouTube HLS / m3u8 串流網址")
                return url

        # 沒有 m3u8 就取第一個
        print("✅ 已取得 YouTube 直連影片網址")
        return urls[0]

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        raise RuntimeError(
            "yt-dlp 解析 YouTube 失敗。\n"
            "可能原因：\n"
            "1. yt-dlp 版本太舊\n"
            "2. YouTube 影片需要登入 / cookies\n"
            "3. 影片是私人影片或地區限制\n"
            "4. 網路連線問題\n\n"
            "你可以先嘗試更新：\n"
            "pip install -U yt-dlp\n\n"
            "如果需要 cookies，請把 CONFIG['use_browser_cookies'] 改成 True。\n\n"
            f"錯誤內容：\n{error_msg}"
        )


def resolve_video_source(source):
    """
    將輸入來源統一轉成 OpenCV 可以讀的來源。

    支援：
    1. 本機檔案
    2. YouTube 連結
    3. m3u8 / rtsp / rtmp / http 影片串流
    """

    if source is None or str(source).strip() == "":
        raise ValueError("CONFIG['video_path'] 是空的，請填入本機影片路徑或 YouTube 連結。")

    source = str(source).strip()

    # 1. 本機檔案存在：直接回傳
    if os.path.exists(source):
        print(f"✅ 使用本機影片來源：{source}")
        return source

    # 2. YouTube URL：用 yt-dlp 解析
    if CONFIG["enable_youtube_url"] and is_youtube_url(source):
        return resolve_youtube_stream_url(source)

    # 3. 直接串流 URL：直接回傳
    if is_url(source) and is_direct_stream_url(source):
        print(f"✅ 使用直接串流來源：{source}")
        return source

    # 4. 其他 URL：仍然嘗試丟給 OpenCV
    if is_url(source):
        print("⚠️ 偵測到一般 URL，將直接交給 OpenCV 嘗試開啟。")
        return source

    # 5. 不是 URL，也不是存在的本機檔案
    raise FileNotFoundError(
        f"找不到影片來源：{source}\n"
        "請確認：\n"
        "1. 本機影片路徑是否正確\n"
        "2. YouTube 連結是否完整\n"
        "3. 如果是 Windows 路徑，請使用 r\"C:\\路徑\\影片.mp4\""
    )


def open_video_capture(source):
    """
    統一建立 OpenCV VideoCapture。
    """

    resolved_source = resolve_video_source(source)

    if CONFIG["use_ffmpeg_backend"]:
        cap = cv2.VideoCapture(resolved_source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(resolved_source)

    if not cap.isOpened():
        raise FileNotFoundError(
            f"OpenCV 無法開啟影片來源。\n"
            f"原始來源：{source}\n"
            f"解析後來源：{resolved_source}\n\n"
            "可能原因：\n"
            "1. OpenCV 沒有 FFmpeg 支援\n"
            "2. YouTube 串流網址過期\n"
            "3. 網路不穩\n"
            "4. 影片格式 OpenCV 不支援\n"
            "5. 直播目前沒有開播"
        )

    return cap, resolved_source


# ============================================================
# 模型與雷達參數
# ============================================================
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


# ============================================================
# 即時視訊串流播放與排雷監控
# ============================================================
def play_and_live_inference(
    video_path,
    yolo_model,
    model,
    device,
    centroids_norm,
    normalizer,
    threshold,
    dist_weight,
    mse_weight
):
    cap, resolved_source = open_video_capture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or np.isnan(fps):
        fps = 30

    is_live_like_source = (
        ".m3u8" in resolved_source.lower()
        or "rtsp://" in resolved_source.lower()
        or "rtmp://" in resolved_source.lower()
        or total_frames <= 0
    )

    print("\n============================================================")
    print("🚀 啟動即時序列串流推論監控系統...")
    print(f"原始影片來源: {video_path}")

    if resolved_source != video_path:
        print("影片來源已解析成 OpenCV 可讀串流。")

    if is_live_like_source:
        print("影片模式: 串流 / 直播 / 無固定總長度")
        print(f"FPS: {fps:.2f}")
    else:
        print(f"影片預估總長度: {total_frames / fps:.2f} 秒 (共 {total_frames} 幀)")
        print(f"FPS: {fps:.2f}")

    threshold_label = (
        "使用儲存閾值"
        if CONFIG["use_saved_threshold"]
        else f"手動閾值 = {threshold:.4f}"
    )

    print(f"閾值模式: {threshold_label}")

    if CONFIG["use_consecutive_alert"]:
        print(f"報警模式: 連續 {CONFIG['consecutive_alert_sec']} 秒才報警")
    else:
        print("報警模式: 單次即報警")

    print("============================================================\n")

    skeleton_buffer = []
    frame_idx = 0
    last_alert_time = -9999
    current_radar_res = None

    consecutive_anomaly_start = None

    while True:
        ret, frame = cap.read()

        if not ret:
            if CONFIG["is_live_stream"]:
                print("⚠️ 直播串流暫時中斷，嘗試重新連線...")

                cap.release()

                try:
                    cap, resolved_source = open_video_capture(video_path)
                    print("✅ 重新連線成功，繼續推論。")
                    continue
                except Exception as e:
                    print("❌ 重新連線失敗，停止推論。")
                    print(e)
                    break
            else:
                print("🏁 影片已播放完畢，結束推論。")
                break

        # Step 1: 抽取當前單幀骨架
        one_frame_skeleton = np.zeros((2, 17), dtype=np.float32)

        results = yolo_model(frame, verbose=False)

        if len(results) > 0 and results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy

            if keypoints is not None and len(keypoints) > 0:
                person_kpts = keypoints[0].cpu().numpy()

                if person_kpts.shape[0] >= 17:
                    one_frame_skeleton[0, :] = person_kpts[:17, 0]
                    one_frame_skeleton[1, :] = person_kpts[:17, 1]

        skeleton_buffer.append(one_frame_skeleton)
        current_sec = frame_idx / fps

        # Step 2: 滑動視窗觸發評估
        if (
            len(skeleton_buffer) >= CONFIG["window_size"]
            and frame_idx % CONFIG["stride"] == 0
        ):
            clip = np.stack(
                skeleton_buffer[-CONFIG["window_size"]:], axis=0
            )

            clip = np.transpose(clip, (1, 0, 2))
            clip_padded = pad_or_cut_to_300(clip)

            current_radar_res = predict_one_clip(
                model,
                clip_padded,
                device,
                centroids_norm,
                normalizer,
                threshold,
                dist_weight,
                mse_weight
            )

            # Step 3: 報警判斷
            if current_radar_res["is_unknown"]:
                if CONFIG["use_consecutive_alert"]:
                    if consecutive_anomaly_start is None:
                        consecutive_anomaly_start = current_sec

                    consecutive_duration = current_sec - consecutive_anomaly_start

                    if consecutive_duration >= CONFIG["consecutive_alert_sec"]:
                        if current_sec - last_alert_time >= CONFIG["alert_cooldown_sec"]:
                            print(
                                f"🚨 【異常爆警!!】影片播放到 [ {current_sec:6.2f} 秒 ] 🔴 "
                                f"連續異常 {consecutive_duration:.1f} 秒，"
                                f"綜合異常分：{current_radar_res['combined_score']:.4f} "
                                f"超過閾值 ({threshold:.4f})！"
                            )

                            print(
                                f"    -> 最接近正常動作："
                                f"{current_radar_res['nearest_action_name']}"
                            )

                            last_alert_time = current_sec
                        else:
                            print(
                                f"⚠️ [持續異常偵測中] {current_sec:6.2f} 秒 | "
                                f"冷卻中，跳過重複報警。"
                            )
                    else:
                        print(
                            f"⏳ [異常累積中] {current_sec:6.2f} 秒 | "
                            f"已持續 {consecutive_duration:.1f}/"
                            f"{CONFIG['consecutive_alert_sec']} 秒 "
                            f"| 分數：{current_radar_res['combined_score']:.4f}"
                        )
                else:
                    if current_sec - last_alert_time >= CONFIG["alert_cooldown_sec"]:
                        print(
                            f"🚨 【異常爆警!!】影片播放到 [ {current_sec:6.2f} 秒 ] 🔴 "
                            f"綜合異常分：{current_radar_res['combined_score']:.4f} "
                            f"超過閾值 ({threshold:.4f})！"
                        )

                        print(
                            f"    -> 最接近正常動作："
                            f"{current_radar_res['nearest_action_name']}"
                        )

                        last_alert_time = current_sec
                    else:
                        print(
                            f"⚠️ [持續異常偵測中] {current_sec:6.2f} 秒 | "
                            f"冷卻中，跳過重複報警。"
                        )
            else:
                if consecutive_anomaly_start is not None:
                    print(
                        f"✅ [{current_sec:6.2f} 秒] "
                        f"異常解除，連續異常中斷。"
                    )
                    consecutive_anomaly_start = None

        # Step 4: 即時渲染 UI
        if CONFIG["show_yolo_window"]:
            display_frame = frame.copy()

            cv2.rectangle(
                display_frame,
                (10, 10),
                (360, 50),
                (0, 0, 0),
                -1
            )

            if is_live_like_source:
                time_text = f"Time: {current_sec:.2f}s / LIVE"
            else:
                time_text = f"Time: {current_sec:.2f}s / {total_frames / fps:.2f}s"

            cv2.putText(
                display_frame,
                time_text,
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            if current_radar_res is not None:
                is_alerting = (
                    current_radar_res["is_unknown"]
                    and (
                        not CONFIG["use_consecutive_alert"]
                        or (
                            consecutive_anomaly_start is not None
                            and current_sec - consecutive_anomaly_start >= CONFIG["consecutive_alert_sec"]
                        )
                    )
                )

                if is_alerting:
                    cv2.rectangle(
                        display_frame,
                        (0, 0),
                        (display_frame.shape[1], 60),
                        (0, 0, 255),
                        -1
                    )

                    cv2.putText(
                        display_frame,
                        f"ALARM: UNKNOWN ANOMALY AT {current_sec:.2f}s",
                        (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                    cv2.putText(
                        display_frame,
                        f"Score: {current_radar_res['combined_score']:.4f} "
                        f"(Thresh: {threshold:.4f})",
                        (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                    )

                    cv2.putText(
                        display_frame,
                        f"Nearest: {current_radar_res['nearest_action_name']}",
                        (20, 125),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                    )

                elif current_radar_res["is_unknown"] and CONFIG["use_consecutive_alert"]:
                    elapsed = (
                        current_sec - consecutive_anomaly_start
                        if consecutive_anomaly_start is not None
                        else 0
                    )

                    cv2.putText(
                        display_frame,
                        f"WARNING: Accumulating {elapsed:.1f}/"
                        f"{CONFIG['consecutive_alert_sec']}s",
                        (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA
                    )

                    cv2.putText(
                        display_frame,
                        f"Score: {current_radar_res['combined_score']:.4f}",
                        (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA
                    )

                else:
                    cv2.putText(
                        display_frame,
                        f"STATUS: NORMAL ({current_radar_res['combined_score']:.4f})",
                        (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                    cv2.putText(
                        display_frame,
                        f"ACT: {current_radar_res['action_name']} "
                        f"({current_radar_res['confidence'] * 100:.1f}%)",
                        (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

            cv2.imshow(
                "ST-CROSR Live Real-Time Radar Monitor",
                display_frame
            )

            # delay = max(1, int(1000 / fps)) #等33毫秒
            delay = max(1, int(1000 / fps * 0.5))
            
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                print("🛑 使用者手動中斷串流播放。")
                break
            
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\n🏁 影片串流即時掃描結束。")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("============================================================")
    print(f"🖥️ 使用裝置：{device}")
    print("============================================================")

    # 1. 加載雷達統計配置
    centroids_norm, normalizer, threshold, dist_weight, mse_weight = load_radar_meta_params(device)

    # 2. 依據類別數量組裝模型
    model = load_st_crosr_model(device)

    # 3. 實體化 YOLO 姿態追踪引擎
    yolo_model = YOLO(CONFIG["yolo_model_path"])

    # 4. 啟動一體化即時推論播放器
    play_and_live_inference(
        CONFIG["video_path"],
        yolo_model,
        model,
        device,
        centroids_norm,
        normalizer,
        threshold,
        dist_weight,
        mse_weight
    )


if __name__ == "__main__":
    main()