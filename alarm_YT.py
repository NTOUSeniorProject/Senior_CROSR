import sys
import os
import torch
from ultralytics import YOLO
from constants import CONFIG
from model_loader import load_radar_meta_params, load_st_crosr_model
from real_time_detector import play_and_live_inference

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Functions"))


def main(video_path=None, line_user_id=None):
    # 有傳入影片來源，就覆蓋 CONFIG 預設值
    if video_path:
        video_path = video_path.strip()
        CONFIG["video_path"] = video_path

        # RTSP、RTMP、m3u8 都視為直播來源
        CONFIG["is_live_stream"] = (
            video_path.lower().startswith(("rtsp://", "rtmp://"))
            or ".m3u8" in video_path.lower()
        )

        if CONFIG["is_live_stream"]:
            print("✅ 已判斷為即時串流來源")
        else:
            print("✅ 已判斷為一般影片來源")

    if line_user_id:
        print(f"✅ 已接收 LINE user_id：{line_user_id}")
    else:
        print("⚠️ 未接收 LINE user_id，本次只會在終端機顯示警報，不會推播到 LINE。")

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
        mse_weight,
        line_user_id=line_user_id
    )


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    line_user_id = sys.argv[2] if len(sys.argv) > 2 else None

    main(video_path, line_user_id)
