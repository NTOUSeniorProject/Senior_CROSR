"""使用原 constants 設定和模型檔，離線逐幀執行既有異常偵測。"""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('video')
    p.add_argument('--yolo', required=True, help='你的 YOLO pose 權重檔')
    p.add_argument('--log', required=True)
    p.add_argument('--timeout', type=int, default=600, help='每次 backend 請求的讀取逾時秒數')
    p.add_argument('--device', default=None, help='例如 cuda:0 或 cpu')
    p.add_argument('--show', action='store_true')
    p.add_argument('--wall-clock', action='store_true', help='改用原本牆鐘的啟動/週期排程')
    args = p.parse_args()
    if args.timeout <= 0:
        p.error('timeout 必須大於 0')
    if not Path(args.video).is_file():
        p.error('請使用本機影片')
    import torch
    from ultralytics import YOLO
    from constants import CONFIG
    from inference.model_loader import load_radar_meta_params, load_st_crosr_model
    from inference.real_time_detector import play_and_live_inference
    CONFIG['show_yolo_window'] = args.show
    CONFIG['is_live_stream'] = False
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    centroids, normalizer, threshold, dist_weight, mse_weight = load_radar_meta_params(device)
    model = load_st_crosr_model(device)
    yolo_model = YOLO(args.yolo)
    play_and_live_inference(str(Path(args.video).resolve()), yolo_model, model,
        device, centroids, normalizer, threshold, dist_weight, mse_weight,
        line_user_id=None, token_log_path=args.log,
        experiment_media_clock=not args.wall_clock, vlm_timeout=args.timeout)


if __name__ == '__main__':
    main()
