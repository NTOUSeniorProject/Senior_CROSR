"""在原專案根目錄執行；不載入 YOLO / CROSR。"""
import argparse
import json
import math
import tempfile
from pathlib import Path
import cv2
from constants import PRE_EVENT_SECONDS, POST_EVENT_SECONDS, VLM_SAMPLE_FRAME_COUNT
from inference.event_handler import sample_vlm_frames
from inference.VLM_check import analyze_frames_with_ollama
from inference.token_stats import begin_run, finish_run, record_error


def main():
    p = argparse.ArgumentParser()
    p.add_argument('video')
    p.add_argument('--log', required=True)
    p.add_argument('--timeout', type=int, default=600, help='每次 backend 請求的讀取逾時秒數')
    p.add_argument('--window-sec', type=float, default=PRE_EVENT_SECONDS + POST_EVENT_SECONDS)
    args = p.parse_args()
    if not math.isfinite(args.window_sec) or args.window_sec <= 0:
        p.error('window-sec 必須是大於 0 的有限值')
    if args.timeout <= 0:
        p.error('timeout 必須大於 0')
    video = Path(args.video).resolve()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError('無法開啟影片或取得有效 FPS')
    try:
        begin_run(args.log, video, 'baseline', vlm_timeout=args.timeout)
    except Exception:
        cap.release()
        raise
    window_size = max(1, round(args.window_sec * fps))
    index = 0
    eof = False
    completed = False
    try:
        with tempfile.TemporaryDirectory() as folder:
            while not eof:
                frames = []
                for _ in range(window_size):
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        eof = True
                        break
                    frames.append({'time': index / fps, 'frame': frame})
                    index += 1
                if not frames:
                    break
                paths = sample_vlm_frames(frames, folder, VLM_SAMPLE_FRAME_COUNT)
                print(f"[baseline] {frames[0]['time']:.2f}–{index / fps:.2f} 秒 | "
                      f"{len(paths)} 張圖片 | VLM 讀取逾時 {args.timeout} 秒", flush=True)
                result = analyze_frames_with_ollama(paths, timeout=args.timeout)
                print(json.dumps(result, ensure_ascii=False))
            completed = eof and (expected_frames <= 0 or index >= expected_frames)
    except BaseException as exc:
        record_error('baseline_processing', str(exc))
        raise
    finally:
        cap.release()
        print(json.dumps(finish_run(completed, processed_frames=index, fps=fps,
                                    window_sec=args.window_sec), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
