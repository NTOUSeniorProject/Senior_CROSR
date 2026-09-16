"""VLM 實驗請求入口。用法見 SYSTEM_VS_VLM.md。"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
import uuid

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)


_LOCK = threading.Lock()
_RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]


def _seconds(value):
    return value / 1_000_000_000 if isinstance(value, (int, float)) else None


def _metrics(data):
    """只記錄供應端回報的 token；缺值不當成零。"""
    ollama = "message" in data and "choices" not in data
    usage = data.get("usage") or {}
    prompt = data.get("prompt_eval_count") if ollama else usage.get("prompt_tokens")
    completion = data.get("eval_count") if ollama else usage.get("completion_tokens")
    total = usage.get("total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    details = usage.get("completion_tokens_details") or {}
    return {
        "backend": "ollama" if ollama else "openai-compatible",
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "reasoning_tokens": details.get("reasoning_tokens"),
        "thinking_seconds": None,  # 非串流回應無法量測獨立思考階段。
        "prompt_eval_seconds": _seconds(data.get("prompt_eval_duration")),
        "generation_seconds": _seconds(data.get("eval_duration")),
        "server_total_seconds": _seconds(data.get("total_duration")),
        "load_seconds": _seconds(data.get("load_duration")),
    }


def request_vlm(*args, experiment_layer=None, **kwargs):
    """實際呼叫 VLM 並逐次追加 JSONL；回傳原始 Response 供原流程解析。"""
    payload = kwargs.get("json") or {}
    path = Path(os.getenv("VLM_EXPERIMENT_LOG", "output/system_vs_vlm/requests.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": os.getenv("VLM_EXPERIMENT_RUN_ID", _RUN_ID),
        "request_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "arm": os.getenv("VLM_EXPERIMENT_ARM", "system"),
        "video": _video_label(),
        "layer": experiment_layer,
        "model": payload.get("model"),
        "status_code": None,
        **_metrics({}),
    }
    record["backend"] = "ollama" if "options" in payload else "openai-compatible"
    started = time.perf_counter()
    try:
        response = requests.post(*args, **kwargs)
        record["status_code"] = response.status_code
        try:
            data = response.json()
            if isinstance(data, dict):
                record.update(_metrics(data))
        except ValueError:
            record["response_json_error"] = True
        return response
    except Exception as error:
        record["error_type"] = type(error).__name__
        raise
    finally:
        record["request_seconds"] = time.perf_counter() - started
        # 不保存影像、提示詞、API key 或完整回應。
        with _LOCK, path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _video_label():
    explicit = os.getenv("VLM_EXPERIMENT_VIDEO")
    if explicit:
        return explicit
    from constants import CONFIG
    source = str(CONFIG["video_path"])
    return source if "://" in source else str(Path(source).resolve())


def summarize(path):
    groups = {}
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            key = (row["run_id"], row["video"], row["arm"])
            groups.setdefault(key, []).append(row)
    result = []
    for (run_id, video, arm), rows in groups.items():
        item = {"run_id": run_id, "video": video, "arm": arm,
                "requests": len(rows),
                "failed_requests": sum(r.get("status_code") is None or r["status_code"] >= 400 for r in rows)}
        for field in ("prompt_tokens", "completion_tokens", "total_tokens", "request_seconds",
                      "generation_seconds", "thinking_seconds"):
            values = [r[field] for r in rows if r.get(field) is not None]
            item[field + "_known_sum"] = sum(values) if values else None
            item[field + "_missing_requests"] = len(rows) - len(values)
        result.append(item)
    return result


def run_video(video, clip_seconds, frames_per_clip, double_vlm, vlm_78b_layer):
    """純 VLM 基準：逐段讀取完整影片，每段均勻抽樣，包含最後不足一段。"""
    import cv2
    import VLM_check
    from video_source import normalize_video_source

    video = normalize_video_source(video)
    if not Path(video).is_file():
        raise FileNotFoundError(f"請指定本機影片：{video}")
    video = str(Path(video).resolve())
    print(f"[vlm_only] 本次分析影片：{video}", flush=True)
    cap = cv2.VideoCapture(video)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
            raise ValueError("無法讀取影片或 FPS 無效")
        clip_size = max(1, round(fps * clip_seconds))
        VLM_check.EXPERIMENT_MODE = True
        os.environ["VLM_EXPERIMENT_VIDEO"] = str(Path(video).resolve())
        os.environ["VLM_EXPERIMENT_ARM"] = "vlm_only"
        with tempfile.TemporaryDirectory(prefix="vlm_frames_") as directory:
            while True:
                # 先將一段 JPEG 暫存到磁碟，避免長片段占用大量 RAM。
                count = 0
                for index in range(clip_size):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if not cv2.imwrite(str(Path(directory) / f"{index}.jpg"), frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 85]):
                        raise RuntimeError("無法保存抽樣影格")
                    count += 1
                if not count:
                    break
                n = min(count, frames_per_clip)
                indices = [round(i * (count - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
                paths = [str(Path(directory) / f"{i}.jpg") for i in indices]
                result = VLM_check.analyze_frames_with_ollama(
                    paths, double_vlm=double_vlm, vlm_78b_layer=vlm_78b_layer)
                print(json.dumps(result, ensure_ascii=False))
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    summary = sub.add_parser("summary", help="依 run、影片、比較組別彙總")
    summary.add_argument("--log", default=os.getenv("VLM_EXPERIMENT_LOG", "output/system_vs_vlm/requests.jsonl"))
    video = sub.add_parser("video", help="執行完整本機影片的純 VLM 基準")
    video.add_argument("video", nargs="?", help="省略時使用 constants.CONFIG['video_path']，限本機影片")
    video.add_argument("--clip-seconds", type=float, default=10)
    video.add_argument("--frames-per-clip", type=int, default=8)
    video.add_argument("--double-vlm", type=int, choices=[0, 1], default=0)
    video.add_argument("--vlm-78b-layer", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    if args.command == "summary":
        print(json.dumps(summarize(args.log), ensure_ascii=False, indent=2))
    else:
        if not math.isfinite(args.clip_seconds) or args.clip_seconds <= 0 or args.frames_per_clip <= 0:
            parser.error("片段秒數與抽樣張數必須大於零")
        source = args.video
        if source is None:
            from constants import CONFIG
            source = CONFIG["video_path"]
        run_video(source, args.clip_seconds, args.frames_per_clip, args.double_vlm, args.vlm_78b_layer)


if __name__ == "__main__":
    main()
