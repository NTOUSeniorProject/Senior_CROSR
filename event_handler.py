import os
import cv2
import numpy as np
from constants import (
    PRE_EVENT_SECONDS,
    POST_EVENT_SECONDS,
    POST_MIN_ANOMALY_VOTES,
    POST_ANOMALY_RATIO_THRESHOLD,
    VLM_SAMPLE_FRAME_COUNT,
    ANOMALY_OUTPUT_ROOT,
)


def start_event_collection(
    pre_event_buffer,
    current_sec,
    anomaly_event_start,
    pre_event_seconds=PRE_EVENT_SECONDS,
    post_event_seconds=POST_EVENT_SECONDS,
):
    """複製正式確認前 5 秒影格，並設定後 5 秒截止時間"""

    event_start_sec = max(
        0.0,
        current_sec - pre_event_seconds,
    )

    event_collect_until_sec = (
        current_sec + post_event_seconds
    )

    event_frames = [
        {
            "time": item["time"],
            "frame": item["frame"].copy(),
        }
        for item in pre_event_buffer
        if item["time"] >= event_start_sec
    ]

    return (
        event_frames,
        event_start_sec,
        event_collect_until_sec,
    )


def should_keep_event(
    anomaly_flags,
    min_votes=POST_MIN_ANOMALY_VOTES,
    ratio_threshold=POST_ANOMALY_RATIO_THRESHOLD,
):
    """依異常確認後的滑動視窗結果，決定是否保留事件"""

    total_count = len(anomaly_flags)
    anomaly_count = int(sum(anomaly_flags))

    anomaly_ratio = (
        anomaly_count / total_count
        if total_count > 0
        else 0.0
    )

    keep = (
        anomaly_count >= min_votes
        and anomaly_ratio >= ratio_threshold
    )

    return (
        keep,
        anomaly_count,
        total_count,
        anomaly_ratio,
    )


def save_anomaly_event_frames(
    event_id,
    event_frames,
    fps,
    output_root=ANOMALY_OUTPUT_ROOT,
    vlm_frame_count=VLM_SAMPLE_FRAME_COUNT,
):
    """保存完整事件影片，並平均抽取數張圖片供 VLM 分析"""
    if not event_frames:
        raise ValueError("異常事件沒有可保存的影格。")

    event_dir = os.path.join(output_root, event_id)
    sampled_dir = os.path.join(event_dir, "vlm_frames")
    os.makedirs(sampled_dir, exist_ok=True)

    first_frame = event_frames[0]["frame"]
    height, width = first_frame.shape[:2]
    video_path = os.path.join(event_dir, "event.mp4")

    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"無法建立異常影片：{video_path}")

    try:
        for item in event_frames:
            output_frame = item["frame"]
            if output_frame.shape[:2] != (height, width):
                output_frame = cv2.resize(output_frame, (width, height))
            writer.write(output_frame)
    finally:
        writer.release()

    sample_count = min(vlm_frame_count, len(event_frames))
    if sample_count <= 1:
        sampled_indices = [0]
    else:
        sampled_indices = np.linspace(
            0,
            len(event_frames) - 1,
            sample_count,
            dtype=int,
        ).tolist()

    sampled_paths = []
    for order, frame_index in enumerate(sampled_indices):
        item = event_frames[frame_index]
        image_path = os.path.join(
            sampled_dir,
            f"frame_{order:02d}_{item['time']:.2f}s.jpg",
        )
        if cv2.imwrite(
            image_path,
            item["frame"],
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        ):
            sampled_paths.append(image_path)

    print(f"🎬 異常影片已保存：{video_path}")
    print(f"🖼️ 已抽取 {len(sampled_paths)} 張 VLM 影格：{sampled_dir}")

    return {
        "event_id": event_id,
        "video_path": video_path,
        "frame_paths": sampled_paths,
        "start_time": event_frames[0]["time"],
        "end_time": event_frames[-1]["time"],
        "total_frames": len(event_frames),
    }


def finish_event_collection(
    event_id,
    event_frames,
    anomaly_flags,
    fps,
    force_partial=False,
):
    """
    完整收集後 5 秒時：
    至少 2 個異常視窗，且異常比例達 30%。

    影片提早結束時：
    至少 1 個異常視窗，且異常比例達 50%。
    """

    if force_partial:
        min_votes = 1
        ratio_threshold = 0.5
    else:
        min_votes = POST_MIN_ANOMALY_VOTES
        ratio_threshold = POST_ANOMALY_RATIO_THRESHOLD

    (
        keep,
        anomaly_count,
        total_count,
        anomaly_ratio,
    ) = should_keep_event(
        anomaly_flags=anomaly_flags,
        min_votes=min_votes,
        ratio_threshold=ratio_threshold,
    )

    if not keep:
        print(
            f"🗑️ 丟棄事件 {event_id} | "
            f"後段異常 {anomaly_count}/{total_count} | "
            f"比例 {anomaly_ratio:.0%} | "
            f"門檻：至少 {min_votes} 票、"
            f"比例至少 {ratio_threshold:.0%}"
        )
        return None

    print(
        f"✅ 保留事件 {event_id} | "
        f"後段異常 {anomaly_count}/{total_count} | "
        f"比例 {anomaly_ratio:.0%}"
    )

    return save_anomaly_event_frames(
        event_id=event_id,
        event_frames=event_frames,
        fps=fps,
    )
