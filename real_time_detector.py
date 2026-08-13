import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from constants import CONFIG, PRE_EVENT_SECONDS, POST_EVENT_SECONDS
from video_source import open_video_capture
from inference import predict_one_clip, pad_or_cut_to_300
from event_handler import start_event_collection, finish_event_collection
from line_notifier import push_line_message
from VLM_check import analyze_frames_with_ollama
from movement_detection import MovementDetector


def _analyze_event_with_vlm(frame_paths, line_user_id):
    """在背景執行 VLM 判讀，避免阻塞即時影片分析。"""
    print(f"🤖 將 {len(frame_paths)} 張影格交給 Ollama VLM")
    vlm_result = analyze_frames_with_ollama(frame_paths)
    print("🧠 Ollama VLM 分析結果：", vlm_result)

    should_alert = (
        vlm_result["is_abnormal"]
        and vlm_result["need_alert"]
        and vlm_result["confidence"] >= 0.75
    )

    if not should_alert:
        print("✅ VLM 判斷未達警報門檻，不發送 LINE。")
        return

    alert_text = (
        "🚨 VLM 確認異常事件\n"
        f"類型：{vlm_result['category']}\n"
        f"信心度：{vlm_result['confidence']:.0%}\n"
        f"描述：{vlm_result['description']}"
    )
    push_line_message(line_user_id, alert_text)


def _report_background_vlm_result(future):
    """集中回報背景 VLM 工作中未處理的例外。"""
    try:
        future.result()
    except Exception as exc:
        print(f"❌ 背景 VLM 分析失敗：{exc}")


def play_and_live_inference(
    video_path,
    yolo_model,
    model,
    device,
    centroids_norm,
    normalizer,
    threshold,
    dist_weight,
    mse_weight,
    line_user_id=None
):
    """
    改良版異常偵測邏輯（不加入骨架品質判斷）：

    1. 每次滑動視窗都送入模型判斷。
    2. 使用最近一段時間內的 unknown 比例，而不是單次 unknown 就報警。
    3. 異常候選需持續一段時間才正式報警。
    4. 異常成立後，需連續多個正常視窗才解除。
    5. 保留通知冷卻，避免持續異常時重複洗版。
    """

    cap, resolved_source = open_video_capture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    pre_event_buffer = deque(
        maxlen=max(1, int(np.ceil(fps * PRE_EVENT_SECONDS)))
    )
    collecting_event = False
    event_id = None
    event_start_sec = None
    event_collect_until_sec = None
    event_frames = []
    post_event_anomaly_flags = []

    is_live_like_source = (
        ".m3u8" in resolved_source.lower()
        or "rtsp://" in resolved_source.lower()
        or "rtmp://" in resolved_source.lower()
        or total_frames <= 0
    )

    window_size = int(CONFIG.get("window_size", 60))
    stride = max(1, int(CONFIG.get("stride", 5)))

    confirm_window_sec = float(
        CONFIG.get("anomaly_confirm_window_sec", 3.0)
    )
    anomaly_vote_ratio = float(
        CONFIG.get("anomaly_vote_ratio", 0.70)
    )
    min_anomaly_votes = int(
        CONFIG.get("min_anomaly_votes", 3)
    )

    consecutive_alert_sec = float(
        CONFIG.get("consecutive_alert_sec", 2.0)
    )

    clear_normal_windows = int(
        CONFIG.get("clear_normal_windows", 3)
    )

    alert_cooldown_sec = float(
        CONFIG.get("alert_cooldown_sec", 30.0)
    )

    evaluation_interval_sec = stride / fps
    vote_history_size = max(
        min_anomaly_votes,
        int(
            np.ceil(
                confirm_window_sec
                / max(evaluation_interval_sec, 1e-6)
            )
        )
    )

    print("\n============================================================")
    print("🚀 啟動改良版即時異常動作監控系統...")
    print(f"原始影片來源: {video_path}")

    if resolved_source != video_path:
        print("影片來源已解析成 OpenCV 可讀串流。")

    if is_live_like_source:
        print("影片模式: 串流 / 直播 / 無固定總長度")
    else:
        print(
            f"影片預估總長度: "
            f"{total_frames / fps:.2f} 秒 ({total_frames} 幀)"
        )

    print(f"FPS: {fps:.2f}")
    print(f"骨架視窗: {window_size} 幀，步長: {stride} 幀")
    print(
        f"異常投票: 最近約 {confirm_window_sec:.1f} 秒內，"
        f"unknown 比例至少 {anomaly_vote_ratio:.0%}"
    )
    print(f"候選異常維持: {consecutive_alert_sec:.1f} 秒")
    print(f"解除條件: 連續 {clear_normal_windows} 個正常視窗")
    print(f"通知冷卻: {alert_cooldown_sec:.1f} 秒")
    print("============================================================\n")

    skeleton_buffer = []
    anomaly_vote_history = []

    frame_idx = 0
    last_alert_time = -9999.0
    current_radar_res = None
    final_video_sec = 0.0
    stopped_by_user = False
    vlm_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="vlm-analysis",
    )

    motion_grace_frames = max(1, int(fps * 2.0))
    motion_hold_remaining = 0

    detection_state = "normal"
    anomaly_candidate_start = None
    anomaly_event_start = None
    consecutive_normal_count = 0
    current_anomaly_ratio = 0.0

    alert_events = []
    alert_notification_sent = False

    motion_detector = MovementDetector(
            history=500,
            var_threshold=25,
            detect_shadows=True,
            min_motion_ratio=0.01,
            warmup_frames=max(30, int(fps * 2)),
            confirm_frames=3,
            resize_width=320,
    )

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            if CONFIG.get("is_live_stream", is_live_like_source):
                print("⚠️ 直播串流暫時中斷，嘗試重新連線...")

                cap.release()

                reconnect_success = False

                for attempt in range(1, 6):
                    print(f"🔄 第 {attempt}/5 次重新連線...")

                    try:
                        time.sleep(2)

                        cap, resolved_source = open_video_capture(video_path)

                        test_ret, test_frame = cap.read()

                        if test_ret and test_frame is not None:
                            print("✅ RTSP 重新連線成功。")
                            frame = test_frame
                            ret = True
                            reconnect_success = True

                            skeleton_buffer.clear()
                            anomaly_vote_history.clear()
                            pre_event_buffer.clear()

                            detection_state = "normal"
                            anomaly_candidate_start = None
                            anomaly_event_start = None
                            consecutive_normal_count = 0
                            current_anomaly_ratio = 0.0
                            motion_hold_remaining = 0
                            break

                        cap.release()

                    except Exception as e:
                        print(f"⚠️ 第 {attempt} 次重新連線失敗：{e}")

                if not reconnect_success:
                    print("❌ RTSP 連續重新連線失敗，停止推論。")
                    break
            else:
                print("🏁 影片已播放完畢，結束推論。")
                break

        current_sec = frame_idx / fps
        final_video_sec = current_sec

        pre_event_buffer.append({
            "time": current_sec,
            "frame": frame.copy(),
        })

        if collecting_event:
            if (
                not event_frames
                or current_sec > event_frames[-1]["time"] + 1e-6
            ):
                event_frames.append({
                    "time": current_sec,
                    "frame": frame.copy(),
                })

        if (
            collecting_event
            and event_collect_until_sec is not None
            and current_sec >= event_collect_until_sec
        ):
            try:
                event_result = finish_event_collection(
                    event_id=event_id,
                    event_frames=event_frames,
                    anomaly_flags=post_event_anomaly_flags,
                    fps=fps,
                )

                if event_result is None:
                    print("✅ 此事件已丟棄，不進行 VLM 分析。")
                else:
                    frame_paths = event_result["frame_paths"]
                    vlm_future = vlm_executor.submit(
                        _analyze_event_with_vlm,
                        frame_paths,
                        line_user_id,
                    )
                    vlm_future.add_done_callback(
                        _report_background_vlm_result
                    )
                    print("▶️ VLM 已在背景執行，持續進行影片分析。")

            except Exception as exc:
                print(f"❌ 異常事件處理失敗：{exc}")

            finally:
                collecting_event = False
                event_id = None
                event_start_sec = None
                event_collect_until_sec = None
                event_frames = []
                post_event_anomaly_flags = []

        raw_has_movement, motion_ratio, fg_mask = (
            motion_detector.check_whether_move(frame)
        )

        if raw_has_movement:
            motion_hold_remaining = motion_grace_frames
        elif motion_hold_remaining > 0:
            motion_hold_remaining -= 1

        has_movement = raw_has_movement or motion_hold_remaining > 0
        if not has_movement:
            if CONFIG.get("show_yolo_window", True):
                display_frame = frame.copy()
                cv2.putText(
                    display_frame,
                    f"NO MOVEMENT ({motion_ratio:.2%})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(
                    "ST-CROSR Live Real-Time Radar Monitor",
                    display_frame,
                )
                cv2.imshow(
                    "MOG2 Foreground Mask",
                    fg_mask,
                )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stopped_by_user = True
                    break

            frame_idx += 1
            continue

        one_frame_skeleton = np.zeros(
            (2, 17),
            dtype=np.float32,
        )
        results = yolo_model(
            frame,
            verbose=False,
        )

        if (
            len(results) > 0
            and results[0].keypoints is not None
        ):
            keypoints = results[0].keypoints.xy

            if (
                keypoints is not None
                and len(keypoints) > 0
            ):
                person_kpts = (
                    keypoints[0]
                    .detach()
                    .cpu()
                    .numpy()
                )

                if person_kpts.shape[0] >= 17:
                    one_frame_skeleton[0, :] = (
                        person_kpts[:17, 0]
                    )
                    one_frame_skeleton[1, :] = (
                        person_kpts[:17, 1]
                    )

        skeleton_buffer.append(one_frame_skeleton)

        max_buffer_size = max(
            window_size * 2,
            window_size + stride
        )

        if len(skeleton_buffer) > max_buffer_size:
            del skeleton_buffer[:-max_buffer_size]

        if (
            len(skeleton_buffer) >= window_size
            and frame_idx % stride == 0
        ):
            clip = np.stack(
                skeleton_buffer[-window_size:],
                axis=0
            )

            clip = np.transpose(
                clip,
                (1, 0, 2)
            )

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

            raw_is_unknown = bool(
                current_radar_res["is_unknown"]
            )

            anomaly_vote_history.append(
                raw_is_unknown
            )

            if len(anomaly_vote_history) > vote_history_size:
                del anomaly_vote_history[:-vote_history_size]

            anomaly_votes = int(
                sum(anomaly_vote_history)
            )
            total_votes = len(
                anomaly_vote_history
            )

            current_anomaly_ratio = (
                anomaly_votes / total_votes
                if total_votes > 0
                else 0.0
            )

            enough_history = (
                total_votes >= min_anomaly_votes
            )

            voted_anomaly = (
                enough_history
                and anomaly_votes >= min_anomaly_votes
                and current_anomaly_ratio
                >= anomaly_vote_ratio
            )

            if (
                collecting_event
                and event_collect_until_sec is not None
                and current_sec <= event_collect_until_sec
            ):
                post_event_anomaly_flags.append(voted_anomaly)

            if detection_state == "normal":
                consecutive_normal_count = 0

                if voted_anomaly:
                    detection_state = "candidate"

                    first_anomaly_index = next(
                        (
                            index
                            for index, is_anomaly
                            in enumerate(anomaly_vote_history)
                            if is_anomaly
                        ),
                        len(anomaly_vote_history) - 1
                    )

                    evaluations_ago = (
                        len(anomaly_vote_history)
                        - 1
                        - first_anomaly_index
                    )

                    anomaly_candidate_start = max(
                        0.0,
                        current_sec
                        - evaluations_ago * evaluation_interval_sec
                    )

                    candidate_duration = (
                        current_sec - anomaly_candidate_start
                    )

                    print(
                        f"⏳ [{current_sec:6.2f} 秒] "
                        f"進入異常候選 | "
                        f"推估已持續 {candidate_duration:.2f} 秒 | "
                        f"異常視窗比例 {current_anomaly_ratio:.0%} "
                        f"({anomaly_votes}/{total_votes}) | "
                        f"分數 {current_radar_res['combined_score']:.4f}"
                    )

                    if candidate_duration >= consecutive_alert_sec:
                        detection_state = "alert"
                        anomaly_event_start = anomaly_candidate_start
                        consecutive_normal_count = 0

                        if not collecting_event:
                            event_id = f"event_{int(frame_idx):08d}_{int(current_sec * 1000):010d}"
                            (
                                event_frames,
                                event_start_sec,
                                event_collect_until_sec,
                            ) = start_event_collection(
                                pre_event_buffer=pre_event_buffer,
                                current_sec=current_sec,
                                anomaly_event_start=anomaly_event_start,
                            )
                            post_event_anomaly_flags = [voted_anomaly]
                            collecting_event = True
                            print(
                                f"🎞️ 開始收集事件 {event_id} | "
                                f"前段起點 {event_start_sec:.2f} 秒 | "
                                f"收集至 {event_collect_until_sec:.2f} 秒"
                            )

                        if (
                            current_sec - last_alert_time
                            >= alert_cooldown_sec
                        ):
                            print(
                                f"🚨 【確認異常】"
                                f"{current_sec:6.2f} 秒 | "
                                f"異常持續 {candidate_duration:.1f} 秒 | "
                                f"異常視窗比例 "
                                f"{current_anomaly_ratio:.0%} | "
                                f"綜合分數 "
                                f"{current_radar_res['combined_score']:.4f}"
                            )

                            print("📹 骨架模型確認疑似異常，等待 VLM 二次判斷。")

                            last_alert_time = current_sec

            elif detection_state == "candidate":
                if voted_anomaly:
                    candidate_duration = (
                        current_sec - anomaly_candidate_start
                        if anomaly_candidate_start is not None
                        else 0.0
                    )

                    if (
                        candidate_duration
                        >= consecutive_alert_sec
                    ):
                        detection_state = "alert"
                        anomaly_event_start = (
                            anomaly_candidate_start
                            if anomaly_candidate_start is not None
                            else current_sec
                        )
                        consecutive_normal_count = 0

                        if not collecting_event:
                            event_id = f"event_{int(frame_idx):08d}_{int(current_sec * 1000):010d}"
                            (
                                event_frames,
                                event_start_sec,
                                event_collect_until_sec,
                            ) = start_event_collection(
                                pre_event_buffer=pre_event_buffer,
                                current_sec=current_sec,
                                anomaly_event_start=anomaly_event_start,
                            )
                            post_event_anomaly_flags = [voted_anomaly]
                            collecting_event = True
                            print(
                                f"🎞️ 開始收集事件 {event_id} | "
                                f"前段起點 {event_start_sec:.2f} 秒 | "
                                f"收集至 {event_collect_until_sec:.2f} 秒"
                            )

                        if (
                            current_sec - last_alert_time
                            >= alert_cooldown_sec
                        ):
                            print(
                                f"🚨 【確認異常】"
                                f"{current_sec:6.2f} 秒 | "
                                f"候選持續 "
                                f"{candidate_duration:.1f} 秒 | "
                                f"異常視窗比例 "
                                f"{current_anomaly_ratio:.0%} | "
                                f"綜合分數 "
                                f"{current_radar_res['combined_score']:.4f}"
                            )

                            print(
                                "    -> 最接近正常動作："
                                f"{current_radar_res['nearest_action_name']}"
                            )

                            last_alert_time = current_sec

                    else:
                        print(
                            f"⏳ [異常確認中] "
                            f"{current_sec:6.2f} 秒 | "
                            f"{candidate_duration:.1f}/"
                            f"{consecutive_alert_sec:.1f} 秒 | "
                            f"投票 "
                            f"{current_anomaly_ratio:.0%}"
                        )

                else:
                    print(
                        f"✅ [{current_sec:6.2f} 秒] "
                        "異常候選未持續，恢復正常。"
                    )

                    detection_state = "normal"
                    anomaly_candidate_start = None

            elif detection_state == "alert":
                if voted_anomaly:
                    consecutive_normal_count = 0

                    if (
                        current_sec - last_alert_time
                        >= alert_cooldown_sec
                    ):
                        abnormal_duration = (
                            current_sec - anomaly_event_start
                            if anomaly_event_start is not None
                            else 0.0
                        )

                        print(
                            f"🚨 [異常持續] "
                            f"{current_sec:6.2f} 秒 | "
                            f"已持續約 "
                            f"{abnormal_duration:.1f} 秒 | "
                            f"投票 "
                            f"{current_anomaly_ratio:.0%}"
                        )

                        last_alert_time = current_sec

                else:
                    consecutive_normal_count += 1

                    print(
                        f"🔄 [異常解除確認] "
                        f"{current_sec:6.2f} 秒 | "
                        f"正常 "
                        f"{consecutive_normal_count}/"
                        f"{clear_normal_windows} 個視窗"
                    )

                    if (
                        consecutive_normal_count
                        >= clear_normal_windows
                    ):
                        print(
                            f"✅ [{current_sec:6.2f} 秒] "
                            "異常已解除，恢復正常監控。"
                        )

                        detection_state = "normal"
                        anomaly_candidate_start = None
                        anomaly_event_start = None
                        consecutive_normal_count = 0
                        anomaly_vote_history.clear()
                        current_anomaly_ratio = 0.0

        if CONFIG.get("show_yolo_window", True):
            display_frame = frame.copy()

            if is_live_like_source:
                time_text = (
                    f"Time: {current_sec:.2f}s / LIVE"
                )
            else:
                time_text = (
                    f"Time: {current_sec:.2f}s / "
                    f"{total_frames / fps:.2f}s"
                )

            cv2.rectangle(
                display_frame,
                (10, 10),
                (620, 120),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                display_frame,
                time_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            if detection_state == "alert":
                status_text = (
                    f"ALARM: UNKNOWN "
                    f"(vote "
                    f"{current_anomaly_ratio * 100:.0f}%)"
                )
                status_color = (0, 0, 255)

            elif detection_state == "candidate":
                elapsed = (
                    current_sec - anomaly_candidate_start
                    if anomaly_candidate_start is not None
                    else 0.0
                )

                status_text = (
                    f"WARNING: VERIFYING "
                    f"{elapsed:.1f}/"
                    f"{consecutive_alert_sec:.1f}s "
                    f"(vote "
                    f"{current_anomaly_ratio * 100:.0f}%)"
                )
                status_color = (0, 165, 255)

            else:
                status_text = (
                    f"STATUS: NORMAL "
                    f"(vote "
                    f"{current_anomaly_ratio * 100:.0f}%)"
                )
                status_color = (0, 255, 0)

            cv2.putText(
                display_frame,
                status_text,
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                status_color,
                2,
                cv2.LINE_AA
            )

            if current_radar_res is not None:
                score_text = (
                    f"Score: "
                    f"{current_radar_res['combined_score']:.4f} "
                    f"/ Thresh: {threshold:.4f} | "
                    f"Nearest: "
                    f"{current_radar_res['nearest_action_name']}"
                )

                cv2.putText(
                    display_frame,
                    score_text,
                    (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    status_color,
                    2,
                    cv2.LINE_AA
                )

            cv2.imshow(
                "ST-CROSR Live Real-Time Radar Monitor",
                display_frame
            )

            delay = max(
                1,
                int(1000 / fps * 0.5)
            )

            if cv2.waitKey(delay) & 0xFF == ord("q"):
                print("🛑 使用者手動中斷串流播放。")
                stopped_by_user = True
                break

        frame_idx += 1

    if collecting_event and event_frames:
        print("⚠️ 影片已結束，使用已收集到的後置影格完成事件判斷。")
        try:
            event_result = finish_event_collection(
                event_id=event_id,
                event_frames=event_frames,
                anomaly_flags=post_event_anomaly_flags,
                fps=fps,
                force_partial=True,
            )

            if event_result is None:
                print("✅ 尾端事件未達保留門檻，不進行 VLM 分析。")
            else:
                frame_paths = event_result["frame_paths"]
                vlm_result = analyze_frames_with_ollama(frame_paths)

                print(
                    "🧠 Ollama VLM 分析結果：",
                    vlm_result,
                )
                should_alert = (
                    vlm_result["is_abnormal"]
                    and vlm_result["need_alert"]
                    and vlm_result["confidence"] >= 0.75
                )
                if should_alert:
                    alert_text = (
                        "🚨 VLM 確認異常事件\n"
                        f"類型：{vlm_result['category']}\n"
                        f"信心度：{vlm_result['confidence']:.0%}\n"
                        f"描述：{vlm_result['description']}"
                    )
                    if line_user_id:
                        push_line_message(
                            line_user_id,
                            alert_text,
                        )
                    else:
                        print(
                            "⚠️ line_user_id 為空，"
                            "不發送 LINE 警報。"
                        )
                else:
                    print(
                        "✅ VLM 判斷未達警報門檻，"
                        "不發送 LINE。"
                    )
        except Exception as exc:
            print(f"❌ 尾端異常事件保存失敗：{exc}")

    cap.release()
    cv2.destroyAllWindows()
    vlm_executor.shutdown(wait=False)

    print("\n🏁 影片串流即時掃描結束。")
