import requests
from constants import LINE_CHANNEL_ACCESS_TOKEN


def format_video_time(seconds):
    """
    將秒數轉換成適合 LINE 顯示的時間格式。
    例如：
    65.3 秒 -> 01:05
    3665 秒 -> 01:01:05
    """
    seconds = max(0, int(seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    return f"{minutes:02d}:{secs:02d}"


def push_line_message(user_id: str, text: str):
    """
    使用 LINE Push Message API 主動傳訊息給指定使用者。
    如果沒有 user_id，就進入測試模式，只把訊息印在終端機。
    """
    if not user_id:
        print("\n" + "=" * 60)
        print("🧪 LINE 推播測試模式")
        print("因為沒有 LINE user_id，所以不會真的傳送。")
        print("如果正式推播，LINE 會收到以下訊息：")
        print("-" * 60)
        print(text)
        print("=" * 60 + "\n")
        return

    if not LINE_CHANNEL_ACCESS_TOKEN:
        print("⚠️ 缺少 LINE_CHANNEL_ACCESS_TOKEN，跳過 LINE 推播。")
        return

    url = "https://api.line.me/v2/bot/message/push"

    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    data = {
        "to": user_id,
        "messages": [
            {
                "type": "text",
                "text": text
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)

        print("LINE Push 狀態碼：", response.status_code)
        print("LINE Push 回應：", response.text)

        response.raise_for_status()

    except Exception as e:
        print("❌ LINE Push Message 發送失敗：")
        print(e)


def build_alert_message(current_sec, consecutive_duration, radar_res, threshold):
    """
    組出要傳到 LINE 的異常警報文字。
    """
    nearest_action = radar_res.get("nearest_action_name", "unknown")
    combined_score = radar_res.get("combined_score", 0)

    return (
        "🚨 偵測到異常動作！\n"
        f"發生時間：約第 {current_sec:.2f} 秒\n"
        f"連續異常：約 {consecutive_duration:.1f} 秒\n"
        f"異常分數：{combined_score:.4f}\n"
        f"判斷閾值：{threshold:.4f}\n"
    )


def build_analysis_summary(video_duration, alert_events, stopped_by_user=False):
    """
    建立影片分析完成後的 LINE 摘要訊息。
    """

    if stopped_by_user:
        title = "⛔ 影片分析已停止"
    else:
        title = "✅ 影片分析完成"

    summary_lines = [
        title,
        "",
        f"影片分析長度：{format_video_time(video_duration)}",
        f"異常事件數量：{len(alert_events)} 次"
    ]

    if len(alert_events) == 0:
        summary_lines.extend([
            "",
            "本次分析未偵測到符合警報條件的異常事件。"
        ])

        return "\n".join(summary_lines)

    highest_score = max(
        event.get("max_score", 0)
        for event in alert_events
    )

    summary_lines.append(f"最高異常分數：{highest_score:.4f}")
    summary_lines.append("")
    summary_lines.append("📋 異常事件紀錄")

    max_display_events = 10

    for index, event in enumerate(
        alert_events[:max_display_events],
        start=1
    ):
        start_sec = event.get("start_sec", 0)
        end_sec = event.get("end_sec", start_sec)
        duration = event.get("duration", end_sec - start_sec)
        max_score = event.get("max_score", 0)
        nearest_action = event.get("nearest_action", "unknown")

        summary_lines.extend([
            "",
            f"事件 {index}",
            (
                f"時間：{format_video_time(start_sec)}"
                f" ～ {format_video_time(end_sec)}"
            ),
            f"持續：約 {duration:.1f} 秒",
            f"最高分數：{max_score:.4f}",
            f"最接近正常動作：{nearest_action}"
        ])

    if len(alert_events) > max_display_events:
        remaining = len(alert_events) - max_display_events

        summary_lines.extend([
            "",
            f"另外還有 {remaining} 個異常事件未顯示。"
        ])

    return "\n".join(summary_lines)
