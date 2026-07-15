import os
import sys
import json
import hmac
import base64
import hashlib
import requests
import subprocess
import time

from flask import Flask, request, abort
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

# 記錄每位使用者目前的狀態，例如 "waiting_for_link" 表示正在等待輸入連結
user_states = {}

# 記錄每位使用者目前執行中的影片分析程序
# 格式：
# running_processes[user_id] = {
#     "process": subprocess.Popen,
#     "video_url": "影片連結"
# }
running_processes = {}

# alarm_YT.py 的 checkpoint 路徑是相對路徑，必須以專案根目錄為工作目錄執行
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def start_alarm_process(video_url: str, user_id: str):
    """
    在背景啟動 alarm_YT.py。
    同時保存程序、影片連結與開始時間。
    """
    script_path = os.path.join(BASE_DIR, "alarm_YT.py")

    old_task = running_processes.get(user_id)

    if old_task is not None:
        old_process = old_task.get("process")

        if old_process is not None and old_process.poll() is None:
            raise RuntimeError("目前已有影片正在分析")

        running_processes.pop(user_id, None)

    process = subprocess.Popen(
        [sys.executable, script_path, video_url, user_id],
        cwd=BASE_DIR
    )

    running_processes[user_id] = {
        "process": process,
        "video_url": video_url,
        "started_at": time.time()
    }

    print(f"LINE user_id：{user_id}")
    print(f"已啟動 alarm_YT.py（PID: {process.pid}）")
    print(f"影片來源：{video_url}")

    return process


def stop_alarm_process(user_id: str):
    """
    停止指定 LINE 使用者目前正在執行的影片分析。

    回傳：
    True  -> 成功停止
    False -> 目前沒有正在執行的分析
    """
    task = running_processes.get(user_id)

    if task is None:
        return False

    process = task.get("process")

    # 程序不存在或早已結束
    if process is None or process.poll() is not None:
        running_processes.pop(user_id, None)
        return False

    print(f"準備停止使用者 {user_id} 的影片分析，PID：{process.pid}")

    try:
        # 先嘗試正常終止
        process.terminate()

        try:
            # 最多等待 5 秒
            process.wait(timeout=5)
            print(f"✅ 已停止影片分析，PID：{process.pid}")

        except subprocess.TimeoutExpired:
            # 無法正常停止時強制結束
            print(f"⚠️ 程序未正常停止，強制結束 PID：{process.pid}")
            process.kill()
            process.wait(timeout=5)

    except Exception as e:
        print(f"❌ 停止影片分析失敗：{e}")
        raise

    finally:
        running_processes.pop(user_id, None)

    return True

def format_elapsed_time(seconds: float) -> str:
    """
    將秒數轉成容易閱讀的時間格式。
    """
    seconds = max(0, int(seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours} 小時 {minutes} 分 {secs} 秒"

    if minutes > 0:
        return f"{minutes} 分 {secs} 秒"

    return f"{secs} 秒"


def get_alarm_status(user_id: str) -> str:
    """
    查詢指定使用者的影片分析狀態。
    """
    task = running_processes.get(user_id)

    if task is None:
        return "目前沒有影片分析紀錄。"

    process = task.get("process")
    video_url = task.get("video_url", "未知")
    started_at = task.get("started_at", time.time())

    elapsed_seconds = time.time() - started_at
    elapsed_text = format_elapsed_time(elapsed_seconds)

    if process is None:
        running_processes.pop(user_id, None)
        return "目前沒有正在執行的影片分析。"

    return_code = process.poll()

    # poll() 回傳 None，表示程序仍在執行
    if return_code is None:
        return (
            "🟢 影片分析中\n"
            f"已執行時間：{elapsed_text}\n"
            f"程序編號：{process.pid}\n"
            f"影片來源：{video_url}"
        )

    # return code 為 0，表示正常結束
    if return_code == 0:
        running_processes.pop(user_id, None)

        return (
            "✅ 影片分析已完成\n"
            f"總執行時間：{elapsed_text}"
        )

    # 非 0 通常表示異常退出
    running_processes.pop(user_id, None)

    return (
        "❌ 影片分析已異常停止\n"
        f"執行時間：{elapsed_text}\n"
        f"錯誤代碼：{return_code}"
    )
def verify_signature(body: bytes, signature: str) -> bool:
    """
    驗證 Webhook 是否真的是 LINE 傳來的。
    """
    hash_value = hmac.new(
        LINE_CHANNEL_SECRET.encode("utf-8"),
        body,
        hashlib.sha256
    ).digest()

    expected_signature = base64.b64encode(hash_value).decode("utf-8")

    return hmac.compare_digest(expected_signature, signature)


def reply_messages(reply_token: str, messages: list):
    url = "https://api.line.me/v2/bot/message/reply"

    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    data = {
        "replyToken": reply_token,
        "messages": messages
    }

    response = requests.post(url, headers=headers, json=data)

    print("LINE API 狀態碼：", response.status_code)
    print("LINE API 回應：", response.text)

    response.raise_for_status()
    

def create_menu_button_message():
    return {
        "type": "template",
        "altText": "主選單",
        "template": {
            "type": "buttons",
            "title": "主選單",
            "text": "請選擇你要的功能",
            "actions": [
                {
                    "type": "message",
                    "label": "上傳影片",
                    "text": "上傳影片"
                },
                {
                    "type": "message",
                    "label": "查看狀態",
                    "text": "查看狀態"
                },
                {
                    "type": "message",
                    "label": "停止分析",
                    "text": "停止分析"
                }
                # {
                #     "type": "message",
                #     "label": "開燈",
                #     "text": "開燈"
                # },
                # {
                #     "type": "message",
                #     "label": "關燈",
                #     "text": "關燈"
                # },
                # {
                #     "type": "uri",
                #     "label": "開啟網站",
                #     "uri": "https://www.google.com"
                # }
            ]
        }
    }

@app.route("/", methods=["GET"])
def home():
    return "LINE Bot server is running."


@app.route("/callback", methods=["POST"])
def callback():
    body = request.get_data()
    signature = request.headers.get("X-Line-Signature", "")

    if not verify_signature(body, signature):
        print("簽章驗證失敗")
        abort(400)

    data = json.loads(body.decode("utf-8"))

    print("收到 LINE Webhook：")
    print(json.dumps(data, ensure_ascii=False, indent=2))

    for event in data.get("events", []):
        event_type = event.get("type")

        if event_type == "message":
            message = event.get("message", {})
            message_type = message.get("type")

            if message_type == "text":
                user_text = message.get("text")
                reply_token = event.get("replyToken")
                user_id = event.get("source", {}).get("userId")

                if user_text == "選單":
                    user_states.pop(user_id, None)
                    menu_message = create_menu_button_message()
                    reply_messages(reply_token, [menu_message])

                elif user_text == "上傳影片":
                    user_states[user_id] = "waiting_for_link"
                    reply_messages(reply_token, [
                        {
                            "type": "text",
                            "text": "請輸入連結"
                        }
                    ])
                elif user_text == "查看狀態":
                    status_text = get_alarm_status(user_id)

                    reply_messages(reply_token, [
                        {
                            "type": "text",
                            "text": status_text
                        }
                    ])
                elif user_text == "停止分析":
                    # 同時取消等待連結的狀態
                    user_states.pop(user_id, None)

                    try:
                        stopped = stop_alarm_process(user_id)

                        if stopped:
                            reply_messages(reply_token, [
                                {
                                    "type": "text",
                                    "text": "⛔ 已停止目前的影片分析。"
                                }
                            ])
                        else:
                            reply_messages(reply_token, [
                                {
                                    "type": "text",
                                    "text": "目前沒有正在執行的影片分析。"
                                }
                            ])

                    except Exception as e:
                        print(f"停止影片分析失敗：{e}")

                        reply_messages(reply_token, [
                            {
                                "type": "text",
                                "text": "停止影片分析失敗，請稍後再試。"
                            }
                        ])

                elif user_states.get(user_id) == "waiting_for_link":
                    user_states.pop(user_id, None)
                    print(f"收到使用者 {user_id} 的影片連結：{user_text}")

                    try:
                        start_alarm_process(user_text, user_id)
                        reply_messages(reply_token, [
                            {
                                "type": "text",
                                "text": "已收到連結，開始分析影片！"
                            }
                        ])
                    except RuntimeError as e:
                        print(f"無法啟動分析：{e}")

                        reply_messages(reply_token, [
                            {
                                "type": "text",
                                "text": "目前已有影片正在分析，請先選擇「停止分析」，再上傳新的影片。"
                            }
                        ])

                    except Exception as e:
                        print(f"啟動 alarm_YT.py 失敗：{e}")

                        reply_messages(reply_token, [
                            {
                                "type": "text",
                                "text": "啟動分析失敗，請稍後再試。"
                            }
                        ])

                # elif user_text == "開燈":
                #     reply_messages(reply_token, [
                #         {
                #             "type": "text",
                #             "text": "已送出開燈指令"
                #         }
                #     ])

                # elif user_text == "關燈":
                #     reply_messages(reply_token, [
                #         {
                #             "type": "text",
                #             "text": "已送出關燈指令"
                #         }
                #     ])

                else:
                    reply_messages(reply_token, [
                        {
                            "type": "text",
                            "text": "請輸入「選單」查看功能"
                        }
                    ])

    return "OK"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)