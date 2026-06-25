import os
import json
import hmac
import base64
import hashlib
import requests

from flask import Flask, request, abort
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")


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

                if user_text == "選單":
                    menu_message = create_menu_button_message()
                    reply_messages(reply_token, [menu_message])

                elif user_text == "上傳影片":
                    reply_messages(reply_token, [
                        {
                            "type": "text",
                            "text": "請輸入連結"
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