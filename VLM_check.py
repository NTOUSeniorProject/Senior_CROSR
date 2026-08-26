import base64
import json
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()

DEFAULT_OLLAMA_BASE_URL = "http://26.184.142.137:11434"
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    DEFAULT_OLLAMA_BASE_URL,
).rstrip("/")
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "blaifa/InternVL3_5:8B",
)

ABNORMAL_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "is_abnormal": {
            "type": "boolean"
        },
        "category": {
            "type": "string",
            "enum": [
                "fall",
                "collapse",
                "fighting",
                "prolonged_lying",
                "unsafe_climbing",
                "distress",
                "normal",
                "uncertain"
            ]
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
        },
        "description": {
            "type": "string"
        },
        "need_alert": {
            "type": "boolean"
        }
    },
    "required": [
        "is_abnormal",
        "category",
        "confidence",
        "description",
        "need_alert"
    ],
    "additionalProperties": False
}


VLM_PROMPT = """
你是一個監視器異常事件分析系統。

提供的圖片是由同一段影片按照時間先後順序抽取的影格。
請根據整段動作變化，而不是單獨一張圖片，判斷是否發生異常事件。

需要辨識的事件：

- fall：人物失去平衡並跌倒
- collapse：突然昏倒、癱倒或無力倒下
- fighting：人物之間有攻擊、推擠或肢體衝突
- prolonged_lying：人物倒地後持續躺著，沒有正常起身
- unsafe_climbing：危險攀爬或可能墜落
- distress：明顯求救、痛苦或身體不適
- normal：正常行為
- uncertain：畫面不足或無法確定

判斷規則：

1. 彎腰、蹲下、坐下、撿東西不等於跌倒。
2. 必須比較前後影格的人物高度、姿勢和位置。
3. 跌倒通常包含快速下降、身體傾斜，以及倒地後沒有立即恢復。
4. 畫面不清楚、人物被遮擋或證據不足時，請回傳 uncertain。
5. confidence 為 0 到 1。
6. 只有高風險且需要通知照護者時，need_alert 才設為 true。
7. description 使用繁體中文，簡潔描述觀察到的動作變化。
"""


def image_to_base64(image_path: str) -> str:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"找不到圖片：{image_path}")

    with path.open("rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def analyze_frames_with_ollama(
    frame_paths: list[str],
    model: str = OLLAMA_MODEL,
    timeout: int = 180,
) -> dict[str, Any]:
    if not frame_paths:
        raise ValueError("沒有提供任何影格")

    encoded_images = [
        image_to_base64(frame_path)
        for frame_path in frame_paths
    ]

    payload = {
        "model": model,
        "stream": False,
        "format": ABNORMAL_RESULT_SCHEMA,
        "messages": [
            {
                "role": "user",
                "content": VLM_PROMPT,
                "images": encoded_images
            }
        ],
        "options": {
            "temperature": 0.1,
            "num_predict": 300,
            "num_ctx": 32768
        },
        # 推論結束後保留模型 5 分鐘，避免每次重新載入
        "keep_alive": "5m"
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()

    except requests.ConnectionError as error:
        raise RuntimeError(
            "無法連線到 Ollama，請確認 Ollama 已啟動，"
            f"並可存取 {OLLAMA_BASE_URL}"
        ) from error

    except requests.Timeout as error:
        raise RuntimeError(
            f"Ollama 推論超過 {timeout} 秒"
        ) from error

    except requests.HTTPError as error:
        raise RuntimeError(
            f"Ollama API 發生錯誤："
            f"{response.status_code} {response.text}"
        ) from error

    response_data = response.json()
    raw_content = response_data["message"]["content"]

    try:
        result = json.loads(raw_content)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Ollama 回傳的內容不是合法 JSON：{raw_content}"
        ) from error

    return {
        "is_abnormal": bool(result["is_abnormal"]),
        "category": str(result["category"]),
        "confidence": float(result["confidence"]),
        "description": str(result["description"]),
        "need_alert": bool(result["need_alert"]),
        "total_duration": response_data.get("total_duration"),
        "load_duration": response_data.get("load_duration"),
        "eval_count": response_data.get("eval_count")
    }
