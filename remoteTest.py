"""在 PC-lab 直接測試大型主機 VLM。

文字測試：
    python remoteTest.py

單張圖片測試：
    python remoteTest.py --image C:/path/test.jpg
"""

import argparse
import base64
import mimetypes
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv


load_dotenv()

DEFAULT_BASE_URL = "http://192.168.50.51:8001/v1"
DEFAULT_MODEL = "OpenGVLab/InternVL3-78B-AWQ"


def image_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到圖片：{path}")

    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def request_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def show_models(
    base_url: str,
    headers: dict[str, str],
    connect_timeout: float,
) -> set[str]:
    print(f"[1/2] 讀取模型清單：{base_url}/models")
    response = requests.get(
        f"{base_url}/models",
        headers=headers,
        timeout=connect_timeout,
    )
    response.raise_for_status()
    data = response.json()
    model_ids = {
        model.get("id")
        for model in data.get("data", [])
        if model.get("id")
    }

    if model_ids:
        for model_id in sorted(model_ids):
            print(f"  - {model_id}")
    else:
        print("  API 可連線，但模型清單為空。")
    return model_ids


def build_content(image_path: str | None) -> list[dict]:
    if image_path is None:
        return [
            {
                "type": "text",
                "text": "請只回答：大型主機 VLM API 呼叫成功",
            }
        ]

    return [
        {
            "type": "text",
            "text": (
                "請使用繁體中文簡短描述圖片中的人物、動作與環境；"
                "如果沒有看到人物，也請明確說明。"
            ),
        },
        {
            "type": "image_url",
            "image_url": {
                "url": image_to_data_url(image_path),
            },
        },
    ]


def run_completion(
    base_url: str,
    headers: dict[str, str],
    model: str,
    image_path: str | None,
    connect_timeout: float,
    inference_timeout: float,
) -> str:
    mode = "單張圖片" if image_path else "文字"
    print(f"[2/2] 執行 {mode} Chat Completions 推論...")

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": build_content(image_path),
            }
        ],
        "temperature": 0.1,
        "max_tokens": 200,
    }

    started_at = time.perf_counter()
    response = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        json=payload,
        timeout=(connect_timeout, inference_timeout),
    )
    response.raise_for_status()
    response_data = response.json()

    try:
        content = response_data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise RuntimeError(
            f"API 回應缺少 choices[0].message.content：{response_data}"
        ) from error

    if not content:
        raise RuntimeError("模型回覆內容為空")

    elapsed = time.perf_counter() - started_at
    print(f"  推論時間：{elapsed:.2f} 秒")
    return content


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在 PC-lab 直接呼叫大型主機的 OpenAI-compatible VLM。"
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LARGE_VLM_BASE_URL", DEFAULT_BASE_URL),
        help=f"OpenAI-compatible API 根網址（預設：{DEFAULT_BASE_URL}）",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LARGE_VLM_MODEL", DEFAULT_MODEL),
        help=f"模型名稱（預設：{DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LARGE_VLM_API_KEY", "EMPTY"),
        help="API Key；預設讀取 LARGE_VLM_API_KEY，未設定時使用 EMPTY",
    )
    parser.add_argument(
        "--image",
        help="本機單張圖片路徑；省略時執行文字推論",
    )
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--inference-timeout", type=float, default=300.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base_url = args.base_url.rstrip("/")

    if args.connect_timeout <= 0 or args.inference_timeout <= 0:
        print("[失敗] timeout 必須大於 0。", file=sys.stderr)
        return 1

    headers = request_headers(args.api_key)
    print(f"API：{base_url}")
    print(f"模型：{args.model}")

    try:
        model_ids = show_models(base_url, headers, args.connect_timeout)
        if model_ids and args.model not in model_ids:
            print(
                f"[失敗] 模型清單中找不到 {args.model}",
                file=sys.stderr,
            )
            return 2

        content = run_completion(
            base_url,
            headers,
            args.model,
            args.image,
            args.connect_timeout,
            args.inference_timeout,
        )
    except FileNotFoundError as error:
        print(f"[失敗] {error}", file=sys.stderr)
        return 1
    except requests.ConnectionError:
        print(
            f"[失敗] 無法連線到 {base_url}。請確認 PC-lab 位於正確網路。",
            file=sys.stderr,
        )
        return 1
    except requests.Timeout:
        print("[失敗] API 連線或模型推論逾時。", file=sys.stderr)
        return 1
    except requests.HTTPError as error:
        response = error.response
        status = response.status_code if response is not None else "未知"
        detail = response.text[:1000] if response is not None else str(error)
        print(f"[失敗] HTTP {status}：{detail}", file=sys.stderr)
        return 1
    except (requests.JSONDecodeError, ValueError, RuntimeError) as error:
        print(f"[失敗] {error}", file=sys.stderr)
        return 1

    print("\n[成功] 大型主機 VLM 呼叫完成。")
    print(f"模型回覆：{content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
