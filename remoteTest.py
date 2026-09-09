"""測試兩台 Ollama 或 OpenAI-compatible VLM API。

選項 1、2 使用 Ollama API；選項 3 使用 OpenAI Python SDK。
未指定圖片時只檢查連線；使用 --image 時會執行一次單張圖片推論。
"""

import argparse
import base64
import mimetypes
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


OLLAMA_TARGETS = {
    "1": "26.184.142.137",
    "2": "26.247.236.14",
}
DEFAULT_OLLAMA_PORT = 11434
DEFAULT_MODEL = "blaifa/InternVL3_5:8B"
OPENAI_BASE_URL = "http://26.184.142.137:9000/v1"
OPENAI_MODEL = "OpenGVLab/InternVL3-78B-AWQ"
OPENAI_API_KEY = "EMPTY"


def select_targets(selection: str) -> list[str]:
    """將 1／2 轉成要測試的 Ollama 主機，也接受自訂位址。"""
    selection = selection.strip()
    if selection == "1":
        return [OLLAMA_TARGETS["1"]]
    if selection == "2":
        return [OLLAMA_TARGETS["2"]]
    if selection:
        return [selection]
    raise ValueError("請輸入 1、2、3 或自訂位址")


def prompt_target_selection() -> str:
    print("請選擇測試目標：")
    print(f"  1：Ollama {OLLAMA_TARGETS['1']}")
    print(f"  2：Ollama {OLLAMA_TARGETS['2']}")
    print(f"  3：經 PC-lab 轉送至 78B：{OPENAI_BASE_URL}")
    return input("請輸入 1、2 或 3：").strip()


def image_to_data_url(image_path: str) -> str:
    """將本機圖片轉成 Chat Completions 使用的 data URL。"""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到測試圖片：{path}")

    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def test_openai_compatible(
    image_path: str | None,
    timeout: float,
) -> int:
    """使用 OpenAI Python SDK 測試模型主機及選用的圖片推論。"""
    try:
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            OpenAI,
        )
    except ImportError:
        print(
            "[失敗] 選項 3 需要 OpenAI Python SDK，請先執行："
            "pip install openai",
            file=sys.stderr,
        )
        return 1

    print(f"目標 API：{OPENAI_BASE_URL}")
    print(f"指定模型：{OPENAI_MODEL}")
    client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        timeout=timeout,
        max_retries=0,
    )

    try:
        print("[1/2] 呼叫 /v1/models，測試 OpenAI-compatible API...")
        response = client.models.list()
        model_ids = [item.id for item in response.data]
        print("  API 連線成功。")

        if model_ids:
            print("  主機提供的模型：")
            for model_id in model_ids:
                print(f"    - {model_id}")

        if model_ids and OPENAI_MODEL not in model_ids:
            print(
                f"[失敗] 模型清單中找不到：{OPENAI_MODEL}",
                file=sys.stderr,
            )
            return 1

        if image_path is None:
            print("\n[成功] OpenAI-compatible API 連線正常；未執行圖片推論。")
            return 0

        print(f"[2/2] 執行單張圖片 VLM 推論：{image_path}")
        started_at = time.perf_counter()
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "請用繁體中文簡短描述圖片中的人物、"
                                "動作與環境；若沒有看見人物，請明確說明。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_to_data_url(image_path),
                            },
                        },
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=200,
        )
        elapsed = time.perf_counter() - started_at
        content = (
            completion.choices[0].message.content
            if completion.choices
            else None
        )
        if not content:
            print("[失敗] 模型沒有回傳文字內容。", file=sys.stderr)
            return 1

        print(f"  推論時間：{elapsed:.2f} 秒")
        print(f"  VLM 回覆：{content.strip()}")
        print("\n[成功] OpenAI-compatible 單張圖片推論完成。")
        return 0

    except APIConnectionError as error:
        print(
            f"[失敗] 無法連線到 {OPENAI_BASE_URL}：{error}",
            file=sys.stderr,
        )
    except APITimeoutError:
        print(f"[失敗] API 請求超過 {timeout} 秒。", file=sys.stderr)
    except APIStatusError as error:
        detail = getattr(error.response, "text", str(error))
        print(
            f"[失敗] API 回傳 HTTP {error.status_code}：{detail[:1000]}",
            file=sys.stderr,
        )
    except FileNotFoundError as error:
        print(f"[失敗] {error}", file=sys.stderr)
    except (IndexError, ValueError) as error:
        print(f"[失敗] API 回應格式錯誤：{error}", file=sys.stderr)

    return 1


def extract_host(address: str) -> str:
    """接受 IP、主機名稱或 URL，取出其中的主機部分。"""
    address = address.strip()
    if not address:
        raise ValueError("遠端位址不可為空")

    parsed = urlparse(address if "://" in address else f"//{address}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"無效的遠端位址：{address}")
    return host


def build_ping_command(host: str, count: int, timeout: float) -> list[str]:
    """依目前作業系統建立 Ping 指令，不透過 shell 執行。"""
    if platform.system() == "Windows":
        return [
            "ping", "-n", str(count),
            "-w", str(max(1, int(timeout * 1000))), host,
        ]

    return [
        "ping", "-c", str(count),
        "-W", str(max(1, int(timeout))), host,
    ]


def test_image_inference(
    base_url: str,
    image_path: str,
    model: str,
    timeout: float,
) -> int:
    path = Path(image_path)
    if not path.is_file():
        print(f"[失敗] 找不到測試圖片：{path}", file=sys.stderr)
        return 1

    print(f"[4/4] 執行單張圖片 VLM 推論：{path}")
    encoded_image = base64.b64encode(path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": (
                    "請用繁體中文簡短描述這張圖片中的人物、動作與環境；"
                    "如果沒有看見人物，也請明確說明。"
                ),
                "images": [encoded_image],
            }
        ],
        "options": {
            "temperature": 0.1,
            "num_predict": 200,
        },
        "keep_alive": "5m",
    }

    started_at = time.perf_counter()
    try:
        response = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=(5, timeout),
        )
        response.raise_for_status()
        response_data = response.json()
        content = response_data.get("message", {}).get("content", "").strip()
    except requests.ConnectionError:
        print("[失敗] 圖片推論期間與遠端 Ollama 失去連線。", file=sys.stderr)
        return 1
    except requests.Timeout:
        print(f"[失敗] 圖片推論超過 {timeout} 秒。", file=sys.stderr)
        return 1
    except requests.HTTPError as error:
        response = error.response
        status = response.status_code if response is not None else "未知"
        detail = response.text[:500] if response is not None else str(error)
        print(f"[失敗] 圖片推論 HTTP {status}：{detail}", file=sys.stderr)
        return 1
    except (requests.JSONDecodeError, ValueError) as error:
        print(f"[失敗] 圖片推論回應格式錯誤：{error}", file=sys.stderr)
        return 1

    if not content:
        print("[失敗] 遠端模型沒有回傳文字內容。", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - started_at
    print(f"  推論時間：{elapsed:.2f} 秒")
    print(f"  VLM 回覆：{content}")
    print("\n[成功] 遠端單張圖片 VLM 推論完成。")
    return 0


def test_ollama(
    host: str,
    port: int,
    timeout: float,
    image_path: str | None,
    model: str,
    inference_timeout: float,
) -> int:
    base_url = f"http://{host}:{port}"
    print(f"[3/4] 連線 Ollama：{base_url}")

    try:
        version_response = requests.get(
            f"{base_url}/api/version",
            timeout=timeout,
        )
        version_response.raise_for_status()
        version = version_response.json().get("version", "未知")

        tags_response = requests.get(
            f"{base_url}/api/tags",
            timeout=timeout,
        )
        tags_response.raise_for_status()
        models = [
            model.get("name", "未知")
            for model in tags_response.json().get("models", [])
        ]
    except requests.ConnectionError:
        print(
            f"[失敗] 無法連線到 {base_url}。請確認 Ollama 已監聽外部網路，"
            "且防火牆允許 TCP 11434。",
            file=sys.stderr,
        )
        return 1
    except requests.Timeout:
        print(f"[失敗] 連線 Ollama 超過 {timeout} 秒。", file=sys.stderr)
        return 1
    except requests.HTTPError as error:
        response = error.response
        status = response.status_code if response is not None else "未知"
        detail = response.text[:300] if response is not None else str(error)
        print(f"[失敗] Ollama HTTP {status}：{detail}", file=sys.stderr)
        return 1
    except (requests.JSONDecodeError, ValueError) as error:
        print(f"[失敗] Ollama 回應不是有效 JSON：{error}", file=sys.stderr)
        return 1

    print(f"  Ollama 版本：{version}")
    if models:
        print("  遠端模型：")
        for model in models:
            print(f"    - {model}")
    else:
        print("  遠端目前沒有模型。")

    if image_path is None:
        print("\n[成功] 已連線到遠端 Ollama；未指定圖片，不執行推論。")
        return 0

    if model not in models:
        print(f"[失敗] 遠端找不到指定模型：{model}", file=sys.stderr)
        return 1

    return test_image_inference(
        base_url,
        image_path,
        model,
        inference_timeout,
    )


def test_connection(
    address: str,
    count: int,
    ping_timeout: float,
    port: int,
    ollama_timeout: float,
    image_path: str | None,
    model: str,
    inference_timeout: float,
) -> int:
    host = extract_host(address)

    print(f"目標主機：{host}")
    print("[1/4] 解析主機位址...")
    try:
        resolved_ip = socket.gethostbyname(host)
    except socket.gaierror as error:
        print(f"[失敗] 無法解析主機名稱：{error}", file=sys.stderr)
        return 1

    print(f"  IP 位址：{resolved_ip}")
    print(f"[2/4] 傳送 {count} 次 Ping...")

    try:
        result = subprocess.run(
            build_ping_command(host, count, ping_timeout),
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
        )
    except FileNotFoundError:
        print("[失敗] 系統找不到 ping 指令。", file=sys.stderr)
        return 1

    if result.stdout.strip():
        print(result.stdout.strip())

    if result.returncode == 0:
        print("\n[成功] 本機可以連線到遠端電腦。")
    else:
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        print(
            "\n[失敗] 無法透過 Ping 連線到遠端電腦。"
            "請檢查 IP、網路連線或遠端防火牆的 ICMP 規則。",
            file=sys.stderr,
        )
        return 1

    return test_ollama(
        host,
        port,
        ollama_timeout,
        image_path,
        model,
        inference_timeout,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="測試遠端 Ollama 或 OpenAI-compatible VLM API。"
    )
    parser.add_argument(
        "target", nargs="?",
        help=(
            "Ollama 選擇：1=26.184.142.137、2=26.247.236.14、"
            "3=經 26.184.142.137:9000 轉送至 78B；"
            "也可直接輸入自訂 Ollama IP。省略時顯示選單"
        ),
    )
    parser.add_argument(
        "--count", type=int, default=4,
        help="Ping 次數（預設：4）",
    )
    parser.add_argument(
        "--timeout", type=float, default=2.0,
        help="每次 Ping 的逾時秒數（預設：2）",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_OLLAMA_PORT,
        help=f"Ollama 連接埠（預設：{DEFAULT_OLLAMA_PORT}）",
    )
    parser.add_argument(
        "--ollama-timeout", type=float, default=5.0,
        help="Ollama HTTP 連線逾時秒數（預設：5）",
    )
    parser.add_argument(
        "--image",
        help="要送到遠端 VLM 的單張圖片路徑；省略時只測試連線",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"圖片推論模型（預設：{DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--inference-timeout", type=float, default=180.0,
        help="圖片推論逾時秒數（預設：180）",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.count < 1:
        print("[失敗] --count 必須大於 0。", file=sys.stderr)
        return 1
    if args.timeout <= 0:
        print("[失敗] --timeout 必須大於 0。", file=sys.stderr)
        return 1
    if not 1 <= args.port <= 65535:
        print("[失敗] --port 必須介於 1 到 65535。", file=sys.stderr)
        return 1
    if args.ollama_timeout <= 0:
        print("[失敗] --ollama-timeout 必須大於 0。", file=sys.stderr)
        return 1
    if args.inference_timeout <= 0:
        print("[失敗] --inference-timeout 必須大於 0。", file=sys.stderr)
        return 1

    try:
        selection = args.target if args.target is not None else prompt_target_selection()
        selection = selection.strip()

        if selection == "3":
            return test_openai_compatible(
                image_path=args.image,
                timeout=args.inference_timeout,
            )

        targets = select_targets(selection)

        results = []
        for index, address in enumerate(targets, start=1):
            if len(targets) > 1:
                print(f"\n{'=' * 60}")
                print(f"測試第 {index}/{len(targets)} 台 Ollama：{address}")
                print(f"{'=' * 60}")

            result = test_connection(
                address,
                args.count,
                args.timeout,
                args.port,
                args.ollama_timeout,
                args.image,
                args.model,
                args.inference_timeout,
            )
            results.append((address, result))

        if len(results) > 1:
            print("\n兩台 Ollama 測試結果：")
            for address, result in results:
                status = "成功" if result == 0 else "失敗"
                print(f"  {address}：{status}")

        return 0 if all(result == 0 for _, result in results) else 1

    except (EOFError, KeyboardInterrupt):
        print("\n[取消] 未選擇 Ollama 主機。", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"[失敗] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
