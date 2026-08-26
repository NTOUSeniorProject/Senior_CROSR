"""測試本機是否能連線到遠端電腦及其 Ollama 服務。

未指定圖片時只檢查連線；使用 --image 時會執行一次單張圖片推論。
"""

import argparse
import base64
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


DEFAULT_ADDRESS = "26.184.142.137"
DEFAULT_OLLAMA_PORT = 11434
DEFAULT_MODEL = "blaifa/InternVL3_5:8B"


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
        description="測試遠端 Ollama，並可選擇執行單張圖片 VLM 推論。"
    )
    parser.add_argument(
        "address", nargs="?", default=DEFAULT_ADDRESS,
        help=f"遠端 IP 或主機名稱（預設：{DEFAULT_ADDRESS}）",
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
        return test_connection(
            args.address,
            args.count,
            args.timeout,
            args.port,
            args.ollama_timeout,
            args.image,
            args.model,
            args.inference_timeout,
        )
    except ValueError as error:
        print(f"[失敗] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
