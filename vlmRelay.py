"""PC-lab 上使用的 OpenAI-compatible VLM 轉送服務。

連線路徑：
本機 -> http://26.184.142.137:9000/v1 -> PC-lab
     -> http://192.168.50.51:8001/v1 -> 78B 模型主機

啟動方式（在 PC-lab 執行）：
    python vlmRelay.py
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import requests


RELAY_HOST = "0.0.0.0"
RELAY_PORT = 9000
RELAY_TOKEN = "EMPTY"

UPSTREAM_BASE_URL = "http://192.168.50.51:8001/v1"
UPSTREAM_API_KEY = "EMPTY"
UPSTREAM_CONNECT_TIMEOUT = 10
UPSTREAM_READ_TIMEOUT = 600
MAX_REQUEST_BYTES = 100 * 1024 * 1024

ALLOWED_PATHS = {
    ("GET", "/v1/models"),
    ("POST", "/v1/chat/completions"),
}


class VLMRelayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "VLMRelay/1.0"

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        content_type: str = "application/json; charset=utf-8",
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, body)

    def _authorized(self) -> bool:
        expected = f"Bearer {RELAY_TOKEN}"
        return self.headers.get("Authorization", "") == expected

    def _forward(self) -> None:
        path = urlsplit(self.path).path

        if path == "/health" and self.command == "GET":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "upstream": UPSTREAM_BASE_URL,
                },
            )
            return

        if not self._authorized():
            self._send_json(401, {"error": "invalid relay token"})
            return

        if (self.command, path) not in ALLOWED_PATHS:
            self._send_json(404, {"error": "relay path not allowed"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": "request body is too large"})
            return

        request_body = self.rfile.read(content_length) if content_length else None
        upstream_url = f"{UPSTREAM_BASE_URL}{path.removeprefix('/v1')}"
        upstream_headers = {
            "Authorization": f"Bearer {UPSTREAM_API_KEY}",
            "Accept": "application/json",
        }
        if self.headers.get("Content-Type"):
            upstream_headers["Content-Type"] = self.headers["Content-Type"]

        try:
            response = requests.request(
                method=self.command,
                url=upstream_url,
                headers=upstream_headers,
                data=request_body,
                timeout=(UPSTREAM_CONNECT_TIMEOUT, UPSTREAM_READ_TIMEOUT),
            )
        except requests.ConnectionError as error:
            self._send_json(
                502,
                {
                    "error": "PC-lab 無法連線大型主機",
                    "upstream": UPSTREAM_BASE_URL,
                    "detail": str(error),
                },
            )
            return
        except requests.Timeout as error:
            self._send_json(
                504,
                {
                    "error": "大型主機回應逾時",
                    "detail": str(error),
                },
            )
            return

        content_type = response.headers.get(
            "Content-Type",
            "application/json; charset=utf-8",
        )
        self._send_bytes(response.status_code, response.content, content_type)

    def do_GET(self) -> None:
        self._forward()

    def do_POST(self) -> None:
        self._forward()

    def log_message(self, format: str, *args) -> None:
        client_ip = self.client_address[0]
        print(f"[{client_ip}] {format % args}")


def main() -> int:
    server = ThreadingHTTPServer((RELAY_HOST, RELAY_PORT), VLMRelayHandler)
    print(f"VLM relay 已啟動：http://{RELAY_HOST}:{RELAY_PORT}")
    print(f"轉送目標：{UPSTREAM_BASE_URL}")
    print("按 Ctrl+C 停止。")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止 VLM relay...")
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
