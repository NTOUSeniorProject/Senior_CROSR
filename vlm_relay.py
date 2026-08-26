"""PC-lab VLM 中繼站：PC-lab:8002 -> 大型主機:8001/v1。"""

import os

import requests
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request


load_dotenv()

UPSTREAM_BASE_URL = os.getenv(
    "LARGE_VLM_BASE_URL",
    "http://192.168.50.51:8001/v1",
).rstrip("/")
UPSTREAM_API_KEY = os.getenv("LARGE_VLM_API_KEY", "EMPTY")
RELAY_TOKEN = os.getenv("VLM_RELAY_TOKEN", "")
RELAY_HOST = os.getenv("VLM_RELAY_HOST", "0.0.0.0")
RELAY_PORT = int(os.getenv("VLM_RELAY_PORT", "8002"))
CONNECT_TIMEOUT = float(os.getenv("VLM_RELAY_CONNECT_TIMEOUT", "10"))
READ_TIMEOUT = float(os.getenv("VLM_RELAY_READ_TIMEOUT", "300"))

ALLOWED_PATHS = {
    "models": {"GET"},
    "chat/completions": {"POST"},
}

app = Flask(__name__)


def is_authorized() -> bool:
    if not RELAY_TOKEN:
        return True
    return request.headers.get("Authorization", "") == f"Bearer {RELAY_TOKEN}"


def upstream_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if UPSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"
    return headers


def forward(path: str) -> Response:
    if path not in ALLOWED_PATHS or request.method not in ALLOWED_PATHS[path]:
        return jsonify({"error": "不允許的中繼路徑或方法"}), 404
    if not is_authorized():
        return jsonify({"error": "中繼站驗證失敗"}), 401

    try:
        upstream_response = requests.request(
            method=request.method,
            url=f"{UPSTREAM_BASE_URL}/{path}",
            headers=upstream_headers(),
            data=request.get_data() if request.method == "POST" else None,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
    except requests.ConnectionError:
        return jsonify({
            "error": f"PC-lab 無法連線大型主機 {UPSTREAM_BASE_URL}"
        }), 502
    except requests.Timeout:
        return jsonify({"error": "大型主機 VLM 回應逾時"}), 504

    excluded_headers = {
        "content-encoding",
        "content-length",
        "transfer-encoding",
        "connection",
    }
    response_headers = [
        (name, value)
        for name, value in upstream_response.headers.items()
        if name.lower() not in excluded_headers
    ]
    return Response(
        upstream_response.content,
        status=upstream_response.status_code,
        headers=response_headers,
    )


@app.get("/health")
def health():
    try:
        response = requests.get(
            f"{UPSTREAM_BASE_URL}/models",
            headers=upstream_headers(),
            timeout=(CONNECT_TIMEOUT, 30),
        )
        response.raise_for_status()
        models = [
            item.get("id")
            for item in response.json().get("data", [])
            if item.get("id")
        ]
    except (requests.RequestException, ValueError) as error:
        return jsonify({
            "status": "unavailable",
            "upstream": UPSTREAM_BASE_URL,
            "error": str(error),
        }), 503

    return jsonify({
        "status": "ok",
        "upstream": UPSTREAM_BASE_URL,
        "models": models,
    })


@app.route("/v1/<path:path>", methods=["GET", "POST"])
def relay(path: str):
    return forward(path)


if __name__ == "__main__":
    print(f"VLM relay listening on http://{RELAY_HOST}:{RELAY_PORT}")
    print(f"Forwarding to {UPSTREAM_BASE_URL}")
    app.run(host=RELAY_HOST, port=RELAY_PORT, debug=False, threaded=True)
