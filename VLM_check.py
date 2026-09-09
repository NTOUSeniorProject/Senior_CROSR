import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

import requests



# ===== VLM 連線設定（直接在此處修改，不從 .env 讀取）=====
OLLAMA_BASE_URL = "http://26.184.142.137:11434"
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_MODEL = "blaifa/InternVL3_5:8B"

# 本機 -> PC-lab relay -> 78B 大型主機
VLM_78B_BASE_URL = "http://26.184.142.137:9000/v1"
VLM_78B_CHAT_URL = f"{VLM_78B_BASE_URL}/chat/completions"
VLM_78B_MODEL = "OpenGVLab/InternVL3-78B-AWQ"
VLM_78B_API_KEY = "EMPTY"

# 0：只使用第一組 VLM
# 1：第一組判定無異常時，再將相同資料送至第二組 VLM
DOUBLE_VLM = 1

# 1：78B 放在第一層，Ollama 4B 放在第二層
# 2：Ollama 4B 放在第一層，78B 放在第二層
VLM_78B_LAYER = 2

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

請只輸出一個合法 JSON 物件，不要加入 Markdown 或其他文字：
{
  "is_abnormal": true 或 false,
  "category": "fall、collapse、fighting、prolonged_lying、unsafe_climbing、distress、normal、uncertain 其中之一",
  "confidence": 0 到 1,
  "description": "繁體中文描述",
  "need_alert": true 或 false
}
"""


def image_to_base64(image_path: str) -> str:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"找不到圖片：{image_path}")

    with path.open("rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def _image_data_url(image_path: str, encoded_image: str) -> str:
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    return f"data:{mime_type};base64,{encoded_image}"


def _parse_result_json(raw_content: str, source_name: str) -> dict[str, Any]:
    """解析模型 JSON；同時容忍模型額外包上的 Markdown code fence。"""
    content = raw_content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                pass
        raise RuntimeError(
            f"{source_name} 回傳的內容不是合法 JSON：{raw_content}"
        )


def _format_result(
    result: dict[str, Any],
    vlm_group: int,
    backend: str,
    model: str,
    **metadata: Any,
) -> dict[str, Any]:
    return {
        "is_abnormal": bool(result["is_abnormal"]),
        "category": str(result["category"]),
        "confidence": float(result["confidence"]),
        "description": str(result["description"]),
        "need_alert": bool(result["need_alert"]),
        "vlm_group": vlm_group,
        "vlm_backend": backend,
        "vlm_model": model,
        **metadata,
    }


def _request_ollama(
    url: str,
    base_url: str,
    payload: dict[str, Any],
    timeout: int,
    vlm_group: int,
) -> dict[str, Any]:
    """送出一次 Ollama VLM 請求，並整理成系統使用的結果格式。"""
    try:
        response = requests.post(
            url,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

    except requests.ConnectionError as error:
        raise RuntimeError(
            f"無法連線到第 {vlm_group} 組 Ollama，請確認 Ollama 已啟動，"
            f"並可存取 {base_url}"
        ) from error

    except requests.Timeout as error:
        raise RuntimeError(
            f"第 {vlm_group} 組 Ollama 推論超過 {timeout} 秒"
        ) from error

    except requests.HTTPError as error:
        raise RuntimeError(
            f"第 {vlm_group} 組 Ollama API 發生錯誤："
            f"{response.status_code} {response.text}"
        ) from error

    response_data = response.json()
    raw_content = response_data["message"]["content"]

    result = _parse_result_json(raw_content, f"第 {vlm_group} 組 Ollama")

    return _format_result(
        result=result,
        vlm_group=vlm_group,
        backend="ollama",
        model=str(payload["model"]),
        total_duration=response_data.get("total_duration"),
        load_duration=response_data.get("load_duration"),
        eval_count=response_data.get("eval_count"),
    )


def _request_openai_compatible(
    payload: dict[str, Any],
    timeout: int,
    vlm_group: int,
) -> dict[str, Any]:
    """經 PC-lab relay 呼叫 OpenAI-compatible 78B VLM。"""
    try:
        response = requests.post(
            VLM_78B_CHAT_URL,
            headers={
                "Authorization": f"Bearer {VLM_78B_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.ConnectionError as error:
        raise RuntimeError(
            f"無法連線到第 {vlm_group} 層 78B VLM，請確認 PC-lab 的 "
            f"vlmRelay.py 已啟動，並可存取 {VLM_78B_BASE_URL}"
        ) from error
    except requests.Timeout as error:
        raise RuntimeError(
            f"第 {vlm_group} 層 78B VLM 推論超過 {timeout} 秒"
        ) from error
    except requests.HTTPError as error:
        raise RuntimeError(
            f"第 {vlm_group} 層 78B VLM API 發生錯誤："
            f"{response.status_code} {response.text}"
        ) from error

    response_data = response.json()
    try:
        raw_content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            f"第 {vlm_group} 層 78B VLM 回傳格式不正確：{response_data}"
        ) from error

    result = _parse_result_json(raw_content, f"第 {vlm_group} 層 78B VLM")
    usage = response_data.get("usage") or {}
    return _format_result(
        result=result,
        vlm_group=vlm_group,
        backend="openai-compatible",
        model=VLM_78B_MODEL,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )


def _build_ollama_payload(
    encoded_images: list[str],
    model: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "stream": False,
        "format": ABNORMAL_RESULT_SCHEMA,
        "messages": [
            {
                "role": "user",
                "content": VLM_PROMPT,
                "images": encoded_images,
            }
        ],
        "options": {
            "temperature": 0.1,
            "num_predict": 300,
            "num_ctx": 32768,
        },
        "keep_alive": "5m",
    }


def _build_openai_payload(
    frame_paths: list[str],
    encoded_images: list[str],
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": VLM_PROMPT}
    ]
    content.extend(
        {
            "type": "image_url",
            "image_url": {
                "url": _image_data_url(frame_path, encoded_image),
            },
        }
        for frame_path, encoded_image in zip(frame_paths, encoded_images)
    )
    return {
        "model": VLM_78B_MODEL,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1,
        "max_tokens": 300,
        "stream": False,
    }


def _request_layer(
    layer: int,
    use_78b: bool,
    frame_paths: list[str],
    encoded_images: list[str],
    ollama_model: str,
    timeout: int,
) -> dict[str, Any]:
    if use_78b:
        return _request_openai_compatible(
            payload=_build_openai_payload(frame_paths, encoded_images),
            timeout=timeout,
            vlm_group=layer,
        )

    return _request_ollama(
        url=OLLAMA_URL,
        base_url=OLLAMA_BASE_URL,
        payload=_build_ollama_payload(encoded_images, ollama_model),
        timeout=timeout,
        vlm_group=layer,
    )


def analyze_frames_with_ollama(
    frame_paths: list[str],
    model: str = OLLAMA_MODEL,
    timeout: int = 180,
    double_vlm: int = DOUBLE_VLM,
    vlm_78b_layer: int = VLM_78B_LAYER,
) -> dict[str, Any]:
    if not frame_paths:
        raise ValueError("沒有提供任何影格")
    if double_vlm not in (0, 1):
        raise ValueError("double_vlm 只能是 0 或 1")
    if vlm_78b_layer not in (1, 2):
        raise ValueError("vlm_78b_layer 只能是 1 或 2")

    encoded_images = [
        image_to_base64(frame_path)
        for frame_path in frame_paths
    ]

    first_is_78b = vlm_78b_layer == 1
    primary_result = _request_layer(
        layer=1,
        use_78b=first_is_78b,
        frame_paths=frame_paths,
        encoded_images=encoded_images,
        ollama_model=model,
        timeout=timeout,
    )
    primary_result["double_vlm_triggered"] = False

    if double_vlm == 1 and not primary_result["is_abnormal"]:
        print(
            "[double_vlm] 第 1 層 VLM 判定無異常，"
            "將相同的圖片與提示詞送往第 2 層 VLM。"
        )
        secondary_result = _request_layer(
            layer=2,
            use_78b=not first_is_78b,
            frame_paths=frame_paths,
            encoded_images=encoded_images,
            ollama_model=model,
            timeout=timeout,
        )
        secondary_result["double_vlm_triggered"] = True
        return secondary_result

    return primary_result
