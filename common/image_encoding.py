"""VLM 正式分析與連線測試共用的圖片編碼工具。"""

import base64
import mimetypes
from pathlib import Path
from constants import project_path


def image_to_base64(image_path: str | Path) -> str:
    path = project_path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到圖片：{path}")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def image_to_data_url(image_path: str | Path, encoded_image: str | None = None) -> str:
    """已有 Base64 時直接組合 URL，避免重複讀檔與編碼。"""
    if encoded_image is None:
        encoded_image = image_to_base64(image_path)
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    return f"data:{mime_type};base64,{encoded_image}"
