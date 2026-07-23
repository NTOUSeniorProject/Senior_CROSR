import os
import re
import sys
import cv2
import shutil
import subprocess
from constants import CONFIG


def is_url(source):
    if not isinstance(source, str):
        return False
    return source.startswith(("http://", "https://", "rtsp://", "rtmp://"))


def is_youtube_url(source):
    if not isinstance(source, str):
        return False

    youtube_patterns = [
        r"youtube\.com",
        r"youtu\.be",
        r"youtube\.com/live",
        r"youtube\.com/watch",
        r"youtube\.com/shorts",
    ]

    return any(re.search(pattern, source, re.IGNORECASE) for pattern in youtube_patterns)


def is_direct_stream_url(source):
    if not isinstance(source, str):
        return False

    lower_source = source.lower()
    direct_keywords = [
        ".m3u8",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".flv",
        ".webm",
        "rtsp://",
        "rtmp://",
    ]

    return any(keyword in lower_source for keyword in direct_keywords)


def check_ytdlp_installed():
    ytdlp_path = shutil.which("yt-dlp")
    if ytdlp_path is None:
        raise RuntimeError(
            "找不到 yt-dlp。\n"
            "請先在你的環境安裝：\n"
            "pip install yt-dlp\n"
            "或使用：\n"
            "python -m pip install yt-dlp"
        )
    return ytdlp_path


def resolve_youtube_stream_url(youtube_url):
    """
    使用 yt-dlp 把 YouTube 連結解析成真正可讀的串流 URL。
    """
    check_ytdlp_installed()

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "-f", CONFIG["youtube_format"],
        "-g",
        youtube_url
    ]

    if CONFIG["use_browser_cookies"]:
        cmd.extend([
            "--cookies-from-browser",
            CONFIG["cookies_browser"]
        ])

    print("\n============================================================")
    print("🔗 偵測到 YouTube 連結，正在解析串流網址...")
    print(f"YouTube URL: {youtube_url}")
    print("============================================================")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="ignore"
        )

        urls = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip().startswith(("http://", "https://"))
        ]

        if len(urls) == 0:
            raise RuntimeError(
                "yt-dlp 沒有輸出可用串流網址。\n"
                f"stderr:\n{result.stderr}"
            )

        # 優先選 m3u8
        for url in urls:
            if ".m3u8" in url.lower():
                print("✅ 已取得 YouTube HLS / m3u8 串流網址")
                return url

        print("✅ 已取得 YouTube 直連影片網址")
        return urls[0]

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        raise RuntimeError(
            "yt-dlp 解析 YouTube 失敗。\n"
            "可能原因：\n"
            "1. yt-dlp 版本太舊\n"
            "2. YouTube 影片需要登入 / cookies\n"
            "3. 影片是私人影片或地區限制\n"
            "4. 網路連線問題\n\n"
            "你可以先嘗試更新：\n"
            "pip install -U yt-dlp\n\n"
            "如果需要 cookies，請把 CONFIG['use_browser_cookies'] 改成 True。\n\n"
            f"錯誤內容：\n{error_msg}"
        )


def resolve_video_source(source):
    """
    將輸入來源統一轉成 OpenCV 可以讀的來源。
    支援：
    1. 本機檔案
    2. YouTube 連結
    3. m3u8 / rtsp / rtmp / http 影片串流
    """

    if source is None or str(source).strip() == "":
        raise ValueError("CONFIG['video_path'] 是空的，請填入本機影片路徑或 YouTube 連結。")

    source = str(source).strip()

    # 1. 本機檔案存在
    if os.path.exists(source):
        print(f"✅ 使用本機影片來源：{source}")
        return source

    # 2. YouTube URL
    if CONFIG["enable_youtube_url"] and is_youtube_url(source):
        return resolve_youtube_stream_url(source)

    # 3. 直接串流 URL
    if is_url(source) and is_direct_stream_url(source):
        print(f"✅ 使用直接串流來源：{source}")
        return source

    # 4. 其他 URL
    if is_url(source):
        print("⚠️ 偵測到一般 URL，將直接交給 OpenCV 嘗試開啟。")
        return source

    # 5. 不是 URL，也不是存在的本機檔案
    raise FileNotFoundError(
        f"找不到影片來源：{source}\n"
        "請確認：\n"
        "1. 本機影片路徑是否正確\n"
        "2. YouTube 連結是否完整\n"
        "3. 如果是 Windows 路徑，請使用 r\"C:\\路徑\\影片.mp4\""
    )


def open_video_capture(source):
    """
    統一建立 OpenCV VideoCapture。
    支援本機影片、YouTube、HTTP、RTSP 與 RTMP。
    """

    resolved_source = resolve_video_source(source)

    is_rtsp = (
        isinstance(resolved_source, str)
        and resolved_source.lower().startswith("rtsp://")
    )

    if CONFIG["use_ffmpeg_backend"]:
        cap = cv2.VideoCapture(
            resolved_source,
            cv2.CAP_FFMPEG
        )
    else:
        cap = cv2.VideoCapture(resolved_source)

    # 降低 RTSP 延遲
    if is_rtsp:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise FileNotFoundError(
            "OpenCV 無法開啟影片來源。\n"
            f"原始來源：{source}\n"
            f"解析後來源：{resolved_source}\n\n"
            "請確認：\n"
            "1. iPhone 與電腦連接同一個 Wi-Fi\n"
            "2. OctoStream 已開始串流\n"
            "3. RTSP 網址完整且正確\n"
            "4. Windows 防火牆沒有阻擋 Python\n"
            "5. VLC 可以正常播放此 RTSP 網址"
        )

    print(f"✅ OpenCV 已成功開啟：{resolved_source}")

    return cap, resolved_source
