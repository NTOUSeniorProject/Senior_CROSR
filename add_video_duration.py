"""補上既有 run.json 的影片長度，讀取原影片資料即可，不呼叫 VLM。"""
import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import cv2


def update_duration(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    video = Path(data['video'])
    if not video.is_file():
        raise FileNotFoundError(f'找不到原影片：{video}')
    if data.get('video_sha256'):
        digest = hashlib.sha256()
        with video.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != data['video_sha256']:
            raise ValueError(f'原影片內容與實驗紀錄不符：{video}')
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f'無法開啟影片：{video}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(frames) or frames <= 0:
            raise ValueError(f'無法取得有效 FPS 或總影格數：{video}')
        data['video_duration_sec'] = frames / fps
    finally:
        cap.release()
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=path.parent, delete=False) as f:
            temporary_path = Path(f.name)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write('\n')
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    print(f"已更新 {path.name}: video_duration_sec={data['video_duration_sec']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', help='一個或多個 .jsonl.run.json 檔案')
    args = parser.parse_args()
    for path in args.files:
        try:
            update_duration(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f'{path}: {exc}')


if __name__ == '__main__':
    main()
