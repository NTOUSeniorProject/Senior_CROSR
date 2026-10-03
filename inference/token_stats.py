"""每次 HTTP 回應即記錄；解析模型文字失敗也不遺失 usage。"""
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.Lock()


def count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def record_response(data, layer, backend, model):
    usage = data.get('usage') or {}
    prompt = count(data.get('prompt_eval_count')) if backend == 'ollama' else count(usage.get('prompt_tokens'))
    completion = count(data.get('eval_count')) if backend == 'ollama' else count(usage.get('completion_tokens'))
    total = count(usage.get('total_tokens')) if backend != 'ollama' else None
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    metadata = dict(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)
    log = os.environ.get('VLM_TOKEN_LOG')
    if log:
        row = dict(timestamp=datetime.now(timezone.utc).isoformat(),
                   video=os.environ.get('VLM_VIDEO_ID', ''),
                   method=os.environ.get('VLM_METHOD', ''), layer=layer,
                   backend=backend, model=model, **metadata)
        with _lock:
            path = Path(log)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return metadata


def summary(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()] if Path(path).exists() else []
    errors = [row for row in rows if row.get('record_type') == 'error']
    rows = [row for row in rows if row.get('record_type') != 'error']
    result = {'vlm_calls_with_response': len(rows), 'error_count': len(errors)}
    for field in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        missing = sum(row[field] is None for row in rows)
        result[field] = None if missing else sum(row[field] for row in rows)
        result[field + '_missing_calls'] = missing
    groups = {}
    for row in rows:
        key = row['backend'] + '/' + row['model']
        g = groups.setdefault(key, {'calls': 0, 'total_tokens': 0})
        g['calls'] += 1
        if row['total_tokens'] is None or g['total_tokens'] is None:
            g['total_tokens'] = None
        else:
            g['total_tokens'] += row['total_tokens']
    result['by_model'] = groups
    return result


def record_error(stage, message):
    log = os.environ.get('VLM_TOKEN_LOG')
    if log:
        with _lock:
            with Path(log).open('a', encoding='utf-8') as f:
                f.write(json.dumps({'record_type': 'error', 'stage': stage,
                                    'message': message}, ensure_ascii=False) + '\n')


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def experiment_settings(vlm_timeout=180):
    import hashlib
    import constants
    from inference.VLM_check import VLM_PROMPT, ABNORMAL_RESULT_SCHEMA
    names = ('OLLAMA_MODEL', 'VLM_78B_MODEL', 'DOUBLE_VLM', 'VLM_78B_LAYER',
             'PRE_EVENT_SECONDS', 'POST_EVENT_SECONDS', 'VLM_SAMPLE_FRAME_COUNT')
    settings = {name: getattr(constants, name) for name in names}
    settings['prompt_sha256'] = hashlib.sha256(VLM_PROMPT.encode()).hexdigest()
    settings['schema_sha256'] = hashlib.sha256(json.dumps(ABNORMAL_RESULT_SCHEMA, sort_keys=True).encode()).hexdigest()
    settings.update(temperature=0.1, max_output_tokens=300, ollama_num_ctx=32768,
                    jpeg_quality=85, sampling='linspace_dtype_int', image_resize='none',
                    http_connect_timeout_sec=10, http_read_timeout_sec=vlm_timeout)
    return settings


def begin_run(path, video, method, vlm_timeout=180, video_duration_sec=None):
    """一個程序同時只允許一個實驗；固定 log 與影片身分。"""
    path = Path(path).resolve()
    video = Path(video).resolve()
    if os.environ.get('VLM_TOKEN_LOG'):
        raise RuntimeError('已有 VLM_TOKEN_LOG；請使用獨立程序跑每個實驗，並清除舊環境變數')
    if not video.is_file():
        raise ValueError('實驗需要本機影片檔')
    manifest = Path(str(path) + '.run.json')
    if manifest.exists() or path.exists():
        raise FileExistsError('log 或 run.json 已存在，請使用新檔名')
    payload = dict(method=method, video=str(video), video_sha256=sha256_file(video),
                   video_duration_sec=video_duration_sec,
                   settings=experiment_settings(vlm_timeout), completed=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8'):
        pass
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    os.environ.update(VLM_TOKEN_LOG=str(path), VLM_VIDEO_ID=str(video), VLM_METHOD=method)


def finish_run(completed, **metadata):
    path = os.environ['VLM_TOKEN_LOG']
    manifest = Path(path + '.run.json')
    payload = json.loads(manifest.read_text(encoding='utf-8'))
    report = summary(path)
    payload.update(completed=bool(completed), **metadata, summary=report)
    payload['valid_for_comparison'] = bool(completed and report['error_count'] == 0
                                         and all(report[k] is not None for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')))
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    for key in ('VLM_TOKEN_LOG', 'VLM_VIDEO_ID', 'VLM_METHOD'):
        os.environ.pop(key, None)
    return payload
