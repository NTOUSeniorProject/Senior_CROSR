import argparse
import json
from pathlib import Path
from inference.token_stats import summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('baseline', help='run_full_vlm.py 的 --log 檔案路徑')
    p.add_argument('ours', help='run_ours.py 的 --log 檔案路徑')
    args = p.parse_args()
    # 先列出兩組全部缺少的檔案，讓使用者知道是哪個檔名或路徑不符。
    logs = [Path(log).expanduser().resolve() for log in (args.baseline, args.ours)]
    missing = []
    for label, log in zip(('baseline', 'ours'), logs):
        for required in (log, Path(str(log) + '.run.json')):
            if not required.is_file():
                missing.append(f'  {label}: {required}')
    if missing:
        message = '找不到下列檔案：\n' + '\n'.join(missing)
        message += f'\n目前工作目錄：{Path.cwd()}'
        available = []
        for folder in sorted({log.parent for log in logs}):
            if folder.is_dir():
                available.extend(str(path) for path in sorted(folder.glob('*.jsonl')) if path.is_file())
        if available:
            message += '\n這些目錄已有的 log：\n  ' + '\n  '.join(available)
        message += '\n比較參數必須與兩個 runner 的 --log 檔名相同。'
        message += '\n.jsonl.run.json 由 runner 產生；若尚未執行其中一組，請先完成該組測試。'
        p.error(message)
    manifests = []
    for log in logs:
        path = Path(str(log) + '.run.json')
        try:
            run = json.loads(path.read_text(encoding='utf-8'))
            stats = summary(log)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            p.error(f'無法讀取 {log} 或其 run.json：{exc}')
        reasons = []
        if not run.get('completed'):
            reasons.append('影片未完整完成')
        if stats['error_count']:
            reasons.append(f"記錄了 {stats['error_count']} 個錯誤")
        if any(stats[k] is None for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')):
            reasons.append('token usage 缺漏')
        if reasons:
            p.error(f"{log}: {'、'.join(reasons)}，不能計算有效降幅；請檢查該組輸出後重新測試")
        manifests.append(run)
    b_run, o_run = manifests
    if b_run['method'] != 'baseline' or o_run['method'] != 'ours':
        p.error('參數順序必須是 baseline log、ours log')
    if b_run['video_sha256'] != o_run['video_sha256'] or b_run['settings'] != o_run['settings']:
        p.error('影片內容或 VLM/抽幀設定不同，請用相同設定重跑')
    b, o = (summary(log) for log in logs)
    def reduction(base, ours):
        return (base - ours) / base * 100 if base > 0 else None
    print(json.dumps(dict(video_duration_sec=b_run.get('video_duration_sec'), baseline=b, ours=o,
        token_reduction_percent=reduction(b['total_tokens'], o['total_tokens']),
        call_reduction_percent=reduction(b['vlm_calls_with_response'], o['vlm_calls_with_response'])),
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
