import argparse
import json
from pathlib import Path
from inference.token_stats import summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('baseline')
    p.add_argument('ours')
    args = p.parse_args()
    manifests = []
    for log in (args.baseline, args.ours):
        path = Path(log + '.run.json')
        if not Path(log).exists() or not path.exists():
            p.error('缺少 log 或 .run.json；請使用提供的 runner 完整執行')
        run = json.loads(path.read_text(encoding='utf-8'))
        stats = summary(log)
        if not run.get('completed') or stats['error_count'] or any(stats[k] is None for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')):
            p.error('實驗未完整完成、存在錯誤或 usage 缺漏，不能計算有效降幅')
        manifests.append(run)
    b_run, o_run = manifests
    if b_run['method'] != 'baseline' or o_run['method'] != 'ours':
        p.error('參數順序必須是 baseline log、ours log')
    if b_run['video_sha256'] != o_run['video_sha256'] or b_run['settings'] != o_run['settings']:
        p.error('影片內容或 VLM/抽幀設定不同，請用相同設定重跑')
    b, o = summary(args.baseline), summary(args.ours)
    def reduction(base, ours):
        return (base - ours) / base * 100 if base > 0 else None
    print(json.dumps(dict(baseline=b, ours=o,
        token_reduction_percent=reduction(b['total_tokens'], o['total_tokens']),
        call_reduction_percent=reduction(b['vlm_calls_with_response'], o['vlm_calls_with_response'])),
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
