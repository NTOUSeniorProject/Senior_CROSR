# 系統與純 VLM 成本比較

## 使用現有 .env（建議）

兩個 VLM 入口會讀取專案根目錄 `.env`；PowerShell 已設定的環境變數優先。
可將以下設定放入 `.env`（影片路徑請換成自己的本機影片）：

```dotenv
VIDEO_PATH='C:\影片\test.mp4'
VLM_EXPERIMENT_MODE=1
VLM_EXPERIMENT_LOG=output/system_vs_vlm/requests.jsonl
```

`VIDEO_PATH` 會成為 `constants.CONFIG['video_path']`；沒設定時保留原本預設來源。
執行 `python alarm_YT.py` 即可記錄系統組；執行
`python systemVsVLM.py video --double-vlm 1 --vlm-78b-layer 2` 使用同一個 video_path 跑純 VLM 組。
純 VLM 組目前限本機影片。命令列提供影片路徑時優先使用命令列。
執行 `python systemVsVLM.py summary` 會自動使用 `.env` 指定的紀錄位置。
不設定 `VLM_EXPERIMENT_RUN_ID` 即可每次啟動自動產生不同 ID。
系統组未設定 `VLM_EXPERIMENT_VIDEO` 時會使用當前 CONFIG 影片來源作為標籤。
以下是實驗設定的 .env 支援；VLM 伺服器位址與模型仍沿用 `VLM_check.py` 常數。

`VLM_check.py` 的 `EXPERIMENT_MODE` 預設關閉。設為 `True` 或在啟動系統前設定
`VLM_EXPERIMENT_MODE=1` 後，所有 VLM HTTP 請求會轉交 `systemVsVLM.request_vlm`，
由它實際呼叫模型、記錄資料，再把原始回應交回既有判斷流程。兩層各計一次。

## 1. 系統組（PowerShell）

先將 `constants.py` 的 `CONFIG['video_path']` 設為要比較的本機影片，並使用有限影片模式。
以下 `VLM_EXPERIMENT_VIDEO` 是紀錄標籤，不會改變系統影片來源，必須與上述設定一致。

```powershell
$env:VLM_EXPERIMENT_MODE = '1'
$env:VLM_EXPERIMENT_RUN_ID = 'video01-system-01'
$env:VLM_EXPERIMENT_VIDEO = (Resolve-Path '.\video01.mp4').Path
$env:VLM_EXPERIMENT_ARM = 'system'
python alarm_YT.py
```

照平常方式執行系統至影片結束。只計算系統實際觸發的 VLM 請求，不含 YOLO/CROSR 運算。
若完全沒有觸發 VLM，就不會產生請求紀錄；不能單憑沒有紀錄判斷影片是否已完整跑完。

## 2. 純 VLM 組

```powershell
$env:VLM_EXPERIMENT_RUN_ID = 'video01-vlm-01'
python systemVsVLM.py video '.\video01.mp4' --clip-seconds 10 --frames-per-clip 8 --double-vlm 1 --vlm-78b-layer 2
```

逐段分析完整本機影片，每 10 秒均勻抽 8 張，最後不足 10 秒也分析。
採用相同提示詞、影像 JPEG 品質與 VLM 呼叫函式。此命令自動啟用紀錄並標記 `vlm_only`。
為控制變因，請將 `--double-vlm`、`--vlm-78b-layer` 設為與系統組相同；
純 VLM 命令預設為單層 Ollama（`--double-vlm 0 --vlm-78b-layer 2`）。
記錄不同模型成本時應同時檢查逐筆 `model`，token 數並不直接等同金額。

## 3. 查看結果

```powershell
python systemVsVLM.py summary
```

預設追加寫入 `output/system_vs_vlm/requests.jsonl`，可用環境變數
`VLM_EXPERIMENT_LOG` 改位置；summary 對應使用 `--log 路徑`。
每次試驗使用不同 run ID。未設定時自動產生程序級 ID。
彙總依 run ID、影片及比較組別分組；`known_sum` 為已知值加總，
`missing_requests` 非零表示數據不完整，不能視為完整總成本。

- `prompt_tokens`：Ollama 的 `prompt_eval_count` 或 OpenAI-compatible 的 `usage.prompt_tokens`。
- `completion_tokens`：Ollama 的 `eval_count` 或 `usage.completion_tokens`。
- `total_tokens`：後端提供的總數，或輸入與輸出相加；不額外重複加 reasoning tokens。
- `request_seconds`：本機送出至收到完整回應的時間，包含網路、排隊、載入與推論。
- `generation_seconds`、`prompt_eval_seconds`、`server_total_seconds`、`load_seconds`：Ollama 回報的奈秒換算成秒。
- `thinking_seconds`：目前兩種非串流介面沒有獨立思考時間，保留 `null`；生成或等待時間不能當作純思考時間。
- `reasoning_tokens`：後端若有提供則保存，否則 `null`。

HTTP 錯誤與逾時也會記錄並沿用原流程拋錯。JSON 解析失敗仍保留 HTTP 層已收到的使用量；
`failed_requests` 指傳輸或 HTTP 失敗，不代表模型輸出的業務 JSON 一定有效。
紀錄不包含圖片、提示詞、API key 或完整回應。建議每個輸出檔只由一個程序寫入。

關閉實驗模式：將程式中的常數改回 `False`，或執行 `$env:VLM_EXPERIMENT_MODE = '0'` 後重新啟動。
