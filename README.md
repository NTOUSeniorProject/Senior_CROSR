# 直接套用到現有專案的 VLM token 實驗版

## 放置檔案

將本套件的 inference/ 中檔案合併到原專案的 inference/ 目錄。
VLM_check.py、event_handler.py、real_time_detector.py 已修改；token_stats.py 是新增。
其餘五個上傳的 .py 檔保留原內容，附在套件方便對照。
保留原目錄內未附的檔案與 __init__.py，不要刪除整個 inference 目錄。
將 run_full_vlm.py、run_ours.py、compare_tokens.py 放在與 constants.py 同層的專案根目錄。
沿用現有 constants.py、common/、模型權重、雷達參數及套件環境。
這些相依資料未包含在你的上傳附件中，本套件不是可獨立執行的完整專案。

## 執行順序

請在原專案根目錄執行，Windows 命令提示字元與 PowerShell 都可以：

```bash
python run_full_vlm.py "D:/videos/fall01.mp4" --log baseline_fall01.jsonl
python run_ours.py "D:/videos/fall01.mp4" --yolo "D:/models/your_pose_model.pt" --log ours_fall01.jsonl
python compare_tokens.py baseline_fall01.jsonl ours_fall01.jsonl
```

### 78B 回應逾時的更新

run_full_vlm.py 和 run_ours.py 新增 --timeout，預設讀取逾時為 600 秒，
HTTP 連線建立逾時為 10 秒。每層 backend 分別使用這個設定。
兩組須使用同一設定，並會寫入 run.json 供 compare_tokens 驗證。
HTTP 讀取逾時指連續未讀到 socket 資料的時間，不是整支影片的執行時限。
既有直接呼叫 analyze_frames_with_ollama 的預設讀取逾時仍為 180 秒；
若使用自己的主程式，可傳 timeout=600，或在 play_and_live_inference 傳 vlm_timeout=600。

例如針對這次失敗的影片，換一個新的 log 從頭跑：

```bash
python run_full_vlm.py "C:/linebot/Senior_CROSR/test_video/test_video1.mp4" --log baseline_test_video1_600.jsonl --timeout 600
```

baseline 每窗呼叫前也會印出影片時間範圍、圖片張數和讀取逾時設定。
如果讀取逾時仍發生，請對照 PC-lab relay 與模型伺服器同時間的紀錄，
確認是否仍在推論、排隊、模型出錯或連線中斷；僅靠客戶端 traceback 無法區分。
若 relay 對上游也設置較短的等待時限，需一併檢查其設定。
這版不自動重送逾時請求；逾時時伺服器端可能已執行推論，不能當作消耗為零。
失敗 run 的累計仍只涵蓋已收到 usage 的回應，不可直接與完整 ours 比較。

更換影片與 YOLO 權重的實際路徑。
CROSR 權重和 radar_meta_path 沿用 constants.CONFIG；runner 不需要你重新傳這些參數。
YOLO 請選你現在使用的 pose 權重，不是一般物件偵測權重。
若指定裝置可加 --device cuda:0 或 --device cpu。
run_ours 預設關閉播放視窗並且不傳 LINE 訊息；加 --show 可顯示視窗。
同一個 log 檔名只允許建立一次。重新實驗請改檔名，避免累加不同次執行。
每個 runner 執行一次是一支影片，一個獨立程序。

## 實驗實際比較什麼

baseline：把整支影片按不重疊的時間窗分段，每窗均勻抽取 VLM_SAMPLE_FRAME_COUNT 張，
直接送入 VLM，最後不足一窗仍處理。
預設窗長 = PRE_EVENT_SECONDS + POST_EVENT_SECONDS，可用 --window-sec 修改。
ours：沿用 Movement / YOLO / CROSR / 候選確認 / 後段保留條件，
只有原系統保留下來的事件才呼叫 VLM。
兩邊共用 sample_vlm_frames，所以 linspace 抽幀方式、原解析度、JPEG 品質 85 相同。
兩邊使用同一 VLM 模組、prompt、模型、生成設定和 DOUBLE_VLM 條件。
這是「分段抽幀覆蓋完整影片」的 baseline，不是單次把完整影片輸入模型，也不是送入每一幀。
事件時間窗仍可能重疊；對應的 VLM 重複消耗會照實計算。
末尾短窗或短事件影格不足時不重複补幀。

### 排程時間

你的原程式 startup 與定期異常偵測用 time.monotonic()，离線處理速度可能影響前置篩選。
run_ours 預設用影片秒數安排這兩個排程，以免 GPU 速度、顯示視窗影響排程。
只改排程用的 clock，原異常分數、投票、候選與保留門檻照原設定。
若要量測原本牆鐘版本，可加 --wall-clock；報告需說明時間基準和播放方式。
既有 play_and_live_inference 呼叫預設 experiment_media_clock=False，仍沿用原本牆鐘排程。
實驗請使用同一份本機固定幀率影片；直播的丟幀與重連無法保證兩組分析相同內容。
可變幀率影片需要先轉為固定幀率或另改為依媒體時間戳分窗。

## 輸出

每次 backend 成功回傳 HTTP JSON，便立即記錄到 .jsonl。
其中含 video、method、layer、backend、model、prompt_tokens、completion_tokens、total_tokens。
Ollama 對應 prompt_eval_count、eval_count；compatible API 對應 usage。
DOUBLE_VLM 的兩層分別記錄，不會只算最終回傳的第二層。
最終 result 的 token 欄位仍是該層的用量；整體消耗以 JSONL 統計為準。
文字解析失敗時，已收到的 usage 仍會保留。
有錯誤的事件會記為 error，不能拿不完整 run 宣稱有效降幅。

每支影片另生成 .jsonl.run.json，包含影片 SHA256、VLM 設定、完成狀態和彙總。
compare_tokens 驗證影片內容與 VLM/抽樣設定相同，且兩组完整完成、無錯誤、usage 無缺漏。
使用者按 q、事件處理失敗、VLM 失敗或未完整讀完影片均不能計算有效降幅。
若 OpenCV 提供的總影格數不正確，會保守拒絕將較少解碼影格的 run 當作完整。
等待所有背景 VLM 工作與尾端事件完成後才寫入 ours 的最終統計。
完整完成但沒有保留事件時，ours 為 0 次、0 tokens，不會與沒有執行混淆。
usage 缺少欄位顯示 null，不能當作零。

比較輸出含：
- VLM 呼叫次數（收到成功 JSON 回應的 backend 次數，不是事件數）
- input / output / total tokens
- 按 backend/model 分組的 total tokens
- token reduction (%) 與 call reduction (%)

降幅 = (baseline - ours) / baseline × 100%。
baseline 為 0 時不定義降幅，輸出 null。
多支影片應先加總兩組用量再計算整體降幅，不直接平均百分比。

## 若繼續用你原本的主程式

可以不使用 run_ours.py，而在原本 play_and_live_inference(...) 最後加入：

```python
    token_log_path="ours_fall01.jsonl",
    experiment_media_clock=True,
    vlm_timeout=600,
```

本來已傳入的模型等參數保持不變。
離線實驗建議 line_user_id=None、CONFIG['is_live_stream']=False。
不要同時在同一程序跑多支影片；logger 使用程序共用的實驗環境。
舊的 VLM_TOKEN_LOG 環境變數若存在，begin_run 會拒絕啟動，以免覆寫進行中的實驗。

## 計數限制與驗證

本套件統計的是 backend 回報 token，不是整套系統 GPU 算力或金額。
圖片 token 是否完整納入，仍要確認你實際 relay/backend 的計數定義；
不以純文字 tokenizer 估算圖片成本。
雙層使用不同模型時，兩層 token 可描述總回報量，但不能當成相同單位的運算成本；
因此另提供按模型的統計。
比較時同時量測異常召回率、漏報與誤報，避免只因漏掉異常而節省 token。

已驗證：Python 語法、雙層累計、解析失敗保留 usage、0 呼叫、usage 缺漏、
共用抽幀/JPEG 參數、baseline 最後短窗、逾時設定傳入雙層 API 與背景 VLM worker
（使用模擬回應及影片讀取器）。
未在你的 GPU、模型伺服器和真實影片上跑端到端實驗；尚無實測 token 結果。
