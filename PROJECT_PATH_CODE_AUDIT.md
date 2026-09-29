# 專案路徑與重複程式碼檢查（2026-09-29）

檢查範圍：根目錄、Functions、NonUse 共 39 個 Python 檔案，以及專案說明與 .gitignore。排除 yolo 虛擬環境的第三方程式、模型權重、影片與快取。以目前工作區為準，未執行訓練、遠端 VLM 或 LINE 推播，未修改原始程式。

## 優先處理的問題

### 1. 模型套件混用絕對與相對匯入

- `Functions/ST_CROSR.py:4` 使用 `.STGCNEncoder`，但 `Functions/STGCNEncoder.py:6-8` 使用 `from net...`，而實際位置是 `Functions/net/`。
- 目前 Python 環境查不到頂層 `net`。`train_ntu_crosr.py:6`、`test_ntu_dataset.py:10`、`alarm.py:8` 在沒有額外 PYTHONPATH 設定時，匯入 Functions 會走到這個問題。
- `model_loader.py:6`、`inference.py:8`、`alarm_YT.py:9` 三處重複插入 Functions 到 sys.path。主入口透過 model_loader 提前插入路徑，掩蓋了套件本身的匯入問題；alarm_YT 自己的插入發生在模組匯入之後。
- 建議改成 `.net.net`、`.net.unit_gcn`、`.net.st_gcn`，再移除多餘的 sys.path 修改，逐一驗證各入口。不要只刪除 sys.path 插入而不修套件。

### 2. 重複設定鍵會靜默覆蓋

| 檔案 | 鍵 | 前值 → 實際生效值 |
|---|---|---|
| constants.py:26、34 | consecutive_alert_sec | 2.0 → 2（數值相同） |
| constants.py:28、35 | alert_cooldown_sec | 30.0 → 4 |
| train_ntu_crosr.py:43、47 | lambda_center | 0.0001 → 0.001 |

訓練中心損失權重實際是前一設定的十倍。修改前面的設定不會生效。NonUse/alarm_YT_backup.py 也保留重複的警報設定。應每個鍵只保留一個定義，但必須先決定真正想使用的值。

### 3. 動作清單的重複引用沒有同步更新

`model_loader.py:4` 與 `inference.py:6` 都從 constants 匯入 KNOWN_ACTIONS。
`model_loader.py:33` 以新 list 重新綁定自己的 KNOWN_ACTIONS，沒有更新 constants 或 inference 已持有的清單。
如果 meta 檔的 known_actions 順序或內容不同，模型載入使用新清單，而推論標籤仍使用舊清單，可能造成動作名稱錯置或索引錯誤。
建議由載入函式回傳動作清單，明確傳入模型建立與推論流程，避免多處持有不同版本。

### 4. 模型與輸出路徑依賴工作目錄

`constants.py:11-13` 的權重路徑、`:93` 的 abnormal_events，以及訓練／測試腳本的資料與輸出位置，都相對於目前工作目錄，而非 Python 檔案所在位置。
從另一個資料夾用完整路徑啟動 alarm_YT，仍可能找不到模型。`lineBot.py:32-55` 已透過 BASE_DIR 與 subprocess cwd 處理這個問題，但其他啟動方式沒有同樣保護。
建議以專案根目錄集中解析模型與輸出路徑，並明確規定使用者輸入的相對影片路徑以哪個目錄為基準。

### 5. 備份搬到 NonUse 後引用失效

目前 Git 顯示根目錄 alarm_YT_backup.py 刪除、NonUse/alarm_YT_backup.py 新增，為既有工作區變更，本次未修改。
備份 `:23` 仍把 `__file__` 所在目錄加上 Functions，得到不存在的 `NonUse/Functions`；`:14-15` 也直接引用根目錄模組。
NonUse/alarmTest.py、NTU_CROSR.py、draw_ntu_skeleton.py、test_ntu_dataset_detail.py 等仍使用頂層 ST_CROSR、ntu_normalize 等舊匯入方式。
若僅供保存歷史，可標示不可直接執行；若仍要執行，需同步調整套件與路徑。

## 路徑存在性與集中設定

| 參考位置 | 目標 | 檢查結果 |
|---|---|---|
| constants.py:11 | yolo26x-pose.pt | 專案根目錄存在 |
| constants.py:12、test_ntu_dataset.py:43 | checkpoints_20260602_2237/best_val.pth | 存在，但兩處各自寫死 |
| constants.py:13 | radar_meta_params.pth | 存在 |
| test_ntu_dataset.py:254 | radar_meta_params.pth | 產生端與讀取端各自使用相對路徑，換工作目錄可能寫入另一份 |
| train_ntu_crosr.py:25、test_ntu_dataset.py:65、yoloSkeleton.py:113 | NTU60/nturgb+d_yolo_skeletons | 本機專案下不存在 |
| alarm.py:21、22、24 | C:/CROSR/IMG_2033.mov、yolo26x-pose.pt、radar_meta_params.pth | 三個固定絕對路徑均不存在 |
| NonUse/alarm_YT_backup.py:23 | NonUse/Functions | 不存在 |

訓練、測試與轉檔共用同一資料夾是合理依賴，不代表重複執行，但建議集中設定以免改動時只改一處。
訓練的 find_latest_checkpoint 使用目前目錄掃描 checkpoints_*/last.pth；模型推論則固定選某個日期的 best_val.pth，兩者沒有自動對應。

VLM_check.py:12-20 與 remoteTest.py:21-29 分別維護重複的 VLM 端點與模型名稱，可能造成測試連線成功、正式流程卻使用另一端點。vlmRelay.py 的 upstream 是下一跳主機，屬不同角色，不應直接與客戶端位址合併。

## 重複程式碼

以 AST 比對至少 9 行的同名函式，排除格式與註解差異，共找到 15 組完全相同的函式；並非 15 個完全相同檔案。

| 函式 | 重複位置 |
|---|---|
| load_radar_meta_params | alarm.py:82、NonUse/alarmTest.py:82、NonUse/alarm_YT_backup.py:518 |
| pad_or_cut_to_300 | alarm.py:118、NonUse/alarmTest.py:118、NonUse/alarm_YT_backup.py:565、NonUse/TestRealVideo.py:328 |
| predict_one_clip | alarm.py:137、NonUse/alarmTest.py:137 |
| main | alarm.py:320、NonUse/alarmTest.py:288 |
| main | alarm_YT.py:12、NonUse/alarm_YT_backup.py:1719 |
| finish_event_collection | event_handler.py:152、NonUse/alarm_YT_backup.py:819 |
| format_video_time | line_notifier.py:5、NonUse/alarm_YT_backup.py:203 |
| push_line_message | line_notifier.py:24、NonUse/alarm_YT_backup.py:108 |
| build_alert_message | line_notifier.py:73、NonUse/alarm_YT_backup.py:157 |
| build_analysis_summary | line_notifier.py:89、NonUse/alarm_YT_backup.py:221 |
| is_youtube_url | video_source.py:16、NonUse/alarm_YT_backup.py:298 |
| is_direct_stream_url | video_source.py:31、NonUse/alarm_YT_backup.py:313 |
| check_ytdlp_installed | video_source.py:51、NonUse/alarm_YT_backup.py:334 |
| open_video_capture | video_source.py:181、NonUse/alarm_YT_backup.py:472 |
| compute_combined_score | NonUse/alarm_YT_backup.py:579、NonUse/TestRealVideo.py:350 |

其他值得整理的非完全相同邏輯：

- real_time_detector 的 `_analyze_event_with_vlm` 與影片結尾事件處理各寫一份 VLM 判斷、0.75 門檻與通知組裝；目前兩份訊息內容已不同。建議抽出共同結果處理函式，保留背景與結尾執行方式的差異。
- constants、train_ntu_crosr、test_ntu_dataset、alarm 與備份各自維護已知動作清單。訓練／推論映射應由模型 metadata 明確約束。
- real_time_detector.py:6、14 重複匯入 Lock，不會造成兩個 Lock 類型，但可移除重複行。
- .gitignore:1-12 的四個權重副檔名規則重複三次，屬無害重複。
- 備份重複不等於執行時跑兩次：主入口沒有引用 NonUse，不建議因掃描到重複就直接刪除歷史備份。

## 與先前實驗功能的差異

目前沒有 systemVsVLM.py、test_vlm_experiment.py、SYSTEM_VS_VLM.md。
VLM_check.py 目前沒有實驗轉送入口；constants.py:10 的 video_path 是固定 YouTube URL，沒有 os.getenv('VIDEO_PATH')。
因此先前對話中的 .env VIDEO_PATH 與 token 實驗操作方式不適用目前這份工作區。不能單憑對話斷定遺失原因；應先確認版本或分支，再決定是否恢復。

## 建議處理順序與驗證範圍

1. 修 Functions 的套件匯入，驗證訓練、測試、即時推論三種入口。
2. 確認並消除重複設定鍵，統一 known_actions 的來源與傳遞。
3. 集中路徑解析，從專案根目錄及其他工作目錄各測一次。
4. 抽出共用 VLM 結果處理；標示或整理 NonUse 備份。
5. 若需要續做 token 比較，再恢復缺少的實驗模組。

已完成 39 檔 AST 語法解析、函式重複比對、主要路徑存在性與頂層 net 模組查找。未驗證 GPU 推論數值、完整訓練、網路可達性或所有第三方套件相容性。


## 後續更新：設定集中化

正式入口 alarm.py、訓練、資料測試與 YOLO 轉檔已引用 constants.py 的共用設定；VLM_check 與 remoteTest 的共用端點也移入 constants.py。constants 的重複鍵已清除，保留冷卻 4 秒與連續異常 2 秒；TRAINING_CONFIG 保留 lambda_center=0.001。模型 metadata 的動作清單改為原地更新，確保 inference 的引用同步。NonUse 為歷史備份，尚未改動；上述檢查表描述修正前狀態。其餘套件匯入與相對路徑問題仍待處理。


## 後續更新：套件匯入修正與驗證

STGCNEncoder 的三個 net 匯入已改為套件相對匯入，Functions/net 新增 __init__.py；model_loader、inference、alarm_YT 移除 Functions 的 sys.path 插入及不再使用的匯入。

以獨立 Python 程序驗證 Functions.STGCNEncoder、Functions.ST_CROSR、train_ntu_crosr、alarm_YT、alarm 均可匯入，且沒有載入頂層 net 或添加 Functions 搜尋路徑。現有 best_val.pth 以 strict load_state_dict 成功載入，CPU 模擬骨架推論輸出為 (1,33)、(1,2,300,17)、(1,256)、(1,100)，數值均有限。

test_ntu_dataset 的完整匯入受限於缺少 seaborn（目前 Python 與 yolo 虛擬環境皆無此套件）；未執行完整資料集評估。Ultralytics 匯入時對使用者設定目錄發出權限警告，但入口匯入成功。NonUse 歷史腳本與模型檔相對路徑問題不在本次修正範圍。
