# Senior_CROSR

程式依訓練、推論、共用功能分類，保留原檔名。

```text
alarm_YT.py                 影片分析入口
lineBot.py                  LINE Bot 入口
constants.py                統一設定與 PROJECT_ROOT
training/
  train_ntu_crosr.py        訓練
  test_ntu_dataset.py       模型評估（需要 seaborn）
  yoloSkeleton.py           資料轉換
  ntu_skeleton_dataset.py   資料集
  center_loss.py            訓練損失
inference/
  real_time_detector.py     分析迴圈
  inference.py             骨架評分
  model_loader.py
  video_source.py
  movement_detection.py
  event_handler.py
  VLM_check.py
  line_notifier.py
common/
  image_encoding.py
  ntu_normalize.py
  models/
    ST_CROSR.py
    STGCNEncoder.py
    strictly_bottleneck_decoder.py
    actionMemoryModule.py
    net/
      net.py
      st_gcn.py
      unit_gcn.py
tools/
  vlmRelay.py               PC-lab 轉送服務
NonUse/                    連線診斷、自動測試與歷史程式
```

各套件包含 `__init__.py`。原 Functions 的程式已分入 common 與 training；舊目錄若仍有 __pycache__，它是快取，不是現行程式來源。
模型、.env、資料集與異常事件資料維持原位置，路徑由根目錄 constants.py 解析。

## 執行

在專案根目錄執行：

```powershell
python alarm_YT.py
python lineBot.py
python -m training.train_ntu_crosr
python -m training.test_ntu_dataset
python -m training.yoloSkeleton
python -m tools.vlmRelay
python -m NonUse.remoteTest
python -m unittest NonUse.test_vlm_notifications NonUse.test_vlm_event_submission -v
```

子資料夾程式使用 `python -m 套件.模組`，不要直接用 `python training/train_ntu_crosr.py`，以免缺少專案根目錄的匯入位置。
從其他目錄啟動主程式可使用 alarm_YT.py 的完整路徑；模組命令則先切到專案根目錄。
NonUse 的歷史程式僅更新匯入位置，不代表舊資料路徑與演算法已完成驗證。
