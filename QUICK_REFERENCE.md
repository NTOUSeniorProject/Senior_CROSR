# 🚀 快速參考指南

## 執行程式

```bash
# 方式 1：使用預設配置
python alarm_YT.py

# 方式 2：指定 YouTube 連結
python alarm_YT.py "https://youtu.be/kD0RBvXA1q4?si=ZJnV3lV45Yifloay"

# 方式 3：完整參數
python alarm_YT.py "影片路徑" "LINE_user_id"
```

## 模組依賴圖

```
                    alarm_YT.py (主程式)
                        │
         ┌──────────────┼──────────────┐
         ↓              ↓              ↓
  model_loader    real_time_detector  constants
      │                  │                │
      ├──→ constants     │          ┌─────┘
      │                  │          │
      └──→ Functions/    ├─→ video_source
                         │          ├─→ constants
                         │          │
                         ├─→ inference
                         │    ├─→ constants
                         │    └─→ Functions/
                         │
                         ├─→ event_handler
                         │    └─→ constants
                         │
                         ├─→ line_notifier
                         │    └─→ constants
                         │
                         └─→ movement_detection
                         └─→ VLM_check
```

## 修改指南

### ❌ 不要修改這些
```python
# ❌ 別直接改程式碼
alarm_YT.py          # 主程式邏輯
real_time_detector.py # 核心演算法
```

### ✅ 改這些地方
```python
# ✅ 在 constants.py 調整參數
CONFIG = {
    "manual_threshold": 0.4,      # ← 改異常判斷敏感度
    "consecutive_alert_sec": 2,   # ← 改異常持續時間
    "anomaly_vote_ratio": 0.70,   # ← 改投票比例
    "alert_cooldown_sec": 30,     # ← 改推播冷卻
}
```

## 常見調整

| 需求 | 修改位置 | 參數 | 效果 |
|------|--------|------|------|
| 🎯 降低誤報 | constants.py | `manual_threshold` ↑ 0.5 | 只報真異常 |
| 📢 增加靈敏 | constants.py | `manual_threshold` ↓ 0.3 | 更易觸發警報 |
| ⏱️ 延長確認時間 | constants.py | `consecutive_alert_sec` ↑ 3 | 等待更久才報警 |
| 🔕 減少推播 | constants.py | `alert_cooldown_sec` ↑ 60 | 推播間隔更長 |
| 🎬 更小視窗 | constants.py | `window_size` ↓ 60 | 反應更快但可能誤報 |

## 故障排查

### ❌ 找不到影片
```
FileNotFoundError: 找不到影片來源
```
**解決**：
- 檢查路徑是否正確
- Windows 路徑用 `r"C:\path\file.mp4"`
- YouTube 連結確認可直接瀏覽

### ❌ 找不到 yt-dlp
```
RuntimeError: 找不到 yt-dlp
```
**解決**：
```bash
pip install yt-dlp
```

### ❌ LINE 推播失敗
```
⚠️ 缺少 LINE_CHANNEL_ACCESS_TOKEN
```
**解決**：
- 在 `.env` 檔案設置 TOKEN
- 或在終端機設置環境變數

### ❌ YOLO 模型找不到
```
FileNotFoundError: yolo26x-pose.pt
```
**解決**：
```bash
# 會自動下載（第一次執行）
# 或手動下載到目前目錄
```

## 監控指標

執行時你會看到這些輸出：

```
🚀 啟動改良版即時異常動作監控系統...
原始影片來源: https://youtu.be/...
影片模式: 串流 / 直播
FPS: 25.00
骨架視窗: 120 幀，步長: 30 幀
異常投票: 最近約 3.0 秒內，unknown 比例至少 70%
...

[正常運作]
✅ [  50.25 秒] 進入異常候選 | 推估已持續 0.35 秒 | 異常視窗比例 75% (3/4) | 分數 0.5234

[異常確認]
🚨 【確認異常】  55.00 秒 | 異常持續 2.1 秒 | 異常視窗比例 78% | 綜合分數 0.6521

[VLM 判斷]
🤖 將 8 張影格交給 Ollama VLM
🧠 Ollama VLM 分析結果：{'is_abnormal': True, 'confidence': 0.92, ...}

[LINE 推播]
LINE Push 狀態碼：200
```

## 性能優化

| 改進項 | 方法 | 效果 |
|-------|------|------|
| 💨 更快反應 | `stride` ↓ 20 | 每 20 幀推論一次 |
| 💾 更少記憶體 | `window_size` ↓ 60 | 較小視窗 |
| 🔋 更低CPU | 跳過 YOLO | 啟用 MOG2 動作檢測 |
| ⚡ GPU 加速 | 確保 `cuda` | 查看 `🖥️ 使用裝置` |

## 文件位置

| 檔案 | 用途 | 行數 |
|-----|------|------|
| `alarm_YT.py` | 主程式入口 | 57 |
| `constants.py` | 全域配置 | 125 |
| `video_source.py` | 影片解析 (YouTube/RTSP) | 200 |
| `model_loader.py` | 模型載入 | 60 |
| `inference.py` | 推論邏輯 (ST-CROSR) | 120 |
| `event_handler.py` | 事件管理 (偵測/保存) | 220 |
| `line_notifier.py` | LINE 通知 | 140 |
| `real_time_detector.py` | 核心推論迴圈 | 1100 |
| `MODULE_STRUCTURE.md` | 詳細設計文件 | - |

## 快速除錯

```python
# 進入 Python
python

# 測試影片解析
from video_source import resolve_video_source
url = resolve_video_source("https://youtu.be/xxx")
print(url)

# 測試模型載入
import torch
from model_loader import load_radar_meta_params, load_st_crosr_model
device = torch.device("cuda")
centroids_norm, normalizer, threshold, dist_weight, mse_weight = load_radar_meta_params(device)
model = load_st_crosr_model(device)
print("✅ Models loaded successfully")

# 測試推論
from inference import predict_one_clip
import numpy as np
clip = np.random.randn(2, 300, 17).astype(np.float32)
result = predict_one_clip(model, clip, device, centroids_norm, normalizer, threshold, dist_weight, mse_weight)
print(f"Score: {result['combined_score']:.4f}")
```

---

💡 **提示**：所有參數都在 `constants.py`，不用改程式碼就能調整行為！
