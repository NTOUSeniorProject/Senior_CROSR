# alarm_YT 模組化結構

## 📁 檔案結構

```
Senior_CROSR/
├── alarm_YT.py               ← 主程式（只負責啟動）
├── constants.py              ← 全域配置與常數
├── video_source.py           ← 影片來源解析（YouTube、RTSP、本地檔案）
├── model_loader.py           ← 模型加載（ST-CROSR、雷達參數）
├── inference.py              ← 推論邏輯（骨架->特徵->分數）
├── event_handler.py          ← 異常事件管理（偵測、保存、篩選）
├── line_notifier.py          ← LINE 通知（推播、訊息格式）
├── real_time_detector.py     ← 即時推論迴圈（核心邏輯）
│
├── VLM_check.py              ← Ollama VLM 分析（外部）
├── movement_detection.py     ← MOG2 背景相減（外部）
└── Functions/                ← ST-CROSR 模型目錄（外部）
    ├── ST_CROSR.py
    ├── ntu_normalize.py
    └── ...
```

## 🔧 各模組職責

### 1. **alarm_YT.py** (主程式) - 57 行
```python
# 只負責：
# - 設置裝置 (GPU/CPU)
# - 加載所有模型
# - 啟動推論引擎
```

### 2. **constants.py** (配置) - 125 行
```python
CONFIG = {
    "video_path": "...",           # 影片來源
    "window_size": 120,            # 骨架視窗大小
    "anomaly_vote_ratio": 0.70,    # 異常投票比例
    ...
}

ACTION_NAMES = {...}               # 60 種動作名稱
KNOWN_ACTIONS = [...]              # 已知動作列表
```

### 3. **video_source.py** (影片解析) - 200 行
```
is_url() ─────────────────┐
is_youtube_url()          │
is_direct_stream_url()    │─→ resolve_video_source() ─→ open_video_capture()
check_ytdlp_installed()   │
resolve_youtube_stream_url()
```

**功能**：
- ✅ 支援本地影片檔案
- ✅ 解析 YouTube 連結（透過 yt-dlp）
- ✅ RTSP/RTMP/m3u8 直播串流
- ✅ HTTP/HTTPS 影片

### 4. **model_loader.py** (模型載入) - 60 行
```python
load_radar_meta_params(device)
├─ 讀取 radar_meta_params.pth
├─ 提取質心、歸一化參數
└─ 返回 (centroids_norm, normalizer, threshold, ...)

load_st_crosr_model(device)
├─ 讀取 checkpoint_path
├─ 初始化 ST_CROSR 網路
└─ 返回模型 (eval 模式)
```

### 5. **inference.py** (推論邏輯) - 120 行
```
pad_or_cut_to_300(clip)
        ↓
normalize_skeleton_batch()
        ↓
model(clip_tensor) ─→ logits, recon_x, z
        ↓
┌──────────────────────────────────────────┐
│ 1. 已知動作分類 (softmax)                 │
│ 2. 骨架重構誤差 (Masked MSE)              │
│ 3. 特徵距離 (Cosine Distance)            │
│ 4. 融合異常分數 (Combined Score)         │
└──────────────────────────────────────────┘
        ↓
return {
    "combined_score": 0.52,
    "is_unknown": True,
    "nearest_action_name": "falling",
    ...
}
```

### 6. **event_handler.py** (事件管理) - 220 行
```
異常確認 → start_event_collection()
          ├─ 回溯前 5 秒影格
          └─ 預期後 5 秒截止時間
               ↓
          [收集 10 秒影格]
               ↓
          finish_event_collection()
          ├─ should_keep_event()  ← 判斷是否保留
          └─ save_anomaly_event_frames()
             ├─ 儲存完整異常影片
             └─ 抽 8 張影格給 VLM
```

**篩選條件**：
- 完整收集：≥2 個異常視窗 + ≥30% 比例
- 提前結束：≥1 個異常視窗 + ≥50% 比例

### 7. **line_notifier.py** (LINE 通知) - 140 行
```python
format_video_time(seconds)      # 秒 → HH:MM:SS
push_line_message(user_id, text) # 發送 LINE 推播
build_alert_message(...)         # 異常警報訊息
build_analysis_summary(...)      # 分析完成摘要
```

### 8. **real_time_detector.py** (核心引擎) - 1100 行
```
迴圈讀取影格
    ↓
MOG2 背景相減 → 有移動?
    │           ├─ NO  → 跳過 YOLO
    │           └─ YES ↓
    │         YOLO 骨架偵測
    │              ↓
    │         緩衝視窗達 120 幀?
    │              ├─ NO  → 繼續
    │              └─ YES ↓
    │         每 30 幀推論一次
    │              ↓
    │         predict_one_clip()
    │              ↓
    │    ┌────────────────────────────┐
    │    │  異常狀態機                │
    │    │  ┌──────────────────────┐  │
    │    │  │ normal → candidate   │  │
    │    │  │   → alert → normal   │  │
    │    │  └──────────────────────┘  │
    │    └────────────────────────────┘
    │              ↓
    │    是否達到 VLM 判斷點?
    │              ├─ YES ↓
    │              │   VLM 二次確認
    │              │   (信心度 ≥75%)
    │              │   → LINE 推播
    │              └─ NO → 繼續監控
    ↓
影片結束
```

## 🚀 使用方式

### 基本用法
```bash
# 使用 CONFIG 預設值
python alarm_YT.py

# 指定影片來源
python alarm_YT.py "https://youtu.be/xxx" "user_id_123"

# 本地檔案
python alarm_YT.py "C:\video.mp4" "user_id_123"

# RTSP 直播
python alarm_YT.py "rtsp://192.168.1.100:554/stream" "user_id_123"
```

### 修改配置
編輯 `constants.py`：
```python
CONFIG = {
    "video_path": "https://youtu.be/...",
    "window_size": 120,           # ← 骨架視窗大小
    "stride": 30,                 # ← 推論步長
    "anomaly_vote_ratio": 0.70,   # ← 異常投票比例
    "manual_threshold": 0.5,      # ← 異常分數閾值（越高越嚴格）
    "consecutive_alert_sec": 2,   # ← 異常持續時間
    "alert_cooldown_sec": 30,     # ← 推播冷卻時間
}
```

## 🔄 數據流

```
視訊幀 (frame)
    ↓
MOG2 檢測移動
    ↓
YOLO 抽取骨架 (17 keypoints)
    ↓
骨架緩衝 (120 幀)
    ↓
ST-CROSR 推論
├─ Encoder → 特徵向量 z
├─ MSE 重構誤差
└─ Cosine 距離
    ↓
Combined Score = 0.4×norm_dist + 0.6×norm_mse
    ↓
異常投票 (最近 3 秒)
    ├─ ≥70% unknown → candidate
    ├─ 持續 2 秒 → alert
    └─ 連續 3 個正常 → normal
    ↓
事件收集 (前後各 5 秒)
    ↓
VLM 二次判斷
    ├─ 信心度 ≥75% → LINE 推播 ✅
    └─ 信心度 <75% → 丟棄
```

## 📊 模組大小對比

| 模組 | 原始行數 | 現在行數 | 降幅 |
|------|--------|--------|------|
| alarm_YT_backup.py | 1773 | - | - |
| 🔴 合併後 | - | 57 | ↓ 97% |
| 💾 所有模組 | - | ~2000 | ↓ 13% |

## ✅ 重構優勢

1. **模組獨立** - 每個模組專注一個功能
2. **易於測試** - 可單獨測試 `inference.py`、`event_handler.py` 等
3. **易於維護** - 修改邏輯不會影響其他模組
4. **易於擴展** - 新增功能（如 Telegram、Slack）只需改 `line_notifier.py`
5. **易於除錯** - 錯誤來源一目瞭然

## 🐛 除錯技巧

```python
# 測試 inference 邏輯
from inference import predict_one_clip
result = predict_one_clip(model, clip, device, ...)
print(f"Score: {result['combined_score']}, Unknown: {result['is_unknown']}")

# 測試影片解析
from video_source import resolve_video_source
resolved = resolve_video_source("https://youtu.be/xxx")
print(f"Resolved URL: {resolved}")

# 測試模型載入
from model_loader import load_st_crosr_model
model = load_st_crosr_model(device)
print(f"Model loaded: {model}")
```

## 📝 注意事項

1. `constants.py` 中的 `KNOWN_ACTIONS` 會被 `radar_meta_params.pth` 覆蓋
2. 修改 `CONFIG` 後無需重啟，該次執行會使用新值
3. `real_time_detector.py` 內的參數都從 `CONFIG` 讀取，便於動態調整

---

**結論**：通過模組化，原本 1773 行的巨石程式現在分解成 8 個職責明確的模組，主程式只需 57 行！
