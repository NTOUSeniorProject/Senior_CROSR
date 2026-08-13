import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 配置區
# ============================================================
CONFIG = {
    "video_path": r"https://youtu.be/kD0RBvXA1q4?si=ZJnV3lV45Yifloay",
    "yolo_model_path": r"yolo26x-pose.pt",
    "checkpoint_path": r"checkpoints_20260602_2237\best_val.pth",
    "radar_meta_path": r"radar_meta_params.pth",

    "max_frames": 300,
    "num_nodes": 17,
    "center_joint_idx": 11,

    "window_size": 120,
    "stride": 30,

    "anomaly_confirm_window_sec": 3.0,
    "anomaly_vote_ratio": 0.70,
    "min_anomaly_votes": 3,

    "consecutive_alert_sec": 2.0,
    "clear_normal_windows": 3,
    "alert_cooldown_sec": 30.0,

    "use_saved_threshold": False,
    "manual_threshold": 0.35,

    "use_consecutive_alert": True,
    "consecutive_alert_sec": 2,
    "alert_cooldown_sec": 4,

    "show_yolo_window": True,

    "enable_youtube_url": True,
    "youtube_format": "best[protocol^=m3u8][height<=720]/best[protocol^=m3u8]/best[height<=720]/best",
    "use_browser_cookies": False,
    "cookies_browser": "chrome",

    "use_ffmpeg_backend": True,
    "is_live_stream": False,
}

# LINE 通知設定
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

# 已知動作清單
DEFAULT_KNOWN_ACTIONS = [
    1, 2, 3, 4, 5, 6,
    8, 9, 11, 12,
    14, 15, 16, 17, 18, 19, 20, 21,
    23, 25,
    28, 29, 30, 32, 33, 34, 37,
    41, 44, 45, 46, 47, 49
]

KNOWN_ACTIONS = DEFAULT_KNOWN_ACTIONS.copy()

# 動作名稱對應
ACTION_NAMES = {
    1: "drink water", 2: "eat meal/snack", 3: "brushing teeth", 4: "brushing hair",
    5: "drop", 6: "pickup", 7: "throw", 8: "sitting down", 9: "standing up",
    10: "clapping", 11: "reading", 12: "writing", 13: "tear up paper",
    14: "wear jacket", 15: "take off jacket", 16: "wear shoe", 17: "take off shoe",
    18: "wear glasses", 19: "take off glasses", 20: "put on hat/cap", 21: "take off hat/cap",
    22: "cheer up", 23: "hand waving", 24: "kicking something", 25: "reach into pocket",
    26: "hopping", 27: "jump up", 28: "make a phone call", 29: "playing with phone/tablet",
    30: "typing on keyboard", 31: "pointing to something", 32: "taking a selfie",
    33: "check time", 34: "rub two hands", 35: "nod head/bow", 36: "shake head",
    37: "wipe face", 38: "salute", 39: "put palms together", 40: "cross hands in front",
    41: "sneeze/cough", 42: "staggering", 43: "falling", 44: "touch head",
    45: "touch chest", 46: "touch back", 47: "touch neck", 48: "nausea/vomiting",
    49: "use a fan", 50: "punching/slapping", 51: "kicking", 52: "pushing",
    53: "pat on back", 54: "point finger", 55: "hugging", 56: "giving object",
    57: "touch pocket", 58: "shaking hands", 59: "walking towards", 60: "walking apart",
}

# 事件處理常數
PRE_EVENT_SECONDS = 5.0
POST_EVENT_SECONDS = 5.0
POST_MIN_ANOMALY_VOTES = 1
POST_ANOMALY_RATIO_THRESHOLD = 0.30
VLM_SAMPLE_FRAME_COUNT = 8
ANOMALY_OUTPUT_ROOT = "./abnormal_events"
