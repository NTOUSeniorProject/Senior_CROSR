"""人物分組與已知類別校準；不依賴舊 Dataset 的 P<=30 條件。"""

import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from valid.cross_validation_config import SUBJECT_GROUPS
from training.ntu_skeleton_dataset import NTUSkeletonDataset


def subject_split(fold):
    if not 1 <= fold <= 10:
        raise ValueError("fold 必須介於 1 到 10")
    flat = [p for group in SUBJECT_GROUPS for p in group]
    if len(flat) != 40 or set(flat) != set(range(1, 41)):
        raise ValueError("人物分組必須恰好涵蓋 P001-P040，各一次")
    test = set(SUBJECT_GROUPS[fold - 1])
    val = set(SUBJECT_GROUPS[fold % 10])
    return {"train": sorted(set(flat) - test - val),
            "val": sorted(val), "test": sorted(test)}


def scan_samples(data_root):
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"找不到骨架資料夾：{root}")
    records, seen = [], set()
    for path in sorted(root.glob("*.npy")):
        match = re.fullmatch(r"(S(\d{3})C00([123])P(\d{3})R00([12])A(\d{3}))(?:_rgb)?", path.stem)
        if not match:
            raise ValueError(f"無法辨識 NTU60 檔名：{path.name}")
        identity, setup, _, subject, _, action = match.groups()
        setup, subject, action = int(setup), int(subject), int(action)
        if not (1 <= setup <= 17 and 1 <= subject <= 40 and 1 <= action <= 60):
            raise ValueError(f"不是 NTU60 範圍內的檔案：{path.name}")
        if identity in seen:
            raise ValueError(f"重複樣本（例如同時有 _rgb 與無後綴版本）：{identity}")
        seen.add(identity)
        records.append({"path": str(path.resolve()), "subject": subject, "action": action})
    if not records:
        raise ValueError(f"資料夾內沒有 .npy：{root}")
    return records


def split_records(records, fold, known_actions):
    subjects = subject_split(fold)
    known = set(known_actions)
    splits = {
        split: [r for r in records if r["subject"] in ids
                and (split == "test" or r["action"] in known)]
        for split, ids in subjects.items()
    }
    for split, rows in splits.items():
        if not rows:
            raise ValueError(f"Fold {fold} 的 {split} 沒有可用資料")
        missing = known - {r["action"] for r in rows}
        if missing:
            raise ValueError(f"Fold {fold} {split} 缺少已知類別：{sorted(missing)}")
    if not any(r["action"] not in known for r in splits["test"]):
        raise ValueError(f"Fold {fold} 測試集缺少未知類別")
    return splits


def describe_records(records, known_actions):
    known = set(known_actions)
    return {"total": len(records),
            "known": sum(r["action"] in known for r in records),
            "unknown": sum(r["action"] not in known for r in records),
            "by_subject": dict(sorted(Counter(r["subject"] for r in records).items())),
            "by_action": dict(sorted(Counter(r["action"] for r in records).items()))}


class FoldDataset(NTUSkeletonDataset):
    """重用原本的補零/截斷方法，使用明確的樣本清單。"""

    def __init__(self, records, known_actions, max_frames, num_nodes):
        self.records = records
        self.max_frames = max_frames
        self.num_nodes = num_nodes
        self.known_classes = sorted(known_actions)
        labels = {action: i for i, action in enumerate(self.known_classes)}
        self.samples = [(r["path"], labels.get(r["action"], -1),
                         int(r["action"] not in labels)) for r in records]

    def __getitem__(self, index):
        path, label, unknown = self.samples[index]
        data = np.load(path, allow_pickle=False)
        if data.ndim != 3 or data.shape[0] != 2 or data.shape[2] != self.num_nodes or data.shape[1] == 0:
            raise ValueError(f"骨架應為 [2,T,{self.num_nodes}] 且 T>0：{path}, {data.shape}")
        if not np.isfinite(data).all() or not np.any(data):
            raise ValueError(f"骨架包含 NaN/Inf 或全為零：{path}")
        return (torch.from_numpy(self.fix_frames(data.astype(np.float32), self.max_frames)),
                torch.tensor(label, dtype=torch.long), torch.tensor(unknown, dtype=torch.long))


def normalize_samples(inputs, center_joint_idx):
    # 與舊程式相同的中心化/標準差方式，但每個樣本獨立計算。
    centered = inputs - inputs[:, :, :, center_joint_idx:center_joint_idx + 1]
    return centered / (centered.std(dim=(1, 2, 3), keepdim=True) + 1e-6)


def combined_scores(distance, mse, calibration):
    distance, mse = np.asarray(distance), np.asarray(mse)
    d = (distance - calibration["dist_min"]) / (calibration["dist_max"] - calibration["dist_min"] + 1e-8)
    m = (np.log1p(mse) - calibration["mse_min"]) / (calibration["mse_max"] - calibration["mse_min"] + 1e-8)
    # 不截斷測試分數：超出驗證集範圍的異常仍可保留排序資訊。
    weight = calibration["cosine_weight"]
    return weight * d + (1 - weight) * m


def calibrate(distance, mse, known_fpr, cosine_weight):
    if not 0 < known_fpr < 1 or not 0 <= cosine_weight <= 1:
        raise ValueError("known_fpr 須在 (0,1)，cosine_weight 須在 [0,1]")
    distance, mse = np.asarray(distance), np.asarray(mse)
    if not len(distance) or not np.isfinite(distance).all() or not np.isfinite(mse).all():
        raise ValueError("校準分數為空或包含非有限值")
    log_mse = np.log1p(mse)
    result = {"dist_min": float(distance.min()), "dist_max": float(distance.max()),
              "mse_min": float(log_mse.min()), "mse_max": float(log_mse.max()),
              "cosine_weight": cosine_weight, "target_known_fpr": known_fpr,
              "comparison": ">"}
    scores = combined_scores(distance, mse, result)
    # higher + 嚴格大於：即使分數有平手，經驗誤報率也不會超過指定值。
    result["threshold"] = float(np.quantile(scores, 1 - known_fpr, method="higher"))
    result["validation_known_fpr"] = float(np.mean(scores > result["threshold"]))
    return result
