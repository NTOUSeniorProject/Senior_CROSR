import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class NTUSkeletonDataset(Dataset):
    def __init__(self, data_root, known_classes, max_frames=300, is_train=True, split="train"):
        self.data_root = data_root
        self.known_classes = sorted(known_classes)
        self.known_class_set = set(known_classes)
        self.max_frames = max_frames
        self.is_train = is_train
        self.split = split

        # 1. 找出 YOLO 標記好的 .npy 檔案
        self.files = sorted(glob.glob(os.path.join(data_root, "*.npy")))
        self.samples = []

        for path in self.files:
            action_id = self._get_action_id_from_filename(path)
            subject_id = self._get_subject_id_from_filename(path)

            # 依據 NTU 規則切分 train / val (P001~P030 為訓練集)
            if split == "train" and subject_id > 30:
                continue
            if split == "val" and subject_id <= 30:
                continue

            # Open Set 標籤指派
            if action_id in self.known_class_set:
                label = self.known_classes.index(action_id)
                is_unknown = 0
            else:
                label = -1
                is_unknown = 1

            if is_train and is_unknown == 1:
                continue

            self.samples.append((path, label, is_unknown))

        print(f"[NTU YOLO Dataset] split={split}, is_train={is_train}, samples={len(self.samples)}")

    def _get_action_id_from_filename(self, path):
        """
        修正版：從檔名精準抓取 A 後面的 3 位數字 Action ID
        範例：S001C001P001R001A013_rgb.npy -> 13
        """
        filename = os.path.basename(path)
        try:
            # 找到 'A' 的位置
            a_idx = filename.find('A')
            if a_idx == -1:
                raise ValueError(f"檔名中找不到 'A': {filename}")
            
            # 確保 'A' 後面至少有 3 個字元
            action_str = filename[a_idx + 1 : a_idx + 4]
            return int(action_str)
        except Exception as e:
            raise RuntimeError(f"解析 Action ID 失敗！錯誤檔名: {filename}。錯誤訊息: {e}")

    def _get_subject_id_from_filename(self, path):
        """
        修正版：從檔名精準抓取 P 後面的 3 位數字 Subject ID
        範例：S001C001P003R002A013_rgb.npy -> 3
        """
        filename = os.path.basename(path)
        try:
            # 找到 'P' 的位置
            p_idx = filename.find('P')
            if p_idx == -1:
                raise ValueError(f"檔名中找不到 'P': {filename}")
            
            # 確保 'P' 後面至少有 3 個字元
            subject_str = filename[p_idx + 1 : p_idx + 4]
            return int(subject_str)
        except Exception as e:
            raise RuntimeError(f"解析 Subject ID 失敗！錯誤檔名: {filename}。錯誤訊息: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, is_unknown = self.samples[idx]

        # 2. 直接用 np.load 讀取你用 yolo26 轉出來的 2D 骨架 [2, T, 17]
        skeleton = np.load(path)
        skeleton = self.fix_frames(skeleton, self.max_frames)

        return (
            torch.tensor(skeleton, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(is_unknown, dtype=torch.long)
        )

    def fix_frames(self, skeleton, max_frames):
        C, T, V = skeleton.shape
        if T >= max_frames:
            skeleton = skeleton[:, :max_frames, :]
        else:
            pad = np.zeros((C, max_frames - T, V), dtype=np.float32)
            skeleton = np.concatenate([skeleton, pad], axis=1)
        return skeleton