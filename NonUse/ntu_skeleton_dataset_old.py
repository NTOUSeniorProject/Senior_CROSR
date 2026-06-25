import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class NTUSkeletonDataset(Dataset):
    def __init__(
        self,
        data_root,
        known_classes,
        max_frames=300,
        is_train=True,
        split="train"
    ):
        """
        data_root:
            放 .skeleton 檔案的資料夾

        known_classes:
            你要當作「已知動作」的 NTU action 編號
            例如 [1, 2, 3, 4, 5]

        max_frames:
            每個動作統一成幾個 frame

        is_train:
            True  = 訓練模式，只拿已知動作
            False = 驗證模式，已知和未知都會保留

        split:
            train 或 val
        """

        self.data_root = data_root
        self.known_classes = sorted(known_classes)
        self.known_class_set = set(known_classes)
        self.max_frames = max_frames
        self.is_train = is_train
        self.split = split

        # 找出資料夾底下所有 .skeleton 檔案
        self.files = sorted(glob.glob(os.path.join(data_root, "*.skeleton")))

        self.samples = []

        for path in self.files:
            action_id = self._get_action_id_from_filename(path)
            subject_id = self._get_subject_id_from_filename(path)

            # 簡單切分 train / val
            # P001 ~ P030 當 train
            # P031 以上當 val
            if split == "train" and subject_id > 30:
                continue

            if split == "val" and subject_id <= 30:
                continue

            # 判斷是不是已知類別
            if action_id in self.known_class_set:
                label = self.known_classes.index(action_id)
                is_unknown = 0
            else:
                label = -1
                is_unknown = 1

            # 訓練階段只拿已知動作
            if is_train and is_unknown == 1:
                continue

            self.samples.append((path, label, is_unknown))

        print(f"[NTU Dataset] split={split}, is_train={is_train}, samples={len(self.samples)}")

        if len(self.samples) == 0:
            raise RuntimeError(
                f"\n找不到可用的 skeleton 檔案。\n"
                f"目前 data_root = {data_root}\n\n"
                f"請確認：\n"
                f"1. 你有沒有解壓縮 zip\n"
                f"2. data_root 是否指到真正有 .skeleton 的資料夾\n"
                f"3. NTU60 資料夾裡面是不是還有下一層資料夾\n"
            )

    def _get_action_id_from_filename(self, path):
        """
        從檔名抓 action id

        example:
        S001C002P003R002A013.skeleton

        A013 = action id 13
        """
        filename = os.path.basename(path)
        action_str = filename.split("A")[-1].split(".")[0]
        return int(action_str)

    def _get_subject_id_from_filename(self, path):
        """
        從檔名抓 subject id

        example:
        S001C002P003R002A013.skeleton

        P003 = subject id 3
        """
        filename = os.path.basename(path)
        subject_str = filename.split("P")[-1].split("R")[0]
        return int(subject_str)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, is_unknown = self.samples[idx]

        skeleton = self.read_skeleton_file(path)
        skeleton = self.fix_frames(skeleton, self.max_frames)

        return (
            torch.tensor(skeleton, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(is_unknown, dtype=torch.long)
        )

    def read_skeleton_file(self, path):
        """
        讀取 NTU .skeleton 檔案 (已移除 Z 軸)。

        回傳 shape:
            [2, T, 25]

        2  = x, y
        T  = frame 數
        25 = 人體關節點數
        """

        with open(path, "r") as f:
            lines = f.readlines()

        idx = 0

        # 第一行：總 frame 數
        num_frames = int(lines[idx].strip())
        idx += 1

        frames = []

        for _ in range(num_frames):
            # 這一幀裡有幾個人
            num_bodies = int(lines[idx].strip())
            idx += 1

            # 如果這一幀沒有人，補 2D 的 0
            if num_bodies == 0:
                frames.append([[0.0, 0.0] for _ in range(25)])  #  改成 2 個 0 
                continue

            # 這裡先只取第一個人
            # body info 先跳過
            idx += 1

            # 這個人有幾個 joint
            num_joints = int(lines[idx].strip())
            idx += 1

            joints = []

            for _ in range(num_joints):
                joint_info = lines[idx].strip().split()
                idx += 1

                # NTU skeleton 每個 joint 的前 2 個數值是 x, y (捨棄 z)
                x = float(joint_info[0])
                y = float(joint_info[1])

                joints.append([x, y])  # 這裡只丟 x, y

            # 正常應該是 25 個 joint
            # 如果不是 25，做保護處理
            if len(joints) < 25:
                while len(joints) < 25:
                    joints.append([0.0, 0.0])  # 改為 2D 補零
            elif len(joints) > 25:
                joints = joints[:25]

            frames.append(joints)

            # 如果同一幀還有其他人，先跳過
            for _ in range(num_bodies - 1):
                # body info
                idx += 1

                extra_num_joints = int(lines[idx].strip())
                idx += 1

                # 跳過其他人的 joints
                idx += extra_num_joints

        skeleton = np.array(frames, dtype=np.float32)

        # [T, 25, 3] -> [3, T, 25]
        skeleton = np.transpose(skeleton, (2, 0, 1))

        return skeleton

    def fix_frames(self, skeleton, max_frames):
        """
        把不同長度的 skeleton 統一成 max_frames。
        """
        C, T, V = skeleton.shape  # 此時 C 會自動抓到 2

        if T >= max_frames:
            skeleton = skeleton[:, :max_frames, :]
        else:
            pad = np.zeros((C, max_frames - T, V), dtype=np.float32)
            skeleton = np.concatenate([skeleton, pad], axis=1)

        return skeleton