import os
import copy
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# =========================
# 載入你寫好的 PyTorch 正規化模組
# =========================
from ntu_normalize import normalize_skeleton_batch


# =========================
# 你只需要改這裡
# =========================
SKELETON_PATH = "S001C001P001R001A001.skeleton"   # 你的 NTU skeleton 檔案

# 輸出設定：原圖
OUTPUT_DIR_ORIG = "skeleton_frames_orig"
GIF_NAME_ORIG = "ntu_skeleton_orig.gif"

# 輸出設定：正規化後
OUTPUT_DIR_NORM = "skeleton_frames_norm"
GIF_NAME_NORM = "ntu_skeleton_norm.gif"

SAVE_GIF = True                       # 是否輸出 GIF
MAX_FRAMES = 200                      # 最多畫幾張，None = 全部畫完
# =========================


# NTU RGB+D 25 joints 連線規則
NTU_BONES = [
    (1, 2), (2, 21), (21, 3), (3, 4),          # 身體、脖子、頭
    (21, 5), (5, 6), (6, 7), (7, 8),           # 左手臂
    (7, 22), (22, 23),                         # 左手
    (21, 9), (9, 10), (10, 11), (11, 12),      # 右手臂
    (11, 24), (24, 25),                        # 右手
    (1, 13), (13, 14), (14, 15), (15, 16),     # 左腳
    (1, 17), (17, 18), (18, 19), (19, 20)      # 右腳
]

NTU_BONES = [(a - 1, b - 1) for a, b in NTU_BONES]


def read_ntu_skeleton(file_path):
    """讀取 NTU RGB+D skeleton 檔案。"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"找不到檔案：{file_path}")

    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    idx = 0
    num_frames = int(float(lines[idx].strip()))
    idx += 1
    frames = []

    for frame_idx in range(num_frames):
        num_bodies = int(float(lines[idx].strip()))
        idx += 1
        bodies = []
        for body_idx in range(num_bodies):
            idx += 1
            num_joints = int(float(lines[idx].strip()))
            idx += 1
            joints = []
            for joint_idx in range(num_joints):
                values = [float(v) for v in lines[idx].strip().split()]
                idx += 1
                joints.append(values)

            joints = np.array(joints)
            if joints.shape[0] == 25:
                bodies.append(joints)
        frames.append(bodies)

    return frames


def apply_pytorch_normalization(original_frames):
    """
    資料轉換橋接：複製一份 frames 轉為 [B, C, T, V] 的 PyTorch Tensor，
    呼叫 ntu_normalize.py 進行正規化，回傳全新的 frames 結構。
    """
    # 複製一份資料，避免改動到正規化前的原圖
    frames = copy.deepcopy(original_frames)
    
    T = len(frames)
    if T == 0:
        return frames

    max_bodies = max(len(bodies) for bodies in frames)
    if max_bodies == 0:
        return frames

    # 建立 numpy 陣列 [B, C, T, V] -> C=3 (x,y,z), V=25 (joints)
    inputs_np = np.zeros((max_bodies, 3, T, 25), dtype=np.float32)

    for t in range(T):
        for m, body in enumerate(frames[t]):
            inputs_np[m, 0, t, :] = body[:, 0]  # x
            inputs_np[m, 1, t, :] = body[:, 1]  # y
            inputs_np[m, 2, t, :] = body[:, 2]  # z

    inputs_tensor = torch.tensor(inputs_np)

    print("呼叫 ntu_normalize.normalize_skeleton_batch 進行 PyTorch 正規化...")
    normalized_tensor = normalize_skeleton_batch(inputs_tensor, center_joint_idx=1)
    normalized_np = normalized_tensor.numpy()

    for t in range(T):
        for m, body in enumerate(frames[t]):
            body[:, 0] = normalized_np[m, 0, t, :]
            body[:, 1] = normalized_np[m, 1, t, :]
            body[:, 2] = normalized_np[m, 2, t, :]

    return frames


def get_xy_from_joints(joints, use_color_coord=True):
    """取得要畫圖的 x, y"""
    if use_color_coord:
        x = joints[:, 5]       
        y = -joints[:, 6]      
    else:
        x = joints[:, 0]       
        y = joints[:, 1]       
    return x, y


def draw_one_frame(joints, save_path=None, title="NTU Skeleton", use_color_coord=True):
    x, y = get_xy_from_joints(joints, use_color_coord)
    plt.figure(figsize=(7, 9))

    for a, b in NTU_BONES:
        plt.plot([x[a], x[b]], [y[a], y[b]], linewidth=2)

    plt.scatter(x, y, s=45)

    for i, (xi, yi) in enumerate(zip(x, y), start=1):
        offset = 0.05 if not use_color_coord else 4
        plt.text(xi + offset, yi + offset, str(i), fontsize=8)

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def save_frames_as_images(frames, output_dir, use_color_coord, prefix=""):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for frame_idx, bodies in enumerate(frames):
        if MAX_FRAMES is not None and count >= MAX_FRAMES:
            break
        if len(bodies) == 0:
            continue

        joints = bodies[0]
        save_path = output_dir / f"frame_{frame_idx:04d}.png"
        draw_one_frame(joints, save_path=save_path, title=f"{prefix} Frame {frame_idx}", use_color_coord=use_color_coord)
        count += 1

    print(f"[{prefix}] 已輸出 {count} 張骨架圖到：{output_dir}")


def save_as_gif(frames, gif_path, use_color_coord, prefix=""):
    valid_frames = []
    for bodies in frames:
        if len(bodies) > 0:
            valid_frames.append(bodies[0])
        if MAX_FRAMES is not None and len(valid_frames) >= MAX_FRAMES:
            break

    if len(valid_frames) == 0:
        print(f"[{prefix}] 沒有可用的 skeleton frame，無法輸出 GIF。")
        return

    fig, ax = plt.subplots(figsize=(7, 9))
    all_x = []
    all_y = []

    for joints in valid_frames:
        x, y = get_xy_from_joints(joints, use_color_coord)
        all_x.extend(x)
        all_y.extend(y)

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    margin = 0.2 if not use_color_coord else 50

    def update(frame_idx):
        ax.clear()
        joints = valid_frames[frame_idx]
        x, y = get_xy_from_joints(joints, use_color_coord)

        for a, b in NTU_BONES:
            ax.plot([x[a], x[b]], [y[a], y[b]], linewidth=2)

        ax.scatter(x, y, s=45)

        for i, (xi, yi) in enumerate(zip(x, y), start=1):
            offset = 0.05 if not use_color_coord else 4
            ax.text(xi + offset, yi + offset, str(i), fontsize=8)

        ax.set_title(f"{prefix} Frame {frame_idx}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

    anim = FuncAnimation(fig, update, frames=len(valid_frames), interval=80)
    anim.save(gif_path, writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"[{prefix}] GIF 已輸出：{gif_path}")


def main():
    # 1. 讀取原始骨架資料
    orig_frames = read_ntu_skeleton(SKELETON_PATH)
    print(f"總 frame 數：{len(orig_frames)}")

    # 2. 產生正規化後的骨架資料 (原資料 orig_frames 不會被改動)
    norm_frames = apply_pytorch_normalization(orig_frames)

    # 3. 輸出原圖 (使用 colorX, colorY 繪製，所以 use_color_coord=True)
    print("\n--- 開始輸出正規化前的結果 ---")
    save_frames_as_images(orig_frames, OUTPUT_DIR_ORIG, use_color_coord=True, prefix="Original")
    if SAVE_GIF:
        save_as_gif(orig_frames, GIF_NAME_ORIG, use_color_coord=True, prefix="Original")

    # 4. 輸出正規化後的圖 (必須使用 3D 的 x, y 繪製，所以 use_color_coord=False)
    print("\n--- 開始輸出正規化後的結果 ---")
    save_frames_as_images(norm_frames, OUTPUT_DIR_NORM, use_color_coord=False, prefix="Normalized")
    if SAVE_GIF:
        save_as_gif(norm_frames, GIF_NAME_NORM, use_color_coord=False, prefix="Normalized")


if __name__ == "__main__":
    main()