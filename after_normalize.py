from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# -------------------
# 我拉在我自己電腦跑的
# 但我拉不過來 結果
# 有興趣 自己拉去自己的電腦跑
# 她normalize後 把2 那個點 放在 原點 就是人正中間 肚臍附近那個點
# 我看 他比例有點不太一樣 不是一模一樣  但是90%相似



# =========================
# 你只需要改這裡
# =========================
SKELETON_PATH = "S001C001P001R001A001.skeleton"

OUTPUT_DIR = "exact_batch_normalize_compare"
SAVE_GIF = True
GIF_NAME = "exact_batch_normalize_compare.gif"

MAX_FRAMES = 200
CENTER_JOINT_IDX = 1
EPS = 1e-6

# 模擬訓練時的 batch。
# 如果你只看一個檔案，B 就會是 1。
# 這樣流程和 normalize_skeleton_batch() 完全一樣，
# 只是 batch 裡只有一筆資料。
BATCH_REPEAT = 1
# =========================


NTU_BONES = [
    (1, 2), (2, 21), (21, 3), (3, 4),
    (21, 5), (5, 6), (6, 7), (7, 8),
    (7, 22), (22, 23),
    (21, 9), (9, 10), (10, 11), (11, 12),
    (11, 24), (24, 25),
    (1, 13), (13, 14), (14, 15), (15, 16),
    (1, 17), (17, 18), (18, 19), (19, 20)
]
NTU_BONES = [(a - 1, b - 1) for a, b in NTU_BONES]


def get_valid_mask(inputs):
    """
    和你的訓練程式一樣：
    取得有效資料遮罩。
    """
    return (inputs != 0.0).float()


def normalize_skeleton_batch(inputs, center_joint_idx=1, eps=1e-6):
    """
    這裡完全照你的 normalize 函式。

    inputs shape:
        [B, C, T, V]
    """

    inputs = inputs.clone()

    center_pos = inputs[:, :3, :, center_joint_idx:center_joint_idx + 1].clone()
    inputs[:, :3, :, :] = inputs[:, :3, :, :] - center_pos

    std = torch.std(inputs) + eps
    inputs = inputs / std

    return inputs


def read_ntu_skeleton(file_path):
    """
    讀取 NTU RGB+D skeleton 檔案。

    回傳：
    frames[frame_idx][body_idx] = numpy array, shape = (25, 12)

    joint 欄位：
    x y z depthX depthY colorX colorY orientationW orientationX orientationY orientationZ trackingState
    """

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
            idx += 1  # body info

            num_joints = int(float(lines[idx].strip()))
            idx += 1

            joints = []

            for joint_idx in range(num_joints):
                values = [float(v) for v in lines[idx].strip().split()]
                idx += 1
                joints.append(values)

            joints = np.array(joints, dtype=np.float32)

            if joints.shape[0] == 25:
                bodies.append(joints)

        frames.append(bodies)

    return frames


def frames_to_tensor(frames):
    """
    把 NTU skeleton 轉成你的模型輸入格式：

    原始讀進來：
        sequence shape = [T, V, C]

    模型需要：
        inputs shape = [B, C, T, V]

    這裡只取第一個人，只取 3D x, y, z。
    """

    sequence = []

    for bodies in frames:
        if len(bodies) == 0:
            continue

        joints = bodies[0]

        # 只取 3D x, y, z
        xyz = joints[:, 0:3]
        sequence.append(xyz)

        if MAX_FRAMES is not None and len(sequence) >= MAX_FRAMES:
            break

    if len(sequence) == 0:
        raise RuntimeError("找不到可用的 skeleton frame。")

    # [T, V, C]
    sequence = np.array(sequence, dtype=np.float32)

    # [T, V, C] -> [C, T, V]
    sequence = np.transpose(sequence, (2, 0, 1))

    # [C, T, V] -> [1, C, T, V]
    inputs = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

    # 如果想模擬 batch，可以重複同一筆資料
    # BATCH_REPEAT = 1 時就是只看一個 sample
    if BATCH_REPEAT > 1:
        inputs = inputs.repeat(BATCH_REPEAT, 1, 1, 1)

    return inputs


def tensor_to_sequence(inputs, batch_idx=0):
    """
    把 [B, C, T, V] 轉回畫圖用的 [T, V, C]
    """

    sample = inputs[batch_idx]              # [C, T, V]
    sample = sample.permute(1, 2, 0)        # [T, V, C]
    return sample.detach().cpu().numpy()


def draw_skeleton(ax, xyz, title, xlabel, ylabel):
    """
    xyz shape = [V, 3]
    """

    x = xyz[:, 0]
    y = xyz[:, 1]

    for a, b in NTU_BONES:
        ax.plot([x[a], x[b]], [y[a], y[b]], linewidth=2)

    ax.scatter(x, y, s=35)

    for i, (xi, yi) in enumerate(zip(x, y), start=1):
        ax.text(xi, yi, str(i), fontsize=7)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)


def save_compare_images(raw_sequence, normalized_sequence, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in range(len(raw_sequence)):
        raw_xyz = raw_sequence[frame_idx]
        norm_xyz = normalized_sequence[frame_idx]

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        draw_skeleton(
            axes[0],
            raw_xyz,
            title=f"Before Normalize - Frame {frame_idx}",
            xlabel="Raw 3D X",
            ylabel="Raw 3D Y"
        )

        draw_skeleton(
            axes[1],
            norm_xyz,
            title=f"After Exact Batch Normalize - Frame {frame_idx}",
            xlabel="Normalized 3D X",
            ylabel="Normalized 3D Y"
        )

        fig.suptitle("NTU Skeleton: Before vs Exact normalize_skeleton_batch()")
        fig.tight_layout()

        save_path = output_dir / f"exact_compare_frame_{frame_idx:04d}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"已輸出 {len(raw_sequence)} 張對照圖到：{output_dir}")


def save_compare_gif(raw_sequence, normalized_sequence, gif_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    raw_x = raw_sequence[:, :, 0].reshape(-1)
    raw_y = raw_sequence[:, :, 1].reshape(-1)

    norm_x = normalized_sequence[:, :, 0].reshape(-1)
    norm_y = normalized_sequence[:, :, 1].reshape(-1)

    raw_margin = 0.2
    norm_margin = 0.5

    raw_xlim = (raw_x.min() - raw_margin, raw_x.max() + raw_margin)
    raw_ylim = (raw_y.min() - raw_margin, raw_y.max() + raw_margin)

    norm_xlim = (norm_x.min() - norm_margin, norm_x.max() + norm_margin)
    norm_ylim = (norm_y.min() - norm_margin, norm_y.max() + norm_margin)

    def update(frame_idx):
        for ax in axes:
            ax.clear()

        raw_xyz = raw_sequence[frame_idx]
        norm_xyz = normalized_sequence[frame_idx]

        draw_skeleton(
            axes[0],
            raw_xyz,
            title=f"Before Normalize - Frame {frame_idx}",
            xlabel="Raw 3D X",
            ylabel="Raw 3D Y"
        )

        draw_skeleton(
            axes[1],
            norm_xyz,
            title=f"After Exact Batch Normalize - Frame {frame_idx}",
            xlabel="Normalized 3D X",
            ylabel="Normalized 3D Y"
        )

        axes[0].set_xlim(raw_xlim)
        axes[0].set_ylim(raw_ylim)

        axes[1].set_xlim(norm_xlim)
        axes[1].set_ylim(norm_ylim)

        fig.suptitle("NTU Skeleton: Before vs Exact normalize_skeleton_batch()")
        fig.tight_layout()

    anim = FuncAnimation(
        fig,
        update,
        frames=len(raw_sequence),
        interval=80
    )

    anim.save(gif_name, writer=PillowWriter(fps=12))
    plt.close(fig)

    print(f"已輸出 GIF：{gif_name}")


def main():
    frames = read_ntu_skeleton(SKELETON_PATH)

    raw_inputs = frames_to_tensor(frames)

    print("模型輸入格式 raw_inputs.shape =", tuple(raw_inputs.shape))
    print("也就是 [B, C, T, V]")
    print()

    normalized_inputs = normalize_skeleton_batch(
        raw_inputs,
        center_joint_idx=CENTER_JOINT_IDX,
        eps=EPS
    )

    # 轉回畫圖格式
    raw_sequence = tensor_to_sequence(raw_inputs, batch_idx=0)
    normalized_sequence = tensor_to_sequence(normalized_inputs, batch_idx=0)

    print("這次使用的 normalize 函式和你貼的一樣：")
    print("center_pos = inputs[:, :3, :, center_joint_idx:center_joint_idx + 1].clone()")
    print("inputs[:, :3, :, :] = inputs[:, :3, :, :] - center_pos")
    print("std = torch.std(inputs) + eps")
    print("inputs = inputs / std")
    print()
    print("std =", float(torch.std(raw_inputs.clone()[:, :3, :, :] - raw_inputs.clone()[:, :3, :, CENTER_JOINT_IDX:CENTER_JOINT_IDX + 1]) + EPS))
    print()

    save_compare_images(raw_sequence, normalized_sequence, OUTPUT_DIR)

    if SAVE_GIF:
        save_compare_gif(raw_sequence, normalized_sequence, GIF_NAME)


if __name__ == "__main__":
    main()
