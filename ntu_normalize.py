import torch


def get_valid_mask(inputs):
    """
    取得有效資料遮罩。
    原本 NTU skeleton 補零的位置會是 0，
    Masked MSE 要用這個 mask 排除補零區域。
    """
    return (inputs != 0.0).float()


def normalize_skeleton_batch(inputs, center_joint_idx=1, eps=1e-6):
    """
    NTU skeleton batch 標準化流程：

    1. 以指定關節點作為中心點
    2. 所有骨架點減掉中心點座標
    3. 除以整個 batch 的標準差

    inputs shape:
        [B, C, T, V]
        B = batch size
        C = channel，一般是 x, y, z
        T = frame 數
        V = joint 數
    """
    inputs = inputs.clone()

    center_pos = inputs[:, :2, :, center_joint_idx:center_joint_idx + 1].clone()
    inputs[:, :2, :, :] = inputs[:, :2, :, :] - center_pos

    std = torch.std(inputs) + eps
    inputs = inputs / std

    return inputs

# # 骨架中心化 (對齊座標原點)[cite: 8]
            # base_pos = inputs[:, :3, :, 1:2].clone() 
            # inputs[:, :3, :, :] = inputs[:, :3, :, :] - base_pos
            # with torch.no_grad():
            #         std = torch.std(inputs) + 1e-6
            #         inputs = inputs / std