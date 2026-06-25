import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, auc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from ntu_normalize import normalize_skeleton_batch, get_valid_mask

# 引用您的 NTU 專用模組與模型定義
from ntu_skeleton_dataset import NTUSkeletonDataset
from ST_CROSR import ST_CROSR

# 解決 matplotlib 繪圖時的中文字體問題
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

center_joint_idx = 11

if __name__ == '__main__':
    # 檢查運算裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 使用運算裝置: {device}")

    # ==========================================
    # 1. 參數設定與模型載入
    # ==========================================
    # ⚠️ 必須與 train_ntu_crosr.py 訓練時的名單完全一致[cite: 8]
    my_known_actions = [
        1, 2, 3, 4, 5, 6,
        8, 9, 11, 12,
        14, 15, 16, 17, 18, 19, 20, 21,
        23, 25,
        28, 29, 30, 32, 33, 34, 37,
        41, 44, 45, 46, 47, 49
    ]
    num_classes = len(my_known_actions)

    model = ST_CROSR(num_known_classes=num_classes, num_nodes=17, target_frames=300).to(device)
    
    weight_path = 'checkpoints_20260602_2237/best_val.pth'
    if os.path.exists(weight_path):
        # 1. 先載入完整的 checkpoint 字典
        checkpoint = torch.load(weight_path, map_location=device)
        
        # 2. 從字典中提取出真正屬於模型權重的 'model_state_dict'
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 成功從 Checkpoint 載入權重：{weight_path}")
        else:
            # 萬一以後你只存權重檔，這邊做個保險
            model.load_state_dict(checkpoint)
            print(f"✅ 成功載入權重檔案：{weight_path}")
            
        model.eval()
    else:
        print(f"❌ 找不到權重檔案 {weight_path}")
        exit()

    # ==========================================
    # 2. 準備資料載入器 (NTU 版本)
    # ==========================================
    data_root = r".\NTU60\nturgb+d_yolo_skeletons" # 請確認你的 .skeleton 檔案存放路徑[cite: 7]
    
    # 訓練集 (用於計算已知類別的特徵中心點)
    train_dataset = NTUSkeletonDataset(data_root, my_known_actions, is_train=True, split="train")
    # 驗證集 (包含已知與未知動作)
    val_dataset = NTUSkeletonDataset(data_root, my_known_actions, is_train=False, split="val") 
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # ==========================================
    # 🎯 步驟 A：計算已知動作的特徵中心點 (Centroids)
    # ==========================================
    latent_dim = 256
    centroids = torch.zeros(num_classes, latent_dim).to(device)
    class_counts = torch.zeros(num_classes).to(device)

    print("📍 正在計算 NTU 已知動作的特徵中心點...")
    with torch.no_grad():
        for inputs, labels, _ in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 👑 骨架中心化 (NTU 改用 Spine Base, 索引 0)[cite: 2, 8]
            # NTU 是 3 通道 (x, y, z)，所以取 :3
            # base_pos = inputs[:, :3, :, 1:2].clone() 
            # inputs[:, :3, :, :] = inputs[:, :3, :, :] - base_pos
            # with torch.no_grad():
            #         std = torch.std(inputs) + 1e-6
            #         inputs = inputs / std

            inputs = normalize_skeleton_batch(inputs, center_joint_idx)
            
            _, _, z, _ = model(inputs)
            
            for i in range(num_classes):
                mask = (labels == i)
                if mask.sum() > 0:
                    centroids[i] += z[mask].sum(dim=0)
                    class_counts[i] += mask.sum()

    # 計算平均中心點並正規化 (用於 Cosine Similarity)[cite: 9]
    centroids = centroids / (class_counts.unsqueeze(1) + 1e-6)
    centroids_norm = F.normalize(centroids, p=2, dim=1) 

    # ==========================================
    # 🎯 步驟 B：正式測試與雙重雷達評分
    # ==========================================
    y_true = []
    y_scores_combined = []
    y_scores_dist = []
    y_scores_mse = []

    correct_cls_preds = 0
    total_known_samples = 0

    print("🚨 啟動 NTU ST-CROSR 雙重異常雷達 (Masked MSE x Cosine Dist)...")
    with torch.no_grad():
        for inputs, labels, is_unknown in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # # 抓出原始資料的有效遮罩 (處理補零部分)[cite: 2, 8]
            # valid_mask = (inputs != 0.0).float()

            # # 骨架中心化 (對齊座標原點)[cite: 8]
            # base_pos = inputs[:, :3, :, 1:2].clone() 
            # inputs[:, :3, :, :] = inputs[:, :3, :, :] - base_pos
            # with torch.no_grad():
            #         std = torch.std(inputs) + 1e-6
            #         inputs = inputs / std
            valid_mask = get_valid_mask(inputs)
            inputs = normalize_skeleton_batch(inputs, center_joint_idx)
            logits, recon_x, z, _ = model(inputs)
            
            # 1. 計算已知動作的分類準確率[cite: 9]
            preds = torch.argmax(logits, dim=1)
            known_mask = (is_unknown == 0).to(device)
            if known_mask.sum() > 0:
                correct_cls_preds += (preds[known_mask] == labels[known_mask]).sum().item()
                total_known_samples += known_mask.sum().item()
            
            # 2. 雷達 A：遮罩型重建誤差 (Masked MSE)[cite: 8]
            # 同時考量 x, y, z 三軸誤差
            squared_diff = (recon_x - inputs) ** 2
            masked_diff = squared_diff * valid_mask
            mse_scores = torch.sum(masked_diff, dim=[1, 2, 3]) / (torch.sum(valid_mask, dim=[1, 2, 3]) + 1e-6)
            
            # 3. 雷達 B：特徵空間距離 (Cosine Distance)[cite: 9]
            z_norm = F.normalize(z, p=2, dim=1)
            cos_sim = torch.mm(z_norm, centroids_norm.t()) 
            max_sim, _ = torch.max(cos_sim, dim=1)
            dist_scores = 1.0 - max_sim 
            
            # 🟢【核心修改】迴圈內不進行高風險的線性乘法，只純粹收集原始數據
            y_scores_dist.extend(dist_scores.cpu().numpy())
            y_scores_mse.extend(mse_scores.cpu().numpy())
            y_true.extend(is_unknown.numpy())

    # ==========================================
    # 🎯 步驟 C：效能結算與數據診斷
    # ==========================================
    y_scores_dist = np.array(y_scores_dist)
    y_scores_mse = np.array(y_scores_mse)
    y_true = np.array(y_true)

    # 1. 用對數平滑強效馴服 YOLO 像素級別的抖動離群值
    y_scores_mse_log = np.log1p(y_scores_mse)

    # 2. 將兩者縮放到絕對平等的 [0, 1] 物理區間
    dist_min, dist_max = y_scores_dist.min(), y_scores_dist.max()
    mse_min, mse_max = y_scores_mse_log.min(), y_scores_mse_log.max()

    norm_dist = (y_scores_dist - dist_min) / (dist_max - dist_min + 1e-8)
    norm_mse = (y_scores_mse_log - mse_min) / (mse_max - mse_min + 1e-8)

    # 3. 黃金加權融合：鑑於 Cosine (0.7577) 表現優於 Reconstruction (0.7031)
    # 賦予表現更佳的 Cosine 雷達 60% 的話語權，重建雷達 40%
    y_scores_combined = norm_dist * 0.6 + norm_mse * 0.4
    cls_accuracy = (correct_cls_preds / total_known_samples * 100) if total_known_samples > 0 else 0.0

    print(f"\n🎯 [任務報告] 已知動作分類準確率 (Top-1): {cls_accuracy:.2f}%")
    
    # 計算各項 AUROC[cite: 9]
    auc_dist = roc_auc_score(y_true, y_scores_dist)
    auc_mse = roc_auc_score(y_true, y_scores_mse)
    auc_combined = roc_auc_score(y_true, y_scores_combined)

    print(f"🔎 [數值診斷] Cosine 距離平均值: {np.mean(y_scores_dist):.4f}")
    print(f"🔎 [數值診斷] Masked MSE 平均值: {np.mean(y_scores_mse):.4f}")

    print(f"\n📊 終極防禦雷達：戰力評估 (AUROC)")
    print(f"====================================")
    print(f"1. 特徵距離 (Cosine) AUROC : {auc_dist:.4f}")
    print(f"2. 重建誤差 (Masked) AUROC : {auc_mse:.4f}")
    print(f"3. 兩者融合 (Combined)   AUROC : {auc_combined:.4f}")
    print(f"====================================")

    # --- 找尋最佳決策閥值 (Youden's J statistic) ---
    fpr, tpr, thresholds = roc_curve(y_true, y_scores_combined)
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    print(f"💡 系統建議的最佳報警閥值 (Threshold): {optimal_threshold:.4f}")


    # ==========================================
    # 🎯 步驟 D：圖表生成
    # ==========================================
    # 圖表 1：異常分數分佈圖[cite: 9]
    plt.figure(figsize=(10, 6))
    scores_known = [s for s, t in zip(y_scores_combined, y_true) if t == 0]
    scores_unknown = [s for s, t in zip(y_scores_combined, y_true) if t == 1]
    sns.kdeplot(scores_known, fill=True, label='已知動作 (正常)', color='navy', alpha=0.5)
    sns.kdeplot(scores_unknown, fill=True, label='未知動作 (異常)', color='crimson', alpha=0.5)
    plt.axvline(x=optimal_threshold, color='green', linestyle='--', label=f'建議閥值 ({optimal_threshold:.2f})')
    plt.title('NTU 異常分數分佈 (Combined Score Distribution)')
    plt.legend()
    plt.savefig('ntu_chart_1_distribution.png')
    plt.close()

    # 圖表 2：PR 曲線[cite: 9]
    precision, recall, _ = precision_recall_curve(y_true, y_scores_combined)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', label=f'PR Curve (AUC = {auc(recall, precision):.3f})')
    plt.title('Precision-Recall Curve (NTU OSR)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig('ntu_chart_2_pr_curve.png')
    plt.close()

    # 圖表 3：混淆矩陣[cite: 9]
    y_pred = [1 if s >= optimal_threshold else 0 for s in y_scores_combined]
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['預測已知', '預測異常'], yticklabels=['實際已知', '實際異常'])
    plt.title(f'混淆矩陣 (Threshold = {optimal_threshold:.2f})')
    plt.savefig('ntu_chart_3_confusion_matrix.png')
    plt.close()

    print("\n📸 進階效能評估報告已生成 (ntu_chart_1~3.png)")
    
    # 🟢 在 test_ntu_dataset.py 結尾塞入這行，導出校正參數
    meta_params = {
        "centroids_norm": centroids_norm.cpu(),
        "dist_min": float(y_scores_dist.min()), "dist_max": float(y_scores_dist.max()),
        "mse_min": float(y_scores_mse_log.min()), "mse_max": float(y_scores_mse_log.max()),
        "threshold": 0.2456
    }
    torch.save(meta_params, "radar_meta_params.pth")
    print("✅ 雷達校正參數已成功導出為 radar_meta_params.pth！")