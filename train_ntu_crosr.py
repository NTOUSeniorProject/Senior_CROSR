import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from Functions.ntu_normalize import normalize_skeleton_batch, get_valid_mask

from datetime import datetime
import os
from pathlib import Path

from Functions.ntu_skeleton_dataset import NTUSkeletonDataset
from Functions.ST_CROSR import ST_CROSR
from Functions.center_loss import CenterLoss
import time
# ============================================================
# ⚙️ 核心參數設定區 (在此調整即可)
# ============================================================
CONFIG = {
    # --- Checkpoint 設定 ---
    "auto_resume": False,          
    "resume_checkpoint": None,     
    
    # --- 資料路徑與定義 ---
    "data_root": r".\NTU60\nturgb+d_yolo_skeletons",       # ⚠️ 指向你 yolo26 轉出的 npy 資料夾
    "known_actions": [
        1, 2, 3, 4, 5, 6,
        8, 9, 11, 12,
        14, 15, 16, 17, 18, 19, 20, 21,
        23, 25,
        28, 29, 30, 32, 33, 34, 37,
        41, 44, 45, 46, 47, 49
    ],
    "max_frames": 300,             
    "num_nodes": 17,                               # ⚠️ 核心更正：由 25 改為 17
    "feat_dim": 256,               
    
    # --- 訓練超參數 ---
    "num_epochs": 200,              
    "batch_size": 32,              
    "lr_model": 0.001,             
    "weight_decay": 1e-4,          
    "lambda_center": 0.0001,  # 🟢 放行，但從 0.01 降到 0.0001
    "lr_center": 0.005,       # 🟢 【極重要】將 center 學習率從 0.5 暴降到 0.005，防止它瞬間壓扁特徵             
    
    # --- 損失函數權重 (Loss Weights) ---
    "lambda_center": 0.001,         
    "max_lambda_recon": 0.5,       
    "warmup_epochs": 10,           
    
    # --- 骨架中心化基準點 ---
    "center_joint_idx": 11,                        # ⚠️ 核心更正：改用 COCO 17點的左臀 (Index 11)
}

# ============================================================
# 🛠️ 輔助函式
# ============================================================
def find_latest_checkpoint():
    checkpoint_files = list(Path(".").glob("checkpoints_*/last.pth"))
    if len(checkpoint_files) == 0:
        return None
    latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    return str(latest_checkpoint)

def save_checkpoint(path, epoch, model, optimizer_model, optimizer_center, scheduler, scaler, 
                    best_val_accuracy, val_accuracy, avg_train_cls_loss, avg_train_recon_loss, 
                    avg_train_center_loss, known_actions, num_classes):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_model_state_dict": optimizer_model.state_dict(),
        "optimizer_center_state_dict": optimizer_center.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_accuracy": best_val_accuracy,
        "val_accuracy": val_accuracy,
        "avg_train_cls_loss": avg_train_cls_loss,
        "avg_train_recon_loss": avg_train_recon_loss,
        "avg_train_center_loss": avg_train_center_loss,
        "known_actions": known_actions,
        "num_classes": num_classes,
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer_model=None, optimizer_center=None, scheduler=None, scaler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer_model is not None and checkpoint.get("optimizer_model_state_dict") is not None:
            optimizer_model.load_state_dict(checkpoint["optimizer_model_state_dict"])
        if optimizer_center is not None and checkpoint.get("optimizer_center_state_dict") is not None:
            optimizer_center.load_state_dict(checkpoint["optimizer_center_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint.get("epoch", -1) + 1
        best_val_accuracy = checkpoint.get("best_val_accuracy", -1.0)
        print(f"✅ 成功載入 checkpoint: {path}")
        print(f"➡️ 從 Epoch {start_epoch + 1} 繼續訓練")
        print(f"🏆 目前最佳驗證準確率: {best_val_accuracy:.2f}%")
        return start_epoch, best_val_accuracy
    model.load_state_dict(checkpoint)
    print(f"✅ 成功載入舊格式模型權重: {path}")
    return 0, -1.0

# ============================================================
# 🚀 主訓練程序
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 使用運算裝置: {device}")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    cudnn.benchmark = True

    num_classes = len(CONFIG["known_actions"])
    print(f"已知動作數量: {num_classes}")

    # Checkpoint 自動管理邏輯
    resume_checkpoint = CONFIG["resume_checkpoint"]
    if CONFIG["auto_resume"] and resume_checkpoint is None:
        resume_checkpoint = find_latest_checkpoint()

    if resume_checkpoint is not None:
        checkpoint_dir = str(Path(resume_checkpoint).parent)
        print(f"🔁 偵測到 checkpoint：{resume_checkpoint}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        checkpoint_dir = f"checkpoints_{timestamp}"
        print("🆕 沒有找到 checkpoint，將從頭開始訓練")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 資料集準備
    ds_params = {
        "data_root": CONFIG["data_root"],
        "known_classes": CONFIG["known_actions"],
        "max_frames": CONFIG["max_frames"]
    }
    train_dataset = NTUSkeletonDataset(**ds_params, is_train=True, split="train")
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, 
        num_workers=0, pin_memory=torch.cuda.is_available()
    )

    val_dataset = NTUSkeletonDataset(**ds_params, is_train=False, split="val")
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, 
        num_workers=0, pin_memory=torch.cuda.is_available()
    )

    # 模型與 Loss
    model = ST_CROSR(
        num_known_classes=num_classes, 
        num_nodes=CONFIG["num_nodes"], 
        target_frames=CONFIG["max_frames"]
    ).to(device)

    criterion_center = CenterLoss(
        num_classes=num_classes, feat_dim=CONFIG["feat_dim"], use_gpu=torch.cuda.is_available()
    )
    criterion_cls, criterion_recon = nn.CrossEntropyLoss(), nn.MSELoss()

    # Optimizer 與 Scheduler
    optimizer_model = optim.Adam(model.parameters(), lr=CONFIG["lr_model"], weight_decay=CONFIG["weight_decay"])
    optimizer_center = optim.SGD(criterion_center.parameters(), lr=CONFIG["lr_center"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_model, T_max=CONFIG["num_epochs"], eta_min=1e-5)

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val_accuracy, start_epoch = -1.0, 0
    if resume_checkpoint is not None:
        start_epoch, best_val_accuracy = load_checkpoint(
            resume_checkpoint, model, optimizer_model, optimizer_center, scheduler, scaler, device
        )

    print("\n🚀 開始訓練 NTU ST-CROSR 模型！")
    total_start_time = time.perf_counter()
    
    lambda_entropy = CONFIG.get("lambda_entropy", 0.0002)
    
    # ============================================================
    # 9. Training Loop
    # ============================================================
    for epoch in range(start_epoch, CONFIG["num_epochs"]):
        model.train()
        cls_loss_sum, recon_loss_sum, center_loss_sum, entropy_loss_sum = 0.0, 0.0, 0.0, 0.0
        # 🟢【修正】讓第 1 輪也能分到基礎權重，拒絕 0 權重擺爛
        current_lambda_recon = CONFIG["max_lambda_recon"] * min(1.0, (epoch + 1) / CONFIG["warmup_epochs"])

        for batch_idx, (inputs, labels, _) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_model.zero_grad()
            optimizer_center.zero_grad()

            valid_mask = get_valid_mask(inputs)
            if batch_idx == 0 and epoch == start_epoch:
                total_elements = inputs.numel()
                actual_zeros = (inputs == 0.0).sum().item()
                mask_ones = (valid_mask == 1.0).sum().item()
                mask_zeros = (valid_mask == 0.0).sum().item()
                
                print(f"\n📊 【資料集與遮罩終極大檢查】")
                print(f"   -> 這一批次總元素量: {total_elements}")
                print(f"   -> 原始資料中精準為 0.0 的數量: {actual_zeros}")
                print(f"   -> 遮罩判定為有效 (1.0) 的數量: {mask_ones}")
                print(f"   -> 遮罩判定為補零 (0.0) 的數量: {mask_zeros}")
                print(f"   -> inputs 數值區間 (Min~Max): {inputs.min().item():.4f} ~ {inputs.max().item():.4f}\n")

            inputs = normalize_skeleton_batch(
                inputs,
                center_joint_idx=CONFIG["center_joint_idx"]
            )           

            if use_amp:
                with torch.amp.autocast("cuda"):
                    # 【修正 1】接收 4 個回傳值，加入 w_hat
                    logits, recon_x, z, w_hat = model(inputs)

                    # ============================================================
                    # 🟢 終極瘦身：徹底解除內戰，只留【分類】與【重建】
                    # ============================================================
                    loss_cls = criterion_cls(logits, labels)
                    loss_recon = torch.sum(((recon_x - inputs)**2) * valid_mask) / (torch.sum(valid_mask) + 1e-6)

                    # 1. 徹底關掉霸凌全場的 Center Loss (暫時不加入總損失)
                    loss_center = criterion_center(z, labels) 

                    # 2. 修正 Entropy 邏輯：改用「最大響應懲罰」，平滑且絕對方向正確地強迫 w_hat 走向 One-hot
                    # 透過最小化【負的極大值】，逼迫優化器去【放大最大值】，達成極致的稀疏路由
                    loss_sparse = -torch.mean(torch.max(w_hat, dim=1)[0])

                    # 3. 徹底拔除幫倒忙的 Diversity Loss

                    # 最終總損失：只讓分類引導方向，重建雕刻細節，稀疏正則負責縮減槽位
                    loss = loss_cls + current_lambda_recon * loss_recon + CONFIG["lambda_center"] * loss_center + lambda_entropy * loss_sparse

                    scaler.scale(loss).backward()
                    scaler.step(optimizer_model)
                    scaler.step(optimizer_center)
                    scaler.update()
                    # 🟢 插入這段診斷代碼 (放在 scaler.update() 的正下方)
                    if batch_idx == 0:
                        print(f"\n🔮 【解碼器深層通靈診斷】 Epoch {epoch+1}")
                        print(f"   -> 1. 解碼器輸出的真實數值平均 (應大於0): {torch.mean(torch.abs(recon_x)).item():.6f}")

                        # 檢查 Decoder 最後一層有沒有拿到反向傳播的梯度
                        if model.decoder.final_conv.weight.grad is not None:
                            grad_norm = torch.norm(model.decoder.final_conv.weight.grad).item()
                            print(f"   -> 2. Decoder 最後一層權重梯度模長: {grad_norm:.6f}")
                        else:
                            print(f"   -> 2. Decoder 最後一層權重梯度: None (⚠️ 徹底斷路！)")

                        # 計算如果預測全為 0 的理論 Loss
                        zero_baseline = torch.sum((inputs**2) * valid_mask) / (torch.sum(valid_mask) + 1e-6)
                        print(f"   -> 3. 理論上「完全盲猜全 0」的基準 Loss: {zero_baseline.item():.4f}\n")
            else:
                # 【修正 1】非 AMP 模式也同步修正
                logits, recon_x, z, w_hat = model(inputs)

                # ============================================================
                # 🟢 終極瘦身：徹底解除內戰，只留【分類】與【重建】
                # ============================================================
                loss_cls = criterion_cls(logits, labels)
                loss_recon = torch.sum(((recon_x - inputs)**2) * valid_mask) / (torch.sum(valid_mask) + 1e-6)
                
                # 1. 徹底關掉霸凌全場的 Center Loss (暫時不加入總損失)
                loss_center = criterion_center(z, labels) 
                
                # 2. 修正 Entropy 邏輯：改用「最大響應懲罰」，平滑且絕對方向正確地強迫 w_hat 走向 One-hot
                # 透過最小化【負的極大值】，逼迫優化器去【放大最大值】，達成極致的稀疏路由
                loss_sparse = -torch.mean(torch.max(w_hat, dim=1)[0])

                # 3. 徹底拔除幫倒忙的 Diversity Loss
                
                # 最終總損失：只讓分類引導方向，重建雕刻細節，稀疏正則負責縮減槽位
                loss = loss_cls + current_lambda_recon * loss_recon + CONFIG["lambda_center"] * loss_center + lambda_entropy * loss_sparse

            # 累加各個 Loss 的數值用於後續統計
            cls_loss_sum += loss_cls.item()
            recon_loss_sum += loss_recon.item()

            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{CONFIG['num_epochs']}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} (Cls: {loss_cls.item():.4f}, Recon: {loss_recon.item():.4f}, )")

        scheduler.step()

        # ========================================================
        # 10. Validation
        # ========================================================
        model.eval()
        val_cls_loss_sum, correct_preds, total_known_samples = 0.0, 0, 0
        
        # 新增：追蹤 已知 與 未知 動作的重建誤差
        recon_loss_known_sum, recon_loss_unknown_sum = 0.0, 0.0
        total_unknown_samples = 0
        
        with torch.no_grad():
            for inputs, labels, is_unknown in val_loader:
                inputs, labels, is_unknown = inputs.to(device), labels.to(device), is_unknown.to(device)
                
                # 驗證階段同步中心化與標準化
                valid_mask = get_valid_mask(inputs)
                inputs = normalize_skeleton_batch(
                    inputs,
                    center_joint_idx=CONFIG["center_joint_idx"]
                )
                
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        # 【修正 1】接收 4 個回傳值，引入 recon_x 進行評估
                        logits, recon_x, _, _ = model(inputs)
                else:
                    # 【修正 1】非 AMP 模式同步接收 4 個值
                    logits, recon_x, _, _ = model(inputs)

                # 建立遮罩：區分已知 (0) 與未知 (1)
                known_mask = (is_unknown == 0)
                unknown_mask = (is_unknown == 1)

                # ---- 1. 計算已知類別的分類與精準度 ----
                if known_mask.sum() > 0:
                    val_cls_loss_sum += criterion_cls(logits[known_mask], labels[known_mask]).item()
                    preds = torch.argmax(logits, dim=1)
                    correct_preds += (preds[known_mask] == labels[known_mask]).sum().item()
                    total_known_samples += known_mask.sum().item()
                    
                    # 計算已知動作的重建誤差
                    loss_recon_k = torch.sum(((recon_x[known_mask] - inputs[known_mask])**2) * valid_mask[known_mask]) / (torch.sum(valid_mask[known_mask]) + 1e-6)
                    recon_loss_known_sum += loss_recon_k.item() * known_mask.sum().item()

                # ---- 2. 計算未知類別的重建誤差 (Open-Set 的關鍵指標) ----
                if unknown_mask.sum() > 0:
                    total_unknown_samples += unknown_mask.sum().item()
                    
                    # 計算未知動作的重建誤差
                    loss_recon_u = torch.sum(((recon_x[unknown_mask] - inputs[unknown_mask])**2) * valid_mask[unknown_mask]) / (torch.sum(valid_mask[unknown_mask]) + 1e-6)
                    recon_loss_unknown_sum += loss_recon_u.item() * unknown_mask.sum().item()

        # ---- 3. 統計數據計算 ----
        val_accuracy = (correct_preds / total_known_samples * 100) if total_known_samples > 0 else 0.0
        avg_train_cls_loss = cls_loss_sum / len(train_loader)
        avg_train_recon_loss = recon_loss_sum / len(train_loader)
        avg_train_center_loss = center_loss_sum / len(train_loader)
        
        avg_val_cls_loss = val_cls_loss_sum / len(val_loader) if len(val_loader) > 0 else 0.0
        
        # 計算平均重建誤差 (按樣本數加權平均)
        avg_val_recon_known = (recon_loss_known_sum / total_known_samples) if total_known_samples > 0 else 0.0
        avg_val_recon_unknown = (recon_loss_unknown_sum / total_unknown_samples) if total_unknown_samples > 0 else 0.0

        # ---- 4. 儀表板列印 ----
        print(f"\n{'='*60}\nEpoch [{epoch+1}/{CONFIG['num_epochs']}] | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"[Train] Cls: {avg_train_cls_loss:.4f} | Recon: {avg_train_recon_loss:.4f} | Center: {avg_train_center_loss:.4f}")
        print(f"[Val]   Cls: {avg_val_cls_loss:.4f} | Accuracy: {val_accuracy:.2f}%")
        print(f"[👉 Recon Check] Known-Recon: {avg_val_recon_known:.4f} 🟢 | Unknown-Recon: {avg_val_recon_unknown:.4f} 🔴")
        print(f"{'='*60}\n")

        # 儲存 Checkpoint
        save_params = {
            "epoch": epoch, "model": model, "optimizer_model": optimizer_model, "optimizer_center": optimizer_center,
            "scheduler": scheduler, "scaler": scaler, "best_val_accuracy": best_val_accuracy, "val_accuracy": val_accuracy,
            "avg_train_cls_loss": avg_train_cls_loss, "avg_train_recon_loss": avg_train_recon_loss,
            "avg_train_center_loss": avg_train_center_loss, "known_actions": CONFIG["known_actions"], "num_classes": num_classes
        }
        save_checkpoint(os.path.join(checkpoint_dir, "last.pth"), **save_params)
        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(os.path.join(checkpoint_dir, "best_val.pth"), **save_params)
            print(f"🏆 發現新高分！已儲存至 best_val.pth")

    # 訓練結束
    save_checkpoint(os.path.join(checkpoint_dir, "final.pth"), **save_params)
    
    total_time = time.perf_counter() - total_start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    print(f"🎉 訓練完成！最佳模型：{os.path.join(checkpoint_dir, 'best_val.pth')}")
    print(f"⏱️ 總訓練時間: {hours} 小時 {minutes} 分 {seconds:.2f} 秒")