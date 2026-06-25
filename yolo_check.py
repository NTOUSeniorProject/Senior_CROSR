import os
import glob
from collections import Counter

def check_my_yolo_dataset():
    # ============================================================
    # ⚙️ 設定區（請對齊你的 train_ntu_crosr.py CONFIG）
    # ============================================================
    data_root = r".\NTU60\nturgb+d_yolo_skeletons"
    known_actions = {
        1, 2, 3, 4, 5, 6, 8, 9, 11, 12,
        14, 15, 16, 17, 18, 19, 20, 21,
        23, 25, 28, 29, 30, 32, 33, 34, 37,
        41, 42, 43, 44, 45, 46, 47, 48, 49
    }
    
    # NTU 官方 Cross-Subject 訓練集人員 ID
    official_train_subjects = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}

    print("====================================================")
    print(f"🔍 開始掃描 YOLO26 骨架資料夾: {data_root}")
    print("====================================================")

    # 1. 找出所有 .npy 檔案
    all_files = sorted(glob.glob(os.path.join(data_root, "*.npy")))
    total_files = len(all_files)
    
    if total_files == 0:
        print(f"❌ 錯誤：在 {data_root} 底下找不到任何 .npy 檔案！")
        return

    print(f"📂 資料夾內 .npy 檔案總數: {total_files} 筆")

    # 2. 初始化統計變數
    train_split_known = 0      # 訓練集 - 已知動作 (真正的 train_loader 樣本)
    train_split_unknown = 0    # 訓練集 - 未知動作 (OSR 訓練中會被 continue 砍掉)
    
    val_split_known = 0        # 驗證集 - 已知動作
    val_split_unknown = 0      # 驗證集 - 未知動作 (OSR 測試時的異常樣本)
    
    error_files = []           # 檔名解析失敗的壞檔
    action_counter = Counter() # 統計各個動作的分佈

    # 3. 開始逐一剖析檔名
    for path in all_files:
        filename = os.path.basename(path)
        
        try:
            # 精準擷取 A 與 P 後面的 3 位數字
            a_idx = filename.find('A')
            p_idx = filename.find('P')
            if a_idx == -1 or p_idx == -1:
                error_files.append(filename)
                continue
                
            action_id = int(filename[a_idx + 1 : a_idx + 4])
            subject_id = int(filename[p_idx + 1 : p_idx + 4])
            
            action_counter[action_id] += 1
            
            # 判斷屬於哪個 Split (Cross-Subject 邏輯)
            is_train_subject = (subject_id in official_train_subjects)
            is_known_action = (action_id in known_actions)
            
            if is_train_subject:
                if is_known_action:
                    train_split_known += 1
                else:
                    train_split_unknown += 1
            else:
                if is_known_action:
                    val_split_known += 1
                else:
                    val_split_unknown += 1
                    
        except Exception:
            error_files.append(filename)

    # ============================================================
    # 🎯 輸出審計報告
    # ============================================================
    print("\n====================================================")
    print("📊 NTU OSR 資料樹與可用樣本分配報告")
    print("====================================================")
    print(f"1. 🚂 訓練階段 (split='train', is_train=True):")
    print(f"   👉 實際進入模型訓練 (已知動作): {train_split_known} 筆")
    print(f"   ❌ 被過濾捨棄 (未知動作)      : {train_split_unknown} 筆")
    
    print(f"\n2. 🚨 測試/驗證階段 (split='val', is_train=False):")
    print(f"   🟢 測試集中的「正常樣本」(已知) : {val_split_known} 筆")
    print(f"   🔴 測試集中的「異常雷達」(未知) : {val_split_unknown} 筆")
    print(f"   👉 test_ntu_dataset.py 總數  : {val_split_known + val_split_unknown} 筆")
    
    print(f"\n3. ⚠️ 異常防呆檢查:")
    print(f"   💥 檔名解析失敗或損毀檔案     : {len(error_files)} 筆")
    if error_files:
        print(f"      （前 3 個壞檔範例: {error_files[:3]}）")
        
    print("====================================================")
    
    # 驗證總數是否對齊
    sum_check = train_split_known + train_split_unknown + val_split_known + val_split_unknown + len(error_files)
    print(f"🔍 數據校對: 分類加總 = {sum_check} (與總數 {total_files} {'對齊 ✅' if sum_check == total_files else '不對齊 ❌'})")
    print("====================================================\n")

    # 4. 顯示動作類別的覆蓋率，幫你檢查有沒有哪些動作根本沒轉到
    print("💡 動作類別 (Action ID) 覆蓋狀況檢查:")
    existing_actions = sorted(list(action_counter.keys()))
    print(f"   - 目前資料夾內共包含 {len(existing_actions)} 個動作類別 (NTU 60 最高為 60)")
    missing_known = [a for a in known_actions if a not in action_counter]
    if missing_known:
        print(f"   ⚠️  警告：你的 CONFIG 設了已知動作，但資料夾內完全沒有這些 .npy 檔案: {missing_known}")
    else:
        print("   ✅ 完美！你 CONFIG 設定的已知動作，硬碟裡全部都有對應的 .npy 檔案。")

if __name__ == "__main__":
    check_my_yolo_dataset()