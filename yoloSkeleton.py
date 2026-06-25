import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

def process_avi_to_skeleton_v26(video_path, model, target_frames=300):
    """
    讀取單個 avi 影片，使用 YOLOv26-pose 提取 17 點骨架，輸出 Shape 為 [2, target_frames, 17]
    """
    cap = cv2.VideoCapture(video_path)
    
    # 初始化骨架矩陣 [C, T, V] -> [2, 300, 17] (C=2 代表只有 X, Y 軸)
    skeleton_matrix = np.zeros((2, target_frames, 17), dtype=np.float32)
    
    frame_idx = 0
    while cap.isOpened() and frame_idx < target_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 進行 YOLOv26-pose 預測 (stream=False 確保當前幀穩定獲取)
        results = model(frame, verbose=False, conf=0.3, stream=False)
        
        for r in results:
            if r.keypoints is not None and len(r.keypoints.data) > 0:
                # 預設抓取畫面中置信度最高（第一個）的人 [0]
                kp = r.keypoints.data[0].cpu().numpy()
                
                # 填入 X 軸與 Y 軸
                skeleton_matrix[0, frame_idx, :] = kp[:, 0]  # X 座標
                skeleton_matrix[1, frame_idx, :] = kp[:, 1]  # Y 座標
                break 
                
        frame_idx += 1
        
    cap.release()
    return skeleton_matrix

def load_progress(progress_file):
    """載入已完成的影片檔名清單"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    return set()

def save_progress(progress_file, video_name):
    """將剛完成的影片檔名寫入紀錄檔中"""
    with open(progress_file, 'a', encoding='utf-8') as f:
        f.write(f"{video_name}\n")

def batch_convert_dataset_v26_with_checkpoint(video_root_dir, output_root_dir):
    """
    批次轉換整個資料夾底下的 avi 影片，並具備自動斷點續傳 (Checkpoint) 功能
    """
    # 1. 載入 YOLOv26-pose 權重檔
    model_path = 'yolo26x-pose.pt' 
    if not os.path.exists(model_path):
        print(f"📦 找不到本地權重，系統將自動下載模型至當前目錄...")
    model = YOLO(model_path) 
    
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir)
        
    # 定義進度紀錄 Checkpoint 檔案路徑
    progress_file = os.path.join(output_root_dir, ".yolo_progress.txt")
    
    # 2. 讀取歷史進度與當前目錄下的所有 AVI 影片
    completed_videos = load_progress(progress_file)
    all_video_files = sorted([f for f in os.listdir(video_root_dir) if f.endswith('.avi')])
    
    # 過濾出「尚未處理」的影片
    videos_to_process = [f for f in all_video_files if f not in completed_videos]
    
    total_all = len(all_video_files)
    total_skip = len(completed_videos)
    total_todo = len(videos_to_process)
    
    print(f"\n====================================================")
    print(f"🔄 YOLOv26-pose 斷點續傳檢查機制啟動:")
    print(f"   - 資料夾內總影片數: {total_all} 個")
    print(f"   - ⚖️ 偵測到歷史 Checkpoint 已完成: {total_skip} 個")
    print(f"   - 🚀 本次需要繼續執行 (剩餘): {total_todo} 個")
    print(f"====================================================\n")
    
    if total_todo == 0:
        print("🎉 檢查完畢！所有影片先前皆已轉換成功，無需重複執行。")
        return

    # 3. 開始繼續處理剩餘影片
    for idx, video_file in enumerate(videos_to_process):
        video_path = os.path.join(video_root_dir, video_file)
        
        # 提取骨架
        skeleton_data = process_avi_to_skeleton_v26(video_path, model, target_frames=300)
        
        # 檔名轉換與儲存 (如 S001C001...avi -> S001C001...npy)
        file_name_without_ext = os.path.splitext(video_file)[0]
        output_path = os.path.join(output_root_dir, f"{file_name_without_ext}.npy")
        np.save(output_path, skeleton_data)
        
        # 核心：成功存檔後，立即更新 Checkpoint 紀錄
        save_progress(progress_file, video_file)
        
        # 計算整體總進度 (包含過去做完的)
        current_global_idx = total_skip + idx + 1
        if current_global_idx % 10 == 0 or current_global_idx == total_all:
            print(f"✅ 總進度: [{current_global_idx}/{total_all}] | 本次執行第 {idx + 1} 個 -> {file_name_without_ext}.npy 儲存並紀錄成功")

if __name__ == "__main__":
    # 💡 請設定你的 AVI 影片來源路徑，以及輸出的 17 點骨架 npy 存放路徑
    MY_AVI_DIR = r".\NTU60\nturgb+d_rgb"
    MY_OUTPUT_DIR = r".\NTU60\nturgb+d_yolo_skeletons"
    
    batch_convert_dataset_v26_with_checkpoint(MY_AVI_DIR, MY_OUTPUT_DIR)
    print("\n🎉 YOLOv26-pose 批次任務與進度紀錄完全結束！")