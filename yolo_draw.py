import os
import numpy as np
import matplotlib.pyplot as plt

# 解決 matplotlib 繪圖時的中文字體問題
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

def plot_coco_17_skeleton(npy_path, frame_idx=0):
    """
    讀取 yolo26 轉出的 17 點骨架檔 (.npy)，並繪製指定幀數的 2D 骨架圖
    """
    if not os.path.exists(npy_path):
        print(f"❌ 找不到骨架檔案：{npy_path}")
        return

    # 1. 載入骨架數據，Shape 為 [2, T, 17]
    skeleton = np.load(npy_path)
    C, T, V = skeleton.shape
    
    print(f"📊 成功載入檔案。Shape: {skeleton.shape} (通道數:{C}, 總幀數:{T}, 關節點數:{V})")
    
    if frame_idx >= T:
        print(f"⚠️ 指定的 frame_idx ({frame_idx}) 超過該影片總幀數 ({T})，自動切換至第 0 幀。")
        frame_idx = 0

    # 2. 擷取特定幀的 X, Y 座標 -> Shape: (17,)
    x = skeleton[0, frame_idx, :]
    y = skeleton[1, frame_idx, :]

    # 💡 檢查防呆：如果該幀完全沒有偵測到人（全為 0），則提醒使用者
    if np.all(x == 0) and np.all(y == 0):
        print(f"ℹ️ 提示：第 {frame_idx} 幀的數據完全為 0 (YOLO 在此幀可能沒偵測到人)。")

    # 3. 定義 COCO 17 點的骨骼連接關係 (與模型 Adjacency 相同)
    coco_edges = [
        (0, 1), (0, 2), (1, 3), (2, 4),       # 臉部 (鼻、眼、耳)
        (5, 6),                               # 雙肩
        (5, 7), (7, 9),                       # 左手臂 (肩-肘-腕)
        (6, 8), (8, 10),                      # 右手臂 (肩-肘-腕)
        (5, 11), (6, 12), (11, 12),           # 軀幹 (雙肩連雙臀)
        (11, 13), (13, 15),                   # 左腿 (臀-膝-踝)
        (12, 14), (14, 16)                    # 右腿 (臀-膝-踝)
    ]

    # 4. 開始繪圖
    plt.figure(figsize=(6, 8))
    
    # 畫出 17 個關節點（紅點）
    plt.scatter(x, y, color='crimson', s=40, zorder=3, label='關節點')

    # 畫出骨骼連接線（藍線）
    for i, edge in enumerate(coco_edges):
        p1, p2 = edge
        # 如果任一點為 0（代表未偵測到該關節），則不畫線以免連到原點
        if (x[p1] == 0 and y[p1] == 0) or (x[p2] == 0 and y[p2] == 0):
            continue
        
        # 區分左右邊（自由選擇是否著色，這裡統一用寶藍色）
        plt.plot([x[p1], x[p2]], [y[p1], y[p2]], color='royalblue', linewidth=2, zorder=2)

    # 5. 加上一些輔助標記，方便肉眼看懂點位
    # 在肩膀、臀部與手腕處標註文字
    important_joints = {0: "鼻", 5: "左肩", 6: "右肩", 11: "左臀", 12: "右臀"}
    for joint_idx, name in important_joints.items():
        if x[joint_idx] != 0 or y[joint_idx] != 0:
            plt.text(x[joint_idx] + 0.01, y[joint_idx] + 0.01, name, fontsize=10, color='darkgreen')

    plt.title(f'YOLO26 17點骨架視覺化 (第 {frame_idx} 幀)')
    plt.xlabel('X 座標')
    plt.ylabel('Y 座標')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # ⚠️ YOLO 的影像座標系通常 Y 軸朝下（上方為 0），為了符合人眼視覺，將 Y 軸反轉
    plt.gca().invert_yaxis() 
    
    # 保持 X, Y 軸比例 1:1，人才不會變形（變胖或變高）
    plt.axis('equal') 
    
    # 儲存並顯示
    output_img = f"skeleton_frame_{frame_idx}.png"
    plt.savefig(output_img, bbox_inches='tight', dpi=150)
    print(f"📸 骨架圖已成功生成並儲存至：{output_img}")
    plt.show()

if __name__ == "__main__":
    # 💡 請填入你其中一個標記好的 .npy 檔案路徑進行測試
    # 範例路徑：r".\NTU60_YOLO26_SKELETONS\S001C001P001R001A001.npy"
    TEST_NPY_PATH = r".\NTU60\nturgb+d_yolo_skeletons\S001C001P001R001A001_rgb.npy"
    
    # 繪製第 0 幀（第一幀）
    plot_coco_17_skeleton(TEST_NPY_PATH, frame_idx=0)