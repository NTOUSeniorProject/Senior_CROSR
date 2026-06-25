import torch
import torch.nn as nn
import torch.nn.functional as F

# 假設這些元件已經存放在你的專案中 (參考前幾篇的官方程式碼)
from net.net import Unit2D
from net.unit_gcn import unit_gcn
from net.st_gcn import TCN_GCN_unit 

class STGCN_Encoder(nn.Module):
    def __init__(self, in_channels=2, latent_dim=256, A=None): 
        super(STGCN_Encoder, self).__init__()
        
        # 確保有傳入計算好的 Spatial Configuration 鄰接矩陣 (3, 18, 18)
        if A is None:
            raise ValueError("必須傳入鄰接矩陣 A")
        self.register_buffer('A', A)

        # ==========================================
        # 0. 初始資料映射層 (把座標映射到高維特徵空間)
        # ==========================================
        self.gcn0 = unit_gcn(in_channels, 64, self.A, mask_learning=True)
        self.tcn0 = Unit2D(64, 64, kernel_size=9)

        # ==========================================
        # 1. 第一階段: 64 Channels (時間長度不變)
        # ==========================================
        self.layer1 = TCN_GCN_unit(64, 64, self.A)
        self.layer2 = TCN_GCN_unit(64, 64, self.A)
        self.layer3 = TCN_GCN_unit(64, 64, self.A)

        # ==========================================
        # 2. 第二階段: 128 Channels (時間長度減半)
        # ==========================================
        # 注意：layer4 設定了 stride=2，這會自動把時間軸縮半，取代原來的 MaxPool
        self.layer4 = TCN_GCN_unit(64, 128, self.A, stride=2) 
        self.layer5 = TCN_GCN_unit(128, 128, self.A)
        self.layer6 = TCN_GCN_unit(128, 128, self.A)

        # ==========================================
        # 3. 第三階段: 256 Channels (時間長度再減半)
        # ==========================================
        self.layer7 = TCN_GCN_unit(128, 256, self.A, stride=2)
        self.layer8 = TCN_GCN_unit(256, 256, self.A)
        self.layer9 = TCN_GCN_unit(256, 256, self.A)

        # ==========================================
        # 4. 潛在空間壓縮 (Latent Space)
        # ==========================================
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_latent = nn.Linear(256, latent_dim)

    def forward(self, x):
        # 預期 x 形狀: (N, C, T, V) -> (Batch, 3, Frames, 18)
        
        # 初始映射
        x = self.tcn0(self.gcn0(x))

        # ---------------- Stage 1 ----------------
        x = self.layer1(x)
        x = self.layer2(x)
        f1 = self.layer3(x) 
        # f1 形狀約為: (Batch, 64, Frames, 18)

        # ---------------- Stage 2 ----------------
        x = self.layer4(f1) # 時間減半發生在這裡！
        x = self.layer5(x)
        f2 = self.layer6(x)
        # f2 形狀約為: (Batch, 128, Frames/2, 18)

        # ---------------- Stage 3 ----------------
        x = self.layer7(f2) # 時間再減半發生在這裡！
        x = self.layer8(x)
        f3 = self.layer9(x)
        # f3 形狀約為: (Batch, 256, Frames/4, 18)

        # ---------------- Latent Z ----------------
        # 將時間與節點全部平均掉，濃縮成全域特徵
        flat = self.global_pool(f3)      # -> (Batch, 256, 1, 1)
        flat = torch.flatten(flat, 1)    # -> (Batch, 256)
        z = self.fc_latent(flat)         # -> (Batch, latent_dim)

        # 回傳三個階段的特徵供 Decoder 重建，以及 z 供分類器/Center Loss 使用
        return [f1, f2, f3], z