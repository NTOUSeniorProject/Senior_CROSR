import torch
import torch.nn as nn  
import torch.optim as optim
import os
import numpy as np

# 1. 記得修改 Import 的 Decoder 名稱
from STGCNEncoder import STGCN_Encoder
from strictly_bottleneck_decoder import Strictly_Bottleneck_Decoder 

class ST_CROSR(nn.Module):
    def __init__(self, num_known_classes, num_nodes=25, target_frames=300):
        super(ST_CROSR, self).__init__()
        if num_nodes == 25:
            self.register_buffer('A', self.build_ntu_25_matrix(num_nodes))
        else:
            self.register_buffer('A', self.build_kinetics_18_matrix(num_nodes))
        
        self.encoder = STGCN_Encoder(in_channels=3, latent_dim=256, A=self.A)
        self.classifier = nn.Linear(256, num_known_classes)
        
        # 設定壓縮後的維度
        compress_dim = 32
        
        # --- 重新加回 Bottleneck 層 ---
        # 這些層負責將不同階段的 Feature Map 通道數對齊，以便輸入 Decoder
        #self.bottleneck1 = nn.Conv2d(64, compress_dim, kernel_size=1)  # 對應 f1
        #self.bottleneck2 = nn.Conv2d(128, compress_dim, kernel_size=1) # 對應 f2
        self.bottleneck3 = nn.Conv2d(256, compress_dim, kernel_size=1) # 對應 f3
        
        # 初始化 Decoder
        self.decoder = Strictly_Bottleneck_Decoder(compress_dim=compress_dim, out_channels=3, target_frames=target_frames)

    def forward(self, x):
        # 1. Encoder 提取多層級特徵
        features, z = self.encoder(x)
        f1, f2, f3 = features 
        
        logits = self.classifier(z)
        
        # 2. 獲取各層級的壓縮特徵 (CROS 機制核心)
        # 將原本省去的 c1, c2 重新計算出來
        c1 = self.bottleneck1(f1)
        c2 = self.bottleneck2(f2)
        c3 = self.bottleneck3(f3)
        
        # 3. 呼叫 Decoder，將所有層級的壓縮特徵傳入
        # 確保你的 Strictly_Bottleneck_Decoder 接收的是壓縮後的特徵 (c1, c2, c3)
        recon_x = self.decoder(c3, c2, c1)
        
        return logits, recon_x, z

    def build_ntu_25_matrix(self, num_nodes=25):
        self_link = [(i, i) for i in range(num_nodes)]
        # NTU 連結定義 (0-based)
        inward = [
            (1-1, 2-1), (2-1, 21-1), (21-1, 3-1), (3-1, 4-1),
            (21-1, 5-1), (5-1, 6-1), (6-1, 7-1), (7-1, 8-1), (8-1, 22-1), (8-1, 23-1),
            (21-1, 9-1), (9-1, 10-1), (10-1, 11-1), (11-1, 12-1), (12-1, 24-1), (12-1, 25-1),
            (1-1, 13-1), (13-1, 14-1), (14-1, 15-1), (15-1, 16-1),
            (1-1, 17-1), (17-1, 18-1), (18-1, 19-1), (19-1, 20-1)
        ]
        return self._process_adjacency(num_nodes, self_link, inward)

    # --- 專屬函式 2：處理 18 點 (Kinetics/OpenPose) ---
    def build_kinetics_18_matrix(self, num_nodes=18):
        self_link = [(i, i) for i in range(num_nodes)]
        # 原有 18 點連結定義
        inward = [
            (4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), 
            (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1), 
            (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)
        ]
        return self._process_adjacency(num_nodes, self_link, inward)
    def _process_adjacency(self, num_nodes, self_link, inward):
        outward = [(j, i) for (i, j) in inward]

        def edge2mat(link, num_node):
            A = np.zeros((num_node, num_node))
            for i, j in link:
                A[j, i] = 1
            return A

        def normalize_digraph(A):
            Dl = np.sum(A, 0)
            num_node = A.shape[0]
            Dn = np.zeros((num_node, num_node))
            for i in range(num_node):
                if Dl[i] > 0:
                    Dn[i, i] = Dl[i]**(-1)
            return np.dot(A, Dn)

        I = edge2mat(self_link, num_nodes)
        In = normalize_digraph(edge2mat(inward, num_nodes))
        Out = normalize_digraph(edge2mat(outward, num_nodes))

        A_spatial = np.stack((I, In, Out))
        return torch.tensor(A_spatial, dtype=torch.float32)