import torch
import torch.nn as nn
import torch.nn.functional as F

class Strictly_Bottleneck_Decoder(nn.Module):
    def __init__(self, compress_dim=32, out_channels=2, target_frames=300, num_nodes=17):
        super(Strictly_Bottleneck_Decoder, self).__init__()
        self.target_frames = target_frames
        self.num_nodes = num_nodes
        
        # 🟢 先透過 Linear 層，將全域的 256 維向量，精準擴展回初始時空瓶頸形狀 [32, 75, num_nodes]
        self.fc_up = nn.Sequential(
            nn.Linear(256, compress_dim * 75 * num_nodes),
            nn.SiLU()
        )
        
        # 第一階段：時間軸 75 -> 150 幀
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(compress_dim, 64, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        
        # 第二階段：時間軸 150 -> 300 幀
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )
        
        # 第三階段：通道壓縮 (128 ch -> 32 ch)
        self.up3 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def forward(self, z_purified):
        # 輸入 z_purified 形狀: [B, 256]
        B = z_purified.shape[0]
        
        # 1. 映射並還原為 4D 特徵圖
        x = self.fc_up(z_purified)
        x = x.view(B, 32, 75, self.num_nodes) # -> [B, 32, 75, 17]
        
        # 2. 轉置卷積解碼
        x = self.up1(x) # -> [B, 64, 150, 17]
        x = self.up2(x) # -> [B, 128, 300, 17]
        x = self.up3(x) # -> [B, 32, 300, 17]
        
        if x.shape[2] != self.target_frames:
            x = F.interpolate(x, size=(self.target_frames, x.shape[3]), mode='nearest')
            
        recon_x = self.final_conv(x) # -> [B, 2, 300, 17]
        return recon_x