import torch
import torch.nn as nn
import torch.nn.functional as F

class Strictly_Bottleneck_Decoder(nn.Module):
    def __init__(self, compress_dim=32, out_channels=3, target_frames=300):
        super(Strictly_Bottleneck_Decoder, self).__init__()
        self.target_frames = target_frames
        
        # --- Skip Connection 轉換層 (保持不變) ---
        self.enc_f2_conv = nn.Conv2d(128, 64, kernel_size=1)
        self.enc_f1_conv = nn.Conv2d(64, 32, kernel_size=1)

        # --- Decoder 主幹 ---
        
        # 第一階段：使用 Transposed Conv 將 c3 放大 (T/4 -> T/2)
        # kernel_size=(3, 1) 配合 stride=(2, 1) 與 padding=(1, 0) 可以穩定實現 2 倍上採樣
        self.up_trans1 = nn.ConvTranspose2d(
            compress_dim, 64, 
            kernel_size=(3, 1), 
            stride=(2, 1), 
            padding=(1, 0), 
            output_padding=(1, 0)
        )
        
        # 第二階段：自身 64 ch + f2 融合後的 64 ch = 128 ch
        self.up2_block = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # 第三階段：再次使用 Transposed Conv 放大 (T/2 -> T)
        self.up_trans2 = nn.ConvTranspose2d(
            64, 64, 
            kernel_size=(3, 1), 
            stride=(2, 1), 
            padding=(1, 0), 
            output_padding=(1, 0)
        )
        
        # 融合 f1 後：自身 64 ch + f1 融合後的 32 ch = 96 ch
        self.up3_block = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def forward(self, c3, f2_enc, f1_enc):
        # c3: [B, 32, T/4, 18]
        # f2_enc: [B, 128, T/2, 18]
        # f1_enc: [B, 64, T, 18]

        # 1. 放大到 T/2 並融合 f2
        x = F.relu(self.up_trans1(c3)) # [B, 64, T/2, 18]
        
        f2_p = self.enc_f2_conv(f2_enc)
        x = torch.cat([x, f2_p], dim=1) # [B, 128, T/2, 18]
        x = self.up2_block(x)
        
        # 2. 放大到 T 並融合 f1
        x = F.relu(self.up_trans2(x)) # [B, 64, T, 18]
        
        f1_p = self.enc_f1_conv(f1_enc)
        x = torch.cat([x, f1_p], dim=1) # [B, 96, T, 18]
        x = self.up3_block(x)
        
        # 3. 強制對齊原始幀數 (預防 target_frames 不是 4 的倍數時產生的偏差)
        if x.shape[2] != self.target_frames:
            x = F.interpolate(x, size=(self.target_frames, x.shape[3]), mode='bilinear', align_corners=False)
        
        recon_x = self.final_conv(x)
        return recon_x