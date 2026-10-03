import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActionMemoryModule(nn.Module):
    def __init__(self, mem_dim=100, fea_dim=256):
        super(ActionMemoryModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        
        self.memory = nn.Parameter(torch.randn(mem_dim, fea_dim))
        nn.init.kaiming_uniform_(self.memory)

    def forward(self, z):
        # 使用餘弦相似度，將得分嚴格限制在 [-1, 1]，乘上溫和的溫度係數 10.0
        z_norm = F.normalize(z, dim=-1)
        mem_norm = F.normalize(self.memory, dim=-1)
        scores = torch.mm(z_norm, mem_norm.t()) * 10.0 # [B, mem_dim]
        
        w_hat = F.softmax(scores, dim=-1)
        z_purified = torch.mm(w_hat, self.memory)
        
        return z_purified, w_hat