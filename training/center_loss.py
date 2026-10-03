import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    """
    Center Loss 的核心實作
    目標是最小化每個特徵到其所屬類別中心點的歐式距離。
    """
    def __init__(self, num_classes=4, feat_dim=256, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        # 宣告一個可學習的參數矩陣 (Centroids)，形狀為 [類別數, 特徵維度]
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        x: 形狀為 (batch_size, feat_dim) 的特徵向量 z
        labels: 形狀為 (batch_size,) 的真實標籤
        """
        batch_size = x.size(0)
        
        # 取得當前 batch 中，每個樣本對應的中心點
        centers_batch = self.centers.index_select(0, labels)
        
        # 計算每個樣本與其中心點的距離平方
        # (這裡除以 2 是為了求導時剛好跟平方消掉，保持梯度平滑)
        loss = (x - centers_batch).pow(2).sum() / 2.0 / batch_size
        return loss