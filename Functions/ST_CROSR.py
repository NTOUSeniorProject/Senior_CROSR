import torch
import torch.nn as nn  
import numpy as np
from STGCNEncoder import STGCN_Encoder
from strictly_bottleneck_decoder import Strictly_Bottleneck_Decoder 
from actionMemoryModule import ActionMemoryModule 

class ST_CROSR(nn.Module):
    def __init__(self, num_known_classes, num_nodes=17, target_frames=300):
        super(ST_CROSR, self).__init__()
        
        if num_nodes == 17:
            self.register_buffer('A', self.build_coco_17_matrix(num_nodes))
        elif num_nodes == 25:
            self.register_buffer('A', self.build_ntu_25_matrix(num_nodes))
        else:
            self.register_buffer('A', self.build_kinetics_18_matrix(num_nodes))
        
        self.encoder = STGCN_Encoder(in_channels=2, latent_dim=256, A=self.A)
        self.classifier = nn.Linear(256, num_known_classes)
        self.memoryModule = ActionMemoryModule(mem_dim=100, fea_dim=256)
        self.decoder = Strictly_Bottleneck_Decoder(compress_dim=32, out_channels=2, target_frames=target_frames, num_nodes=num_nodes)

    def forward(self, x):
        _, z = self.encoder(x) 
        logits = self.classifier(z)
        
        # 全域特徵過濾
        z_purified, w_hat = self.memoryModule(z)
        
        # 🟢【核心修正：混合特徵門控】
        # 注入 20% 的個體變異數，確保同一個 Batch 內輸入解碼器的特徵絕不相同，徹底打破死鎖！
        z_decoder = 0.2 * z + 0.8 * z_purified
        
        recon_x = self.decoder(z_decoder)
        return logits, recon_x, z, w_hat

    def build_coco_17_matrix(self, num_nodes=17):
        self_link = [(i, i) for i in range(num_nodes)]
        inward = [
            (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), 
            (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), 
            (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        return self._process_adjacency(num_nodes, self_link, inward)

    def build_ntu_25_matrix(self, num_nodes=25):
        self_link = [(i, i) for i in range(num_nodes)]
        inward = [
            (0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
            (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24), (0, 12), (12, 13), (13, 14), (14, 15),
            (0, 16), (16, 17), (17, 18), (18, 19)
        ]
        return self._process_adjacency(num_nodes, self_link, inward)

    def build_kinetics_18_matrix(self, num_nodes=18):
        self_link = [(i, i) for i in range(num_nodes)]
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
            for i, j in link: A[j, i] = 1
            return A
        def normalize_digraph(A):
            Dl = np.sum(A, 0)
            num_node = A.shape[0]
            Dn = np.zeros((num_node, num_node))
            for i in range(num_node):
                if Dl[i] > 0: Dn[i, i] = Dl[i]**(-1)
            return np.dot(A, Dn)
        I = edge2mat(self_link, num_nodes)
        In = normalize_digraph(edge2mat(inward, num_nodes))
        Out = normalize_digraph(edge2mat(outward, num_nodes))
        return torch.tensor(np.stack((I, In, Out)), dtype=torch.float32)