import torch.nn.functional as F
import torch

from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, out_dim, cached=True, normalize=True) 
        # cached: 정규화된 인접 행렬을 한 번 계산하고 캐싱. 이후에는 캐싱된 값을 사용. 
        # normalize: 정규화된 인접 행렬
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        # feature 행렬 (num_nodes x feature_dim)
        x = F.dropout(x, p=self.dropout, training=self.training) # 학습 중(self.training=True)일 때만 Dropout을 적용.
        x = self.conv1(x, edge_index, edge_weight) 
        # 얻은 x: feature 행렬 (num_nodes x out_dim)
        # edge_index: 엣지의 인덱스 (첫 번째 행: 시작 노드, 두 번째 행: 끝 노드)
        # edge_weight: 엣지의 가중치
        return x
