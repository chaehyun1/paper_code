from torch import nn
import torch

class EGCL(nn.Module):
    def __init__(self, raw_dim, hid_dim):
        super(EGCL, self).__init__() 
        linear_xavier = nn.Linear(hid_dim, 1, bias=False) 
        torch.nn.init.xavier_uniform_(linear_xavier.weight, gain=0.001) # xavier initialization
        
        self.msg_mlp = nn.Sequential( # 논문 식 (4), \phi_m(~)
            nn.Linear(raw_dim + raw_dim + 1, hid_dim), # raw_dim + raw_dim + 1: h_i, h_j, sqr_dist를 concat해서 쓰기 때문, 논문 식 (4) 참고
            nn.SiLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.SiLU()
        )

        self.trans_mlp = nn.Sequential( # 논문 식 (5) 참고, \phi_l(m_i)
            nn.Linear(hid_dim, hid_dim),
            nn.SiLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.SiLU(),
            linear_xavier # 출력 차원: 1 -> 최종적으로 [*, 1]의 차원을 가짐
        )
        
        self.posi_mlp = nn.Sequential( # 논문 식 (7), \phi_s(~)
            nn.Linear(raw_dim + hid_dim, hid_dim), # raw_dim + hid_dim: h_i, msg_sum을 concat해서 쓰기 때문, 논문 식 (7) 참고
            nn.SiLU(),
            nn.Linear(hid_dim, raw_dim) # output의 차원은 input의 차원과 같아야 함 (raw_dim)
        )

    def msg_model(self, h_i, h_j, sqr_dist):
        # 논문 식 (4): concat하기 
        out = torch.cat([h_i, h_j, sqr_dist], dim=1) 
        out = self.msg_mlp(out) # 논문 식 (4): message 얻기 

        return out # = message

    def coord_model(self, x, edge_index, coord_diff, msg):
        row, col = edge_index  # row: [(query*way)*(shot*way),]
        trans = coord_diff * self.trans_mlp(msg) # 논문 식 (5): (h_i - h_j)*\phi_l(m_ij), 차원: [(query*way)*(shot*way), hid_dim]
        trans = diff_mean(trans, row, num_nodes=x.size(0)) # 차원: [num_nodes, hid_dim], num_nodes = way*(shot+query)
        x = x + trans # 논문 식 (5) 참고 

        return x

    def posi_model(self, p, edge_index, msg):
        row, col = edge_index # msg: [way*(shot+query), hid_dim]
        msg_sum = msg_collect(msg, row, num_nodes=p.size(0)) # 논문 식 (6), 차원: [way*(shot+query), hid_dim]
        out = torch.cat([p, msg_sum], dim=1)
        out = self.posi_mlp(out) # 논문 식 (7), 차원: [way*(shot+query), raw_dim], raw_dim=16
        out = p + out # Residual Connection

        return out

    def coord2dist(self, edge_index, x):
        # 논문 식 (4): difference between the coordinates of the nodes
        row, col = edge_index
        coord_diff = x[row] - x[col] # h_i - h_j, 차원: [(query*way)*(shot*way), hid_dim]
        sqr_dist = torch.sum(coord_diff**2, 1).unsqueeze(1) # ||h_i - h_j||^2, 차원: [(query*way)*(shot*way), 1]

        return sqr_dist, coord_diff

    def forward(self, edge_index, str_feature, coord_feature):
        row, col = edge_index # row: 시작 노드, col: 끝 노드
        sqr_dist, coord_diff = self.coord2dist(edge_index, coord_feature)
        msg = self.msg_model(str_feature[row], str_feature[col], sqr_dist) # 논문 식 (4) 계산
        coord_feature = self.coord_model(
            coord_feature, edge_index, coord_diff, msg) # 논문 식 (5) 계산, 차원: [way*(shot + query), hid_dim]
        str_feature = self.posi_model(str_feature, edge_index, msg) # 논문 식 (7) 계산, 차원: [way*(shot + query), raw_dim]

        return str_feature, coord_feature


class EGNN(nn.Module):
    def __init__(self, str_dim, in_dim, n_layers):
        super(EGNN, self).__init__()
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, EGCL(str_dim, in_dim)) 
        # "gcl_%d": 추가할 모듈 이름        
        # EGCL: EGCL 모듈을 추가
        # EGCL을 n_layers만큼 쌓은 형태
            
        self.n_layers = n_layers
        self.LayerNorm = nn.LayerNorm(in_dim) # in_dim: 64

    def forward(self, str_feature, coord_feature, edge_index):  # h = hiddin
        # str_feature: [way*(shot + query), anchor_size] = h^{(s)}
        # coord_feature: [way*(shot + query), feature_out_dim] = h^{(l)}
        # edge_index: [2, (query*way)*(shot*way)]
        coord_feature = self.LayerNorm(coord_feature) # 차원: [way*(shot + query), feature_out_dim]
        for i in range(0, self.n_layers):
            str_feature, coord_feature = self._modules["gcl_%d" % i](edge_index, str_feature, coord_feature)
            # EGCL 모듈의 forward 함수 호출

        return str_feature, coord_feature


def diff_mean(data, segment_ids, num_nodes):
    result_shape = (num_nodes, data.size(1)) # data: [(query*way)*(shot*way), hid_dim], segment_ids: [(query*way)*(shot*way),]
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1)) # [750, ] -> [750, 1] -> [750, 64]로 확장 
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0) # 해당 크기의 0으로 초기화된 tensor 생성
    result.scatter_add_(0, segment_ids, data) # data: [750, 64], segment_ids: [750, 64] -> data의 각 행을 segment_ids가 가리키는 result의 위치에 합산.
    count.scatter_add_(0, segment_ids, torch.ones_like(data)) # data와 동일한 크기의 1로 초기화된 tensor 생성 후, segment_ids가 가리키는 count의 위치에 1을 합산.
    return result / count.clamp(min=1) # 0이면 1로 변경 후, result를 count로 나누어 평균을 구함. 논문 식 (5) 참고


def msg_collect(data, segment_ids, num_nodes):
    result_shape = (num_nodes, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result # 논문 식 (6)
