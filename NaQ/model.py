# Implementation was referred to learner.py from
# @misc{MAML_Pytorch,
#   author = {Liangqu Long},
#   title = {MAML-Pytorch Implementation},
#   year = {2018},
#   publisher = {GitHub},
#   journal = {GitHub repository},
#   howpublished = {\url{https://github.com/dragen1860/MAML-Pytorch}},
#   commit = {master}
# }

from builtins import NotImplementedError
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter


class GNNEncoder(nn.Module):
    def __init__(self, config):
        super(GNNEncoder, self).__init__()

        self.config = config
        # containing parameters of GNN Encoder
        self.vars = nn.ParameterList() # 학습 가능한 파라미터를 저장하기 위해서 사용
        # config의 첫번째 요소의 weight와 bias, 두번째 요소의 weight와 bias, ...를 저장
        
        for name, size, bias_ in self.config:
            if name == 'GCN':
                weight = Parameter(torch.ones(*size))
                # *size: tuple을 개별 요소로 분해하여 전달 
                # Parameter: 학습 가능한 파라미터를 정의할 때 사용
                
                torch.nn.init.xavier_uniform_(weight) # xavier initialization
                self.vars.append(weight) 
                if bias_ is True:
                    bias = Parameter(torch.zeros(size[1])) # bias = 0으로 초기화: 학습 초기에 출력이 입력 데이터와 가중치에만 의존하여 안정적인 시작이 가능
                    self.vars.append(bias)
                else:
                    bias = None
                    self.vars.append(bias)

    def forward(self, x, vars=None, adj=None):
        # var: 학습 가능한 파라미터를 저장하는 변수
        if vars is None: 
            vars = self.vars
        
        idx = 0
        for name, _, _ in self.config:
            # 2*i th: weight of ith layer
            # 2*i+1 th: bias of ith layer if exists
            if name == 'GCN': # GCN인 경우에만 실행 
                weight, bias = vars[2*idx], vars[2*idx+1]
                x = torch.mm(x, weight) 
                x = torch.sparse.mm(adj, x)
                if bias is not None:
                    x = x + bias
                x = F.relu(x)
                idx += 1
            else:
                continue
        return x # 차원: [num_nodes, args.latent]

    def parameters(self):

        return self.vars # GNNEncoder의 파라미터 반환



class LinearClassifier(nn.Module):
    
    def __init__(self, config) -> None:
        super(LinearClassifier, self).__init__()
        self.config = config

        self.vars = nn.ParameterList()
        
        for name, size, bias_ in self.config:
        # 'linear': linear classifier using node embeddings embedded by GNN Encoders
        # size = (in_feat, out_feat) if name == 'linear'
            if name == 'linear': 
                weight = Parameter(torch.ones(*size))
                torch.nn.init.xavier_uniform_(weight)
                self.vars.append(weight)
                if bias_ is True:
                    bias = Parameter(torch.zeros(size[1]))
                    self.vars.append(bias)
                else: 
                    bias = None
                    self.vars.append(bias)
            else:
                continue
    
    def forward(self, x, vars=None):

        if vars == None: # vars가 None이 아닌 경우는 평가 단계인 경우 (이미 학습된 파라미터를 사용)
            vars = self.vars
        
        idx = 0
        for name, _, _ in self.config:
            if name == 'linear': # Linear인 경우에만 실행 
                weight, bias = vars[2*idx], vars[2*idx+1]
                x = torch.mm(x, weight)
                if bias is not None:
                    x = x + bias
                idx += 1
            else:
                continue
        # x 차원: [num_nodes, args.n_way]
        return x


    def parameters(self):

        return self.vars # Linear Classifier의 파라미터 반환
        



        
class GCN(nn.Module):
    
    def __init__(self, config) -> None:
        super(GCN, self).__init__()
        
        self.config = config
        # containing parameters of GNN Encoder
        self.vars = nn.ParameterList()

        for name, size, bias_ in self.config:
            if name == 'GCN':
                weight = Parameter(torch.ones(*size))
                torch.nn.init.xavier_uniform_(weight)
                self.vars.append(weight)
                if bias_ is True:
                    bias = Parameter(torch.zeros(size[1]))
                    self.vars.append(bias)
                else:
                    bias = None
                    self.vars.append(bias)

            else:
                continue

    
    def forward(self, x, vars=None, adj=None):
        
        if vars is None:
            vars = self.vars

        idx = 0
        
        for name, _, _, in self.config:
            if name == 'GCN':
                weight, bias = vars[2*idx], vars[2*idx+1]
                x = torch.mm(x, weight)
                x = torch.sparse.mm(adj, x)
                if bias is not None:
                    x = x + bias
                idx += 1
            elif name == 'relu':
                x = F.relu(x)
            elif name == 'dropout':
                x = F.dropout(x)
            elif name == 'linear':
                continue

        return x

    def parameters(self):

        return self.vars


