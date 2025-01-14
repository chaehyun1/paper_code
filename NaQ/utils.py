# Implementation mainly follows utils.py from
# @inproceedings{ding2020graph,
#   title={Graph prototypical networks for few-shot learning on attributed networks},
#   author={Ding, Kaize and Wang, Jianling and Li, Jundong and Shu, Kai and Liu, Chenghao and Liu, Huan},
#   booktitle={Proceedings of the 29th ACM International Conference on Information \& Knowledge Management},
#   pages={295--304},
#   year={2020}
# }

from json import load
import os
import numpy as np
import scipy.sparse as sp
import torch
import random

import torch_geometric
import torch_geometric.transforms as T

import scipy.io as sio
import pickle

from copy import deepcopy




def graph_diffusion(args, Data):
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    num_nodes = Data.features.size(0) # GraphData의 self.features의 행의 개수 = 노드의 개수
    # NaQ-Feat
    if args.type == 'feature': # NaQ-Feat
        features = Data.features
        def cos_sim_matrix(feat1, feat2):
            feat1 = torch.nn.functional.normalize(feat1) # feature를 normalize
            feat2 = torch.nn.functional.normalize(feat2)
            cos_sim = torch.mm(feat1, feat2.mT) # feature 간의 cosine similarity 계산, mT: matrix transpose
            return cos_sim
        Diffusion_matrix = cos_sim_matrix(features, features)
    elif args.type == 'diffusion': # NaQ-Diff
        # Calculation of Diffusion matrix
        gdc = T.GDC() # Graph Diffusion 객체 생성
        edge_weight = torch.ones(Data.edge_index.size(1), device=Data.edge_index.device) # edge_weight 초기화
        edge_index, edge_weight = gdc.transition_matrix(edge_index=Data.edge_index, edge_weight=edge_weight, num_nodes=num_nodes, normalization='sym') 
        # transition matrix T 계산
        # edge_weight를 degree를 활용하여 정규화된 값으로 갱신한다. 
         
        Diffusion_matrix = gdc.diffusion_matrix_exact(edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes, method='ppr', alpha=args.PPR_alpha)
        # diffusion matrix S 계산 

    # Not to sample node itself in query-generation stage.
    Diffusion_matrix = sparse_fill_diagonal_0_(Diffusion_matrix) if Diffusion_matrix.layout == torch.sparse_coo else Diffusion_matrix.fill_diagonal_(0) # 대각선 요소를 0으로 설정
    
    return Diffusion_matrix.to(device)



def normalize(mx):
    '''Row-normalize sparse matrix'''
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    '''Symmetrically normalize adjacency matrix.'''
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    '''Convert a scipy sparse matrix to a torch sparse tensor.'''
    sparse_mx = sparse_mx.tocoo().astype(np.float32) # coo_matrix로 변환
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    ) # 행과 열 좌표를 세로로 쌓아 PyTorch 텐서로 변환 (2 x N), torch.sparse.FloatTensor에 해당하는 좌표 형태로 해야 함. (첫번째 행: 행 좌표, 두번째 행: 열 좌표)
    values = torch.from_numpy(sparse_mx.data) # COO 형식에서 각 좌표 (i, j)에 대응하는 실제 값들의 배열을 NumPy에서 PyTorch 텐서로 변환.
    shape = torch.Size(sparse_mx.shape) # 희소 행렬의 크기를 PyTorch 텐서로 변환
    return torch.sparse.FloatTensor(indices, values, shape) # PyTorch sparse tensor 생성.

def sparse_tensor_to_sparse_mx(sparse_tensor):
    '''Convert a torch sparse tensor to a scipy sparse matrix'''
    sparse_tensor = sparse_tensor.cpu().detach().coalesce() # coalesce: 중복된 인덱스를 제거하고, 중복된 값들을 합침
    # 예시
    # indices: [[0, 0, 1], [1, 1, 2]] -> (0, 1), (0, 1), (1, 2) 의미
    # values: [3, 2, 4]
    # (0, 1)이 중복되므로, [[0, 1], [1, 2]] & (5, 4)로 합침
    row, col = sparse_tensor.indices()[0].numpy(), sparse_tensor.indices()[1].numpy() # indices: (2, non-zero elements), 위 예시 참고 
    value = sparse_tensor.values().numpy()
    shape = (sparse_tensor.size(0), sparse_tensor.size(1))
    return sp.coo_matrix((value, (row, col)), shape=shape) # 즉, 0과 대각선을 제외한 값들을 coo_matrix 형식으로 저장

# Functions for handling (2-D) sparse tensors
def sparse_fill_diagonal_0_(sparse_tensor):
    # sparse tensor인 경우, 대각선 요소를 제거
    coalesced = sparse_tensor.coalesce() # 중복된 인덱스를 제거하고, 중복된 위치의 값들을 합산
    indices = coalesced.indices() # 대각선 포함한 모든 위치의 인덱스, 첫ㅉ째 행: row index, 두 번째 행: column index
    values = coalesced.values()
    remaining_indices = (~(indices[0,:] == indices[1,:])).nonzero().flatten() # 대각선이 아니면서 0이 아닌 요소의 인덱스 남기기 
    indices = indices[:,remaining_indices]
    values = values[remaining_indices]
    return torch.sparse_coo_tensor(indices=indices, values=values, size=sparse_tensor.size())

def sparse_tensor_column_zeroing(sparse_tensor, col_indices):
    '''
    This method can be modified to remove or indexing specified 'sub'-matrix
    '''
    coalesced = sparse_tensor.coalesce()
    indices = coalesced.indices()
    values = coalesced.values()
    for ix in col_indices:
        remaining_indices = (indices[1,:]!=ix).nonzero().flatten()
        indices = indices[:,remaining_indices]
        values = values[remaining_indices]
    return torch.sparse_coo_tensor(indices=indices, values=values, size=sparse_tensor.size())

def sparse_topk_dim1(sparse_tensor, row_ix, k):
    '''
    Row-wise Top-k function for sparse matrices
    Please keep len(row_ix) be small to get efficiency.
    '''
    if type(row_ix) != list:
        row_ix = row_ix.tolist()
    coalesced = sparse_tensor.coalesce() # 중복된 인덱스를 제거하고, 중복된 위치의 값들을 합산
    val_temp, ix_temp = [], [] 
    for ix in row_ix: # spt node에 대해서 
        vals, ixs = sparse_tensor[ix].to_dense().topk(k=k) # 해당 node에 대한 top-Q node를 찾음
        # vals: 가장 큰 Q개의 값
        # ixs: 해당 값의 인덱스
        val_temp.append(vals)
        ix_temp.append(ixs)
    values, indices = torch.stack(val_temp), torch.stack(ix_temp) # dim=0으로 쌓음
    # values: [support node 수, Q], indices: [support node 수, Q]
    return values, indices


def seed_everything(seed=0):
    torch.manual_seed(seed) # CPU 연산 시 seed 고정
    torch.cuda.manual_seed(seed) # GPU 연산 시 seed 고정
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True # cudnn 연산 시 seed 고정
    torch.backends.cudnn.benchmark = False # cudnn 연산 시 최적화 사용하지 않음
    np.random.seed(seed) # numpy 연산 시 seed 고정
    random.seed(seed) # random 연산 시 seed 고정
