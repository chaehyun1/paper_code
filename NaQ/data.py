# The code source: https://github.com/kaize0409/GPN_Graph-Few-shot
# @inproceedings{ding2020graph,
#   title={Graph prototypical networks for few-shot learning on attributed networks},
#   author={Ding, Kaize and Wang, Jianling and Li, Jundong and Shu, Kai and Liu, Chenghao and Liu, Huan},
#   booktitle={Proceedings of the 29th ACM International Conference on Information \& Knowledge Management},
#   pages={295--304},
#   year={2020}
# }
# We modified the code source to utilize this in unsupervised settings.

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric
import random

from sklearn import preprocessing
from torch_geometric.datasets import CitationFull
import torch_geometric
import torch_geometric.transforms as T

import scipy.io as sio
import pickle

from copy import deepcopy
from utils import *



valid_num_dic = {'Amazon_clothing': 17, 'Amazon_eletronics': 37, 'dblp': 27}

def load_data(args):
    dataset_source = args.dataset
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    # If dataset is other than Amazon_clothing, Amazon_eletronics, dblp, then load these
    if dataset_source == 'Cora_full':
        Data = load_data_corafull(args) # GraphData 객체 반환
        return Data
    
    n1s = [] 
    n2s = []
    for line in open('./data/{}_network'.format(dataset_source)):
        n1, n2 = line.strip().split('\t') # n1: 시작 노드, n2: 끝 노드
        n1s.append(int(n1))
        n2s.append(int(n2))

    edge_index = torch.LongTensor([n1s, n2s]) # edge_index 생성, 차원: (2, num_edges)
    num_nodes = max(max(n1s), max(n2s)) + 1 # 총 노드의 개수
    adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)), shape=(num_nodes, num_nodes)) # adjacency matrix 생성

    data_train = sio.loadmat('./data/{}_train.mat'.format(dataset_source)) # train 데이터 로드, 형식: dictionary
    train_class = list(set(data_train['Label'].reshape((1, len(data_train['Label'])))[0])) # train 데이터의 클래스 정보

    data_test = sio.loadmat('./data/{}_test.mat'.format(dataset_source)) # test 데이터 로드, 형식: dictionary
    class_list_test = list(set(data_test['Label'].reshape((1, len(data_test['Label'])))[0]))

    labels = np.zeros((num_nodes, 1)) # 노드의 클래스 정보를 저장할 배열
    labels[data_train['Index']] = data_train['Label'] # 노드 순서에 맞게 클래스 정보 저장
    labels[data_test['Index']] = data_test['Label']
    
    features = np.zeros((num_nodes, data_train['Attributes'].shape[1])) # 노드의 feature 정보를 저장할 배열
    features[data_train['Index']] = data_train['Attributes'].toarray() # 노드 순서에 맞게 feature 정보 저장
    features[data_test['Index']] = data_test['Attributes'].toarray()
    
    class_list = []
    for cla in labels:
        if cla[0] not in class_list:
            class_list.append(cla[0])  # unsorted
    
    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(labels):
        id_by_class[cla[0]].append(id) # 각 클래스에 해당하는 노드의 인덱스 저장
    
    lb = preprocessing.LabelBinarizer() 
    labels = lb.fit_transform(labels) # one-hot encoding, 차원: (num_nodes, num_classes)

    degree = np.sum(adj, axis=1) # 각 노드의 degree 계산
    degree = torch.FloatTensor(degree)
    
    adj = normalize_adj(adj + sp.eye(adj.shape[0])) # adjacency matrix를 정규화
    
    features = torch.FloatTensor(features)
    
    labels = torch.LongTensor(np.where(labels)[1]) # np.where(labels)[1]: 1이 위치한 열 인덱스(label) 반환
    
    adj = sparse_mx_to_torch_sparse_tensor(adj) # torch tensor로 변환

    class_list_valid = random.sample(train_class, valid_num_dic[dataset_source]) # Validation class 선택

    class_list_train = list(set(train_class).difference(set(class_list_valid))) # Training class 선택
    
    # For unsupervised settings, only id_train is required.
    id_train = []
    for cla in class_list:
        id_train = id_train + id_by_class[cla] # 각 클래스에 해당하는 노드의 인덱스 저장

    Data = GraphData(edge_index, adj, features, degree, labels, class_list_train, class_list_valid, class_list_test, id_by_class, id_train)
    
    return Data



def load_data_corafull(args):
    dataset_source = args.dataset
    assert dataset_source == 'Cora_full' # Cora_full 데이터셋만 가능
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    dataset = CitationFull('./data/corafull', 'cora') # Cora_full 데이터셋 로드
    data = dataset[0] # 그래프 데이터 객체를 반환

    features = data.x # Node Feature Matrix, 차원: (num_nodes, num_node_features)
    edge_index = data.edge_index # 노드 간의 연결 정보, 차원: (2, num_edges), 첫 번째 행은 시작 노드, 두 번째 행은 끝 노드
    labels = data.y # 노드의 클래스 정보

    # Train, valid, test class spliting
    class_list = [] # 유니크 클래스 리스트 
    for cla in labels.tolist():
        if cla not in class_list:
            class_list.append(cla) # unsorted

    random.shuffle(class_list) # randomly shuffle given list of classes
    class_list_train, class_list_valid, class_list_test = class_list[:25], class_list[25:-25], class_list[-25:] # Training/Valid/Test class split = 25/20/25
    
    id_by_class = {}
    for i in class_list:
        id_by_class[i] = []
    for id, cla in enumerate(labels.tolist()):
        id_by_class[cla].append(id) # 각 클래스에 해당하는 노드의 인덱스 저장

    # For unsupervised settings, only id_train is required.
    id_train = [] 
    for cla in class_list:
        id_train = id_train + id_by_class[cla] # Training class에 해당하는 노드의 인덱스 저장

    adj = sp.coo_matrix((np.ones(edge_index.size(1)), (edge_index[0].tolist(), edge_index[1].tolist())), shape=(len(data.x), len(data.x)))
    # edge_index를 이용하여 adjacency matrix 생성

    degree = np.sum(adj, axis=1) # 각 노드의 degree 계산
    degree = torch.FloatTensor(degree)

    adj = normalize_adj(adj + sp.eye(adj.shape[0])) # adjacency matrix를 정규화
    adj = sparse_mx_to_torch_sparse_tensor(adj) # torch tensor로 변환

    Data = GraphData(edge_index, adj, features, degree, labels, class_list_train, class_list_valid, class_list_test, id_by_class, id_train)

    return Data




class GraphData:
    '''
    Class which stores Graph-Structured Data loaded from above load_data() function
    '''
    def __init__(self, edge_index, adj, features, degree, labels, class_list_train, class_list_valid, class_list_test, id_by_class, id_train):

        self.edge_index, self.adj, self.features, self.degree, self.labels = edge_index, adj, features, degree, labels
        
        self.class_list_train, self.class_list_valid, self.class_list_test, self.id_by_class, self.id_train = class_list_train, class_list_valid, class_list_test, id_by_class, id_train


    def set_device(self, device):
        if device != 'cpu':
            self.edge_index, self.adj, self.features, self.degree, self.labels = self.edge_index.to(device), self.adj.to(device), self.features.to(device), self.degree.to(device), self.labels.to(device)

        return

    def add_diffusion(self, args):
        Diffusion_matrix = graph_diffusion(args, self) # node similarity matrix or diffusion matrix 생성 -> S
        self.Diffusion_matrix = Diffusion_matrix

        return

    
    
