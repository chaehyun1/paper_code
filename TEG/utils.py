import numpy as np
import scipy.sparse as sp
import random
from sklearn import preprocessing
from sklearn.metrics import f1_score
# from torch_geometric.datasets import Coauthor
import torch_geometric
import scipy.sparse as sp
import torch
import scipy.io as sio
import pickle

valid_num_dic = {"Amazon_clothing": 17, "Amazon_electronics": 36, "dblp": 27}


def load_data(dataset_source):
    if dataset_source in ['Amazon_clothing', 'Amazon_electronics', 'dblp']: # 해당 데이터셋이면
        n1s = [] 
        n2s = []
        for line in open(f"./dataset/{dataset_source}/{dataset_source}_network"):
            n1, n2 = line.strip().split("\t")  # strip: 문자열 양쪽 공백 제거
            n1s.append(int(n1))
            n2s.append(int(n2))

        edges = torch.LongTensor([n1s, n2s]) # n1s와 n2s를 행렬로 변환

        num_nodes = max(max(n1s), max(n2s)) + 1 # 노드의 개수
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)), # 희소 행렬 생성
                            shape=(num_nodes, num_nodes))
        # 예시
        # n1s = [0, 1, 2]
        # n2s = [1, 2, 0]
        # num_nodes = 3
        # (0, 1)	1.0
        # (1, 2)	1.0
        # (2, 0)	1.0
        # 실제로 값이 있는 좌표만 저장, 연산 시에는 내부적으로 효율적인 계산 방식으로 동작하지만 결과는 일반적인 행렬 연산과 동일

        data_train = sio.loadmat(
            f"./dataset/{dataset_source}/{dataset_source}_train.mat") # mat 파일 로드, dictionary 형태로 반환
        train_class = list(
            set(data_train["Label"].reshape((1, len(data_train["Label"])))[0]) # 중복을 제거한 label list
        )

        data_test = sio.loadmat(
            f"./dataset/{dataset_source}/{dataset_source}_test.mat")
        class_list_test = list(
            set(data_test["Label"].reshape((1, len(data_test["Label"])))[0])
        )

        labels = np.zeros((num_nodes, 1)) # 노드의 개수만큼 0으로 초기화
        labels[data_train["Index"]] = data_train["Label"] # train 데이터의 label을 labels에 저장
        labels[data_test["Index"]] = data_test["Label"] # test 데이터의 label을 labels에 저장

        features = np.zeros((num_nodes, data_train["Attributes"].shape[1])) # (노드의 개수 x feature의 차원) 만큼 0으로 초기화
        features[data_train["Index"]] = data_train["Attributes"].toarray() # train 데이터의 feature를 해당 인덱스 위치에 저장
        features[data_test["Index"]] = data_test["Attributes"].toarray() 

        class_list = []
        for cla in labels: # node의 label을 하나씩 꺼내서
            if cla[0] not in class_list: # 해당 label이 class_list에 없으면
                class_list.append(cla[0]) # class_list에 추가

        id_by_class = {} # class 별로 id를 저장할 dictionary
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels): 
            id_by_class[cla[0]].append(id) # class 별로 id를 저장: 해당 클래스에 속하는 노드의 id를 저장

        lb = preprocessing.LabelBinarizer() # label을 one-hot encoding
        labels = lb.fit_transform(labels) # one-hot encoding된 label
        # 입력: labels = [[0], [1], [0], [2]]
        # 출력: labels = [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]

        degree = np.sum(adj, axis=1) # 각 노드의 degree, 행 방향 합산 
        degree = torch.FloatTensor(degree)

        adj = normalize_adj(adj + sp.eye(adj.shape[0])) # 정규화된 adjacency matrix
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1]) # one-hot encoding된 label을 다시 정수로 변환

        adj = sparse_mx_to_torch_sparse_tensor(adj) # PyTorch sparse tensor로 변환

        class_list_valid = random.sample(
            train_class, valid_num_dic[dataset_source]) # train_class에서 valid_num_dic[dataset_source]만큼 label을 랜덤하게 선택

        class_list_train = list(
            set(train_class).difference(set(class_list_valid))) # train_class에서 class_list_valid를 제외한 나머지 label을 선택

    elif dataset_source == 'corafull': # corafull 데이터셋이면
        cora_full = torch_geometric.datasets.CitationFull(
            './dataset', 'cora') # cora 데이터셋 로드

        edges = cora_full.data.edge_index # edge 정보, 크기: (2, num_edges) 첫 번째 행: 엣지의 시작 노드. 두 번째 행: 엣지의 끝 노드.

        num_nodes = max(max(edges[0]), max(edges[1])) + 1 # 노드의 개수

        adj = sp.coo_matrix((np.ones(len(edges[0])), (edges[0], edges[1])), 
                            shape=(num_nodes, num_nodes))

        features = cora_full.data.x # feature 정보
        labels = cora_full.data.y # label 정보

        class_list = cora_full.data.y.unique().tolist() # label의 unique한 값들을 list로 변환

        with open(file='./dataset/cora/cls_split.pkl', mode='rb') as f:
            class_list_train, class_list_valid, class_list_test = pickle.load(f) 

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels.tolist()):
            id_by_class[cla].append(id)

        lb = preprocessing.LabelBinarizer()
        labels = lb.fit_transform(labels)

        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)

        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1])

        adj = sparse_mx_to_torch_sparse_tensor(adj)

    elif dataset_source == 'coauthorCS':
        CS = Coauthor(root='./dataset/CS', name='CS') # CS 데이터셋 로드
        data = CS.data 

        edges = data.edge_index

        num_nodes = max(max(edges[0]), max(edges[1])) + 1

        adj = sp.coo_matrix((np.ones(len(edges[0])), (edges[0], edges[1])),
                            shape=(num_nodes, num_nodes))

        features = data.x
        labels = data.y

        class_list = data.y.unique().tolist()

        with open(file='./dataset/CS/cls_split.pkl', mode='rb') as f:
            class_list_train, class_list_valid, class_list_test = pickle.load(f)

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels.tolist()):
            id_by_class[cla].append(id)

        lb = preprocessing.LabelBinarizer()
        labels = lb.fit_transform(labels)

        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)

        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1])

        adj = sparse_mx_to_torch_sparse_tensor(adj)


    elif dataset_source == 'ogbn-arxiv':

        from ogb.nodeproppred import NodePropPredDataset 

        dataset = NodePropPredDataset(name = 'ogbn-arxiv') # ogbn-arxiv 데이터셋 로드

        split_idx = dataset.get_idx_split() # train, valid, test의 노드 인덱스, 딕셔너리 형태 
        # train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, labels = dataset[0] # graph: library-agnostic graph object
        # graph: dict, 그래프의 주요 정보를 포함
            # graph['edge_index']: 엣지 리스트. 크기 [2, num_edges] (COO 형식).
            # graph['node_feat']: 노드 특성 행렬. 크기 [num_nodes, feat_dim].
            # graph['num_nodes']: 노드의 총 개수.
        # labels: 노드의 클래스 레이블. 크기 [num_nodes, 1].
        
        n1s=graph['edge_index'][0] # edge의 시작 노드
        n2s=graph['edge_index'][1] # edge의 끝 노드

        edges = torch.LongTensor(graph['edge_index']) # edge 정보

        num_nodes = graph['num_nodes'] # 노드의 개수

        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                                        shape=(num_nodes, num_nodes))    
        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        features=torch.FloatTensor(graph['node_feat'])
        labels=torch.LongTensor(labels).squeeze() # squeeze: 차원이 1인 차원을 제거

        class_list = labels.unique().tolist()

        with open(file='./dataset/ogbn_arxiv/cls_split.pkl', mode='rb') as f:
            class_list_train, class_list_valid, class_list_test = pickle.load(f)

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels.tolist()):
            id_by_class[cla].append(id)


    elif dataset_source == 'citeseer':
        citeseer = torch_geometric.datasets.Planetoid(
            './dataset', 'CiteSeer')

        edges = citeseer.data.edge_index

        num_nodes = max(max(edges[0]), max(edges[1])) + 1

        adj = sp.coo_matrix((np.ones(len(edges[0])), (edges[0], edges[1])),
                            shape=(num_nodes, num_nodes))

        features = citeseer.data.x 
        labels = citeseer.data.y

        class_list = citeseer.data.y.unique().tolist()

        with open(file='./dataset/CiteSeer/cls_split.pkl', mode='rb') as f:
            class_list_train, class_list_valid, class_list_test = pickle.load(f)

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels.tolist()):
            id_by_class[cla].append(id)

        lb = preprocessing.LabelBinarizer()
        labels = lb.fit_transform(labels)

        degree = np.sum(adj, axis=1)
        degree = torch.FloatTensor(degree)

        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1])

        adj = sparse_mx_to_torch_sparse_tensor(adj)


    return (
        edges,
        adj,
        features,
        labels,
        degree,
        class_list_train,
        class_list_valid,
        class_list_test,
        id_by_class,
        num_nodes
    ) 
    # edges: edge 정보 (2 x num_edges)
    # adj: adjacency matrix
    # features: feature matrix
    # labels: label
    # degree: degree
    # class_list_train: train 데이터의 label list
    # class_list_valid: valid 데이터의 label list
    # class_list_test: test 데이터의 label list
    # id_by_class: class 별로 id를 저장한 dictionary
    # num_nodes: 노드의 개수


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj) # coo_matrix: 희소 행렬을 효율적으로 저장하는 방법
    rowsum = np.array(adj.sum(1)) # 행 방향으로 합산 = 각 노드의 degree
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^(-0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0 # 0으로 나누는 경우 inf가 발생할 수 있어서 0으로 대체
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # 대각 행렬로 변환
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^(-0.5) * A * D^(-0.5), coo_matrix로 변환

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    # output: 모델의 예측값, labels: 실제 값
    # output: (query*way, way), labels: (query*way, )
    preds = output.max(1)[1].type_as(labels) # 최대값의 인덱스(예측된 클래스)를 가져옴, type_as: labels의 데이터 타입으로 변환
    correct = preds.eq(labels).double() # 예측값과 실제 값이 같은 경우 1, 다른 경우 0
    correct = correct.sum() # 맞은 개수 합산
    return correct / output.shape[0] # acc 계산


def f1(output, labels):
    preds = output.max(1)[1].type_as(labels) 
    f1 = f1_score(labels, preds, average="weighted") # f1 score 계산
    return f1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32) # coo_matrix로 변환
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    ) # 행과 열 좌표를 세로로 쌓아 PyTorch 텐서로 변환 (2 x N), torch.sparse.FloatTensor에 해당하는 좌표 형태로 해야 함. (첫번째 행: 행 좌표, 두번째 행: 열 좌표)
    values = torch.from_numpy(sparse_mx.data) # COO 형식에서 각 좌표 (i, j)에 대응하는 실제 값들의 배열을 NumPy에서 PyTorch 텐서로 변환.
    shape = torch.Size(sparse_mx.shape) # 희소 행렬의 크기를 PyTorch 텐서로 변환
    return torch.sparse.FloatTensor(indices, values, shape) # PyTorch sparse tensor 생성.


def seed_everything(seed=0): 

    torch.manual_seed(seed) # CPU 연산 시 seed 고정
    torch.cuda.manual_seed(seed) # GPU 연산 시 seed 고정
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    
    # cudnn: GPU에서 딥러닝 모델의 계산을 가속화하는 라이브러리
    torch.backends.cudnn.deterministic = True # cudnn 연산 시 seed 고정
    torch.backends.cudnn.benchmark = False # cudnn 연산 시 최적화 사용하지 않음
    np.random.seed(seed) # numpy 연산 시 seed 고정
    random.seed(seed) # random 연산 시 seed 고정


def task_generator_in_class(
    id_by_class, selected_class_list, n_way, k_shot, m_query
):
    # sample class indices
    class_selected = selected_class_list
    id_support = []
    id_query = []

    for cla in class_selected:
        temp = random.sample(id_by_class[cla], k_shot + m_query) # class 별로 k_shot + m_query만큼 랜덤하게 샘플링
        id_support.extend(temp[:k_shot]) # k_shot만큼 support data에 node id 추가
        id_query.extend(temp[k_shot:]) # m_query만큼 query data에 node id 추가

    # return [0] (k-shot x n_way) support data id array
    #        [1] (n_query x n_way) query data id array
    #        [2] (n_way) selected class list
    return np.array(id_support), np.array(id_query), class_selected # 이 부분 차원 이상함. 
    # id_support: len가 shot*way인 리스트, id_query: len가 qry*way인 리스트, class_selected: way


def euclidean_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    # 위 주석 이상함. 
    n = x.size(0) # query*way
    m = y.size(0) # way
    d = x.size(1) # feature_out_dim

    assert d == y.size(1) # feature_out_dim가 같아야 함

    x = x.unsqueeze(1).expand(n, m, d) # x를 (N x 1 x D)로 확장하여 (N x M x D)로 만듦
    y = y.unsqueeze(0).expand(n, m, d) # y를 (1 x M x D)로 확장하여 (N x M x D)로 만듦

    # x - y: (N x M x D)에서 각 요소에 대해 element-wise 제곱 후 마지막 차원(D)을 합산
    return torch.pow(x - y, 2).sum(2)  # N x M
