import torch
import torch.nn as nn
import os
from argument import config2string
import numpy as np
from utils import *
import networkx as nx


class embedder(nn.Module): 
    def __init__(self, args, conf, set_seed):
        super().__init__()

        self.args = args
        self.conf = conf
        self.set_seed = set_seed
        self.config_str = config2string(args) # config를 string으로 변환)
        print("\n[Config] {}\n".format(self.config_str)) 

        # Select GPU device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        # torch.cuda.set_device(self.device) # 해당 device를 사용하도록 설정

        self.edges, self.adj, self.features, self.labels, self.degrees, self.class_list_train, self.class_list_valid, self.class_list_test, self.id_by_class, self.num_nodes = load_data(
            args.dataset)
        self.edges = self.edges.to(self.device)
        self.adj = self.adj.to(self.device)
        self.features = self.features.to(self.device)
        self.labels = self.labels.to(self.device)
        self.degrees = self.degrees.to(self.device)

        # _________________________
        # calculates shortest dists
        self.edges_hub = self.edges
        print("Generating a structural feature...")
        # anchor node 생성 및 연결
        for i in range(self.args.anchor_size): # virtual anchor node의 개수만큼 반복
            hub_index = len(self.features) + i # virtual anchor node의 index, features의 마지막 index부터 시작
            num_sample_node = int(len(self.features)/(2**(i+1))) # 논문에서 제시된 확률, 2^(i+1)로 나누어 anchor node와 연결될 노드 수 정하기 
            if num_sample_node < 1: # 샘플링된 노드의 개수가 1보다 작으면 1로 설정
                print(f'   Virtual Anchor Node [{i+1}] samples less than 1')
                num_sample_node = 1
            selected_nodes_with_hub_i = random.sample(
                list(range(len(self.features))), num_sample_node) # 전체 node 중에서 num_sample_node만큼 랜덤하게 샘플링
            edge_hub_i = torch.LongTensor(
                [hub_index] * len(selected_nodes_with_hub_i)) # anchor node의 인덱스를 num_sample_node만큼 복제하여, 연결되는 엣지의 시작점을 설정. (anhcor node와의 연결을 의미)
            edge_hub_j = torch.LongTensor(
                selected_nodes_with_hub_i)
            edge1_bi = torch.cat([edge_hub_i, edge_hub_j]) # 엣지를 표현하기 위해 엣지의 시작 노드(앵커)와 끝 노드(연결되는 노드)를 저장해두기 (가로로 붙이기)
            edge2_bi = torch.cat([edge_hub_j, edge_hub_i]) # bidirectional graph을 위해 시작 노드(연결되는 노드)와 끝 노드(앵커) 저장
            edge_index_hub_i = torch.stack(
                (edge1_bi, edge2_bi)).to(self.device) # 첫 번째 행: 엣지의 시작 노드, 두 번째 행: 엣지의 끝 노드
            self.edges_hub = torch.cat(
                [self.edges_hub, edge_index_hub_i], 1) # 기존 그래프의 엣지(self.edges_hub)에 새로운 anchor node와 연결된 엣지를 추가. 가로로 붙이기. 
        print("Done.\n")

        graph = nx.Graph() # networkx의 그래프 생성
        edge_list = self.edges_hub.transpose(1, 0).tolist() # 엣지 리스트 생성 (2x? -> ?x2, 양방향으로 구성됨)
        graph.add_edges_from(edge_list) # 엣지 리스트를 그래프에 추가(엣지 리스트는 (첫째 열: 시작 노드, 둘째 열: 끝 노드)로 구성됨)

        # 최단 경로 길이 계산 -> structural feature 생성
        structural_feature = [] # 논문에서 제시된 structural feature, 크기: [anchor 노드 개수, 그래프 내 모든 노드]
        for i in range(self.args.anchor_size):
            hub_index = len(self.features) + i # virtual anchor node의 index
            hub_i_feature = []
            spd_i = nx.single_source_shortest_path_length(
                graph, hub_index) # 해당 anchor node(hub_index)로부터의 최단 경로 길이 계산, {앵커 노드와 연결된 노드: 최단 경로 길이} 형태, 자기 자신의 경우 길이를 0으로 표현
            for j in range(len(self.features)): # 해당 anchor node와 연결된 노드에 대한 최단 경로 길이를 저장하여 해당 structural feature로 사용
                try:
                    hub_spd_ij = spd_i[j] # anchor node와 연결된 노드에 대한 최단 경로 길이 가져오기 
                except:
                    hub_spd_ij = np.inf # 최단 경로가 없는 경우 무한대로 설정
                hub_spd_ij = 1 / (hub_spd_ij+1) # 논문에서 제시된 식, s(v, u) = 1/(d(v, u)+1)
                hub_i_feature.append(hub_spd_ij) 
            structural_feature.append(hub_i_feature)

        self.structural_features = torch.Tensor(
            structural_feature).T.to(self.device) # 크기: [그래프 내 모든 노드, anchor 노드 개수]

        self.n_way = args.way
        self.k_shot = args.shot
        self.n_query = args.qry
