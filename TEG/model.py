from calendar import c
from layers.GCN import GCN
from layers.EGNN import EGNN
from embedder import embedder
from tqdm.auto import tqdm
from utils import *
from torch import optim
import torch.nn.functional as F
import torch
from argument import config2string, parse_args


class teg_trainer(embedder): # TEG 모델, embedder를 상속받음
    def __init__(self, args, conf, set_seed):
        embedder.__init__(self, args, conf, set_seed) # 부모 클래스의 __init__ 호출
        self.conv = GCN(self.features.shape[1],
                        conf['gcn_out'], args.dropout).to(self.device)
        self.egnn = EGNN(self.structural_features.shape[1], conf['egnn_in'],
                         n_layers=args.n_layers).to(self.device)

        self.optim = optim.Adam([
            {'params': self.conv.parameters()},
            {'params': self.egnn.parameters()}],
            lr=args.lr, weight_decay=5e-4
        ) 

        self.config_str = config2string(args) 
        self.set_seed = set_seed

    def train_epoch(self, mode, n_episode, epoch):

        loss_fn = torch.nn.NLLLoss() # Negative Log Likelihood Loss

        if mode == 'train':
            if epoch != 0: 
                self.conv.train() # 학습 모드로 설정
                self.egnn.train()
            else:
                self.conv.eval() # 평가 모드로 설정
                self.egnn.eval()

        else: # mode가 train이 아닌 경우
            self.conv.eval() # 평가 모드로 설정
            self.egnn.eval()

        if mode == 'train' or mode == 'valid':
            loss_epoch = 0 # Loss 초기화

        acc_epoch = []
        f1_epoch = []

        for episode in range(n_episode): # n_episode만큼 반복

            if mode == 'train':
                self.optim.zero_grad() # 새로운 episode마다 gradient를 초기화해야 함

            if mode == 'train':
                class_selected = random.sample(
                    self.class_list_train, self.args.way) # train 데이터셋의 label에서 way만큼 랜덤하게 샘플링

            elif mode == 'valid':
                class_selected = random.sample(
                    self.class_list_valid, self.args.way)

            elif mode == 'test':
                class_selected = random.sample(
                    self.class_list_test, self.args.way)

            id_support, id_query, class_selected = task_generator_in_class(
                self.id_by_class, class_selected, self.n_way, self.k_shot, self.n_query) # support set, query set 생성
            # class_selected에 있는 label 순서대로 해당 label에 대한 id_support, id_query 생성
            # id_support 크기: [shot*way, ], id_query 크기: [query*way, ] (1차원)

            # ________________
            # graph conv (GCN)
            embeddings = self.conv(self.features, self.edges) # GCN forward, graph embedder 부분, obtain the semantic feature = h^{(l)}
            # 참고: graph embedder는 anchor node는 사용하지 않고, 기존의 노드에 대해서만 적용 
            # GCN을 graph embedder으로 사용하여 embeddings를 얻고, 이에 대해 support set과 query set으로 나눈다. 
            # 크기: [num_nodes, feature_out_dim]

            # _____________
            # Task sampling
            embeds_spt = embeddings[id_support] # support set의 embedding, 크기: [shot*way, feature_out_dim], id_support에 해당하는 노드 임베딩
            embeds_qry = embeddings[id_query] # query set의 embedding, 크기: [query*way, feature_out_dim]

            embeds_epi = torch.cat([embeds_spt, embeds_qry]) # 차원: [way*(shot + query), feature_out_dim], 세로로 붙이기
            # support set과 query set의 embedding을 합침, dim=0으로 붙임

            # structural features = h^{(s)}, 차원: [num_nodes, anchor_size]
            spt_str = self.structural_features[id_support] # support set의 structural feature, 해당 노드와 anchor node 사이의 최단 경로 길이
            qry_str = self.structural_features[id_query] # query set의 structural feature

            epi_str = torch.cat([spt_str, qry_str]) # support set과 query set의 structural feature를 합침, 차원: [way*(shot + query), anchor_size]

            # _______________________________
            # calculate a graph embedder loss
            gcn_spt = embeds_epi[:len(id_support), :] # support set의 embedding, 차원: [shot*way, feature_out_dim]
            gcn_qry = embeds_epi[len(id_support):, :] # query set의 embedding, 차원: [query*way, feature_out_dim]
            gcn_spt = gcn_spt.view(
                [self.n_way, self.k_shot, gcn_spt.shape[1]]) # support set의 embedding을 [way, shot, feature_out_dim]로 변환
            proto_embeds_gcn = gcn_spt.mean(1) # 논문 식 (11), p_c^{(G)}, 차원: [way, feature_out_dim]
            dists_gcn = euclidean_dist( 
                gcn_qry, proto_embeds_gcn)  # 논문 식 (12)에 필요한 euclidean_dist 구하기, 차원: [query*way, way]
            output_gcn = F.log_softmax(-dists_gcn, dim=1) # 논문 식 (13)에 필요한 것

            # ___________________
            # Task-specific graph
            edge1 = [] # 차원: [query*way*shot, ], 시작이 query 노드인 엣지의 시작점 저장 리스트
            for i in range(len(id_support), len(id_support)+len(id_query)): # query set의 각 노드에 대해
                temp = [i] * len(id_support) # 각 query 노드가 support 노드들과 연결
                edge1.extend(temp) 
            edge1 = torch.LongTensor(edge1) # 시작이 쿼리 노드, 끝이 support set의 노드인 엣지 생성
            edge2 = torch.LongTensor(
                list(range(len(id_support))) * len(id_query)) # 차원: [(query*way)*(shot*way), ], bidirectional graph를 위해 역방향 엣지 생성
            edge1_bi = torch.cat([edge1, edge2])
            edge2_bi = torch.cat([edge2, edge1])
            edge_index = torch.stack((edge1_bi, edge2_bi)).to(self.device) # 차원: [2, (query*way)*(shot*way)], 첫 번째 행: 엣지의 시작 노드, 두 번째 행: 엣지의 끝 노드
            # 정리: 각 쿼리 노드와 support set의 모든 노드를 연결하는 양방향 엣지 생성

            # ______________________
            # EGNN - Task adaptation
            epi_str, embeds_epi = self.egnn(epi_str, embeds_epi, edge_index) # EGNN forward, task embedder
            # EGNN을 task embedder으로 사용하여 embeds_epi을 얻고, 이에 대해 support set과 query set으로 나눈다. 
            # epi_str = z^{(s)}, embeds_epi = z^{(l)}
            # epi_str: z^{(s)}는 실질적으로 사용되진 않음, h^{(l)}를 업데이트하는 과정에서 h^{(s)}가 사용됨. 
            # epi_str: [way*(shot + query), raw_dim], embeds_epi: [way*(shot + query), hid_dim], hid_dim = 64

            # __________
            # Prototypes
            embeds_spt = embeds_epi[:len(id_support), :]
            embeds_qry = embeds_epi[len(id_support):, :]

            embeds_spt = embeds_spt.view(
                [self.n_way, self.k_shot, embeds_spt.shape[1]])

            embeds_proto = embeds_spt.mean(1) # 논문 식 (8), p_c^{(N)}, 차원: [way, hid_dim]

            # __________
            # Prediction
            dists_output = euclidean_dist(
                embeds_qry, embeds_proto) # 논문 식 (9), euclidean distance 계산, 차원: [query*way, way]
            output = F.log_softmax(-dists_output, dim=1) # 논문 식 (10)에 필요한, 차원: [query*way, way]
            output_softmax = F.softmax(-dists_output, dim=1) # 논문 식 (9)

            # _________________________
            # Relabeling for meta-tasks
            label_list = torch.LongTensor(
                [class_selected.index(i)
                    for i in self.labels[id_query]] 
            ).to(self.device) # query set에 해당하는 노드에 대한 실제 레이블을 가져와서, class_selected의 해당 레이블의 index로 변환
            # 차원: [query*way, ]
            # 원래 라벨을 class_selected에 맞게 0부터 재매핑, loss 계산 시 일관성을 유지하기 위해 필요
            # 예시
            # class_selected: [7, 13, 35]
            # query set의 라벨: [7, 35, 13, 13, 35] -> [0, 2, 1, 1, 2] (재매핑 후)
            # 모델 출력(logit): [0.1, 0.8, 0.1] -> softmax -> [0.3, 0.4, 0.3] 
            # 실제 label이 1일 때 NLLLoss 계산: -log(0.4) = 0.9163
            # 이 확률이 높아지도록 학습한다. = loss가 최소화되도록 학습한다. 

            if mode == 'train' or mode == 'valid':

                # ___________
                # Loss update
                loss_l1_train = loss_fn(
                    output, label_list)  # Network Loss, 논문 식 (10)
                loss_l2_train = loss_fn(
                    output_gcn, label_list)  # Graph Embedder Loss, 논문 식 (13)

                loss_train = self.args.gamma*loss_l1_train + (1-self.args.gamma)*loss_l2_train # 논문 식 (14)
                # loss가 낮아지도록 task embedder와 graph embedder를 학습한다.

            if mode == 'train':
                if epoch != 0:
                    loss_train.backward() # loss_train 역전파
                    self.optim.step() # 파라미터 업데이트
                else:
                    self.optim.zero_grad() # gradient 초기화

            # ________
            # Accuracy
            output = output_softmax.cpu().detach() # detach: 학습과 무관한 값일 때 사용
            label_list = label_list.cpu().detach()

            acc_score = accuracy(output, label_list) 
            f1_score = f1(output, label_list)

            acc_epoch.append(acc_score)
            f1_epoch.append(f1_score)

        acc_total_epoch = sum(acc_epoch) / len(acc_epoch) # epoch별 accuracy 평균
        f1_total_epoch = sum(f1_epoch) / len(f1_epoch) # epoch별 f1 score 평균

        if mode == 'train':
            tqdm.write(f"acc_train : {acc_total_epoch:.4f}")

        elif mode == 'valid':
            tqdm.write(f"acc_valid : {acc_total_epoch:.4f}")

        elif mode == 'test':
            tqdm.write(f"acc_test : {acc_total_epoch:.4f}")

        return acc_total_epoch, f1_total_epoch

    def train(self):

        # _____________
        # Best Accuracy
        best_acc_train = 0
        best_f1_train = 0
        best_epoch_train = 0
        
        best_acc_valid = 0
        best_f1_valid = 0
        best_epoch_valid = 0
        
        best_acc_test = 0
        best_f1_test = 0
        best_epoch_test = 0

        for epoch in tqdm(range(self.args.epochs+1)): 

            acc_train, f1_train = self.train_epoch(
                'train', self.args.episodes, epoch)

            with torch.no_grad(): # gradient 계산을 하지 않음
                # 학습된 모델을 사용하여 valid set과 test set에 대한 성능을 평가

                acc_valid, f1_valid = self.train_epoch(
                    'valid', self.args.meta_val_num, epoch)

                acc_test, f1_test = self.train_epoch(
                    'test', self.args.meta_test_num, epoch) 

            if best_acc_train < acc_train:
                best_acc_train = acc_train
                best_f1_train = f1_train
                best_epoch_train = epoch

            if best_acc_valid < acc_valid:
                best_acc_valid = acc_valid
                best_f1_valid = f1_valid
                best_epoch_valid = epoch

            if best_acc_test < acc_test:
                best_acc_test = acc_test
                best_f1_test = f1_test
                best_epoch_test = epoch

            if acc_valid == best_acc_valid: # valid set에서 가장 높은 성능을 보인 epoch에서의 test set에 대한 성능 저장
                # 최적의 파라미터를 적용하여 test set에 대해 평가
                test_acc_at_best_valid = acc_test
                test_f1_at_best_valid = f1_test

            tqdm.write(f"# Current Settings : {self.config_str}")
            tqdm.write(
                f"# Best_Acc_Train : {best_acc_train:.4f}, F1 : {best_f1_train:.4f} at {best_epoch_train} epoch"
            )
            tqdm.write(
                f"# Best_Acc_Valid : {best_acc_valid:.4f}, F1 : {best_f1_valid:.4f} at {best_epoch_valid} epoch"
            )
            tqdm.write(
                f"# Best_Acc_Test : {best_acc_test:.4f}, F1 : {best_f1_test:.4f} at {best_epoch_test} epoch"
            )
            tqdm.write(
                f"# Test_At_Best_Valid : {test_acc_at_best_valid:.4f}, F1 : {test_f1_at_best_valid:.4f} at {best_epoch_valid} epoch\n"
            )

        np.set_printoptions(
            formatter={'float_kind': lambda x: "{0:0.4f}".format(x)}) # print할 때 소수점 4자리까지만 출력

        return best_acc_train, best_f1_train, best_epoch_train, best_acc_valid, best_f1_valid, best_epoch_valid, best_acc_test, best_f1_test, best_epoch_test, test_acc_at_best_valid, test_f1_at_best_valid
