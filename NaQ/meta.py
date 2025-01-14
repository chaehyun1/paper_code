# Implementation for MAML, UnsupMAML, mainly follows meta.py from
# @misc{MAML_Pytorch,
#   author = {Liangqu Long},
#   title = {MAML-Pytorch Implementation},
#   year = {2018},
#   publisher = {GitHub},
#   journal = {GitHub repository},
#   howpublished = {\url{https://github.com/dragen1860/MAML-Pytorch}},
#   commit = {master}
# # }
# For ProtoNet implementation, we referred to https://github.com/kaize0409/GPN_Graph-Few-shot
# @inproceedings{ding2020graph,
#   title={Graph prototypical networks for few-shot learning on attributed networks},
#   author={Ding, Kaize and Wang, Jianling and Li, Jundong and Shu, Kai and Liu, Chenghao and Liu, Huan},
#   booktitle={Proceedings of the 29th ACM International Conference on Information \& Knowledge Management},
#   pages={295--304},
#   year={2020}
# }

import torch
import numpy as np
from torch import nn
from torch import optim

import torch.nn.functional as F
from torch.nn.parameter import Parameter

from model import GNNEncoder, LinearClassifier, GCN
from utils import *
from copy import deepcopy

from sklearn.linear_model import LogisticRegression



class MAML(nn.Module):
    def __init__(self, args, config, Data):
        super(MAML, self).__init__()
        self.config = config
        self.Data = Data
        self.network = GNNEncoder(config)
        self.dim_latent = args.latent

        self.n = args.n_way
        self.k = args.k_shot
        self.q = args.q_query
        self.meta_batch_size = args.meta_batch_size
        self.num_steps_meta = args.num_steps_meta
        self.inner_lr = args.inner_lr
        self.meta_update_lr = args.meta_update_lr

        self.classifier = LinearClassifier(config) # used in Meta-training
        self.l2_penalty = args.l2_penalty # used in Fine-tuning
        
        self.meta_optimizer = optim.Adam(list(self.network.parameters())+list(self.classifier.parameters()), lr=self.meta_update_lr)


    def forward(self, id_spt, y_spt, id_query, y_query):
        # Load Features/Adjacency matrix
        features, adj = self.Data.features, self.Data.adj

        query_size = self.n*self.q
        num_cls_params = len(list(self.classifier.parameters())) # linear classifier의 파라미터 수 
        if num_cls_params == 0:
            num_cls_params = -len(list(self.network.parameters()))

        # losses_query[j] = validation loss(loss of query) after jth update in inner-loop (j = 0, ..., self.num_steps_meta)
        # num_steps_meta: # of parameter updates for each tasks during the meta-training phase(inner loop)
        losses_query = [0 for _ in range(self.num_steps_meta+1)] 
        corrects = [0 for _ in range(self.num_steps_meta+1)] 
    
        #---------------- <Meta-Training & Loss recording phase(Inner-loop)> ----------------#
        for i in range(self.meta_batch_size): # 한 epoch의 meta_batch_size 수행
            
            # Get Loss & Spt embeddings for ith task for support samples before training
            encodings = self.network(features, vars=None, adj=adj) # 
            x_spt = encodings[id_spt[i]]
            logits = self.classifier(x_spt, vars=None)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, list(self.network.parameters())+list(self.classifier.parameters()))
            # 1st update of model parameters
            weights_updated = list(map(lambda f: f[1] - self.inner_lr*f[0], zip(grad, list(self.network.parameters())+list(self.classifier.parameters()))))

            # Get Query embeddings & Record loss and accuracy for query samples before the 1st update for the meta-update phase
            with torch.no_grad():
                x_query = encodings[id_query[i]]
                logits_query = self.classifier(x_query, vars=self.classifier.parameters())
                loss_query = F.cross_entropy(logits_query, y_query[i])
                losses_query[0] += loss_query
                
                pred_query = F.softmax(logits_query, dim=1).argmax(dim=1)
                correct = torch.eq(pred_query, y_query[i]).sum().item()
                corrects[0] = corrects[0] + correct
            
            # Get Query embeddings & Record loss and accuracy after the 1st update
            encodings = self.network(features, vars=weights_updated[:-num_cls_params], adj=adj)
            x_query = encodings[id_query[i]]
            logits_query = self.classifier(x_query, vars=weights_updated[-num_cls_params:])
            loss_query = F.cross_entropy(logits_query, y_query[i])
            losses_query[1] += loss_query

            with torch.no_grad():
                pred_query = F.softmax(logits_query, dim=1).argmax(dim=1)
                correct = torch.eq(pred_query, y_query[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for j in range(2, self.num_steps_meta+1):
                # (1) Get Spt embeddings & loss for ith task with weights after (j-1)th update (j = 2, ..., self.num_steps_meta)
                x_spt = encodings[id_spt[i]]
                logits = self.classifier(x_spt, vars=weights_updated[-num_cls_params:])
                loss = F.cross_entropy(logits, y_spt[i])
                # (2) Get gradient at current parameter
                grad = torch.autograd.grad(loss, weights_updated)
                # (3) jth update of model parameter
                weights_updated = list(map(lambda f: f[1] - self.inner_lr*f[0], zip(grad, weights_updated)))
                
                # (4) Record loss and accuracy after the jth update for the meta-update phase
                encodings = self.network(features, vars=weights_updated[:-num_cls_params], adj=adj)
                x_query = encodings[id_query[i]]
                logits_query = self.classifier(x_query, vars=weights_updated[-num_cls_params:])
                loss_query = F.cross_entropy(logits_query, y_query[i])
                losses_query[j] += loss_query

                with torch.no_grad():
                    pred_query = F.softmax(logits_query, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_query, y_query[i]).sum().item()
                    corrects[j] = corrects[j] + correct
        #------------------------------------------------------------------------------------#

        #------------------------- <Meta Update Phase(Outer-loop)> --------------------------#
        # Use loss of query samples by using final updated parameter
        final_loss_query = losses_query[-1] / self.meta_batch_size

        # Meta Update
        self.meta_optimizer.zero_grad()
        final_loss_query.backward()
        self.meta_optimizer.step()

        # calculating training accuracy by using final updated parameter
        final_acc = corrects[-1] / (query_size*self.meta_batch_size)
        #------------------------------------------------------------------------------------#

        return final_loss_query, final_acc


    def fine_tuning(self, id_spt, y_spt, id_query, y_query):
        # Load Features/Adjacency matrix
        features, adj= self.Data.features, self.Data.adj

        assert id_spt.shape[0] != self.meta_batch_size

        # Fine-tune the copied model to prevent model learning in test phase
        net_GNN = deepcopy(self.network)
        
        # Get Embeddings
        encodings = net_GNN(features, vars=None, adj=adj).detach().cpu().numpy()
        x_spt, x_query, y_spt, y_query = encodings[id_spt], encodings[id_query], y_spt.detach().cpu().numpy(), y_query.detach().cpu().numpy()

        clf = LogisticRegression(max_iter=1000, C=1/self.l2_penalty).fit(x_spt, y_spt)

        # delete copied net that we used
        del net_GNN

        # Get Test Accuracy
        test_acc = clf.score(x_query, y_query)

        return None, test_acc


class ProtoNet(nn.Module):
    '''
    Prototypical Network that trains GNN encoder
    '''
    def __init__(self, args, config, Data):
        super(ProtoNet, self).__init__()
        self.args = args
        self.config = config
        self.Data = Data
        self.network = GNNEncoder(config)
        self.dim_latent = args.latent
        
        self.n = args.n_way
        self.k = args.k_shot
        self.q = args.q_query
        if self.args.setting == 'unsup':
            self.q_test = args.q_query_test
        
        self.lr = args.lr

        self.l2_penalty = args.l2_penalty

        self.meta_optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
        
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'


    def forward(self, id_spt, y_spt, id_query, y_query):
        '''
        Forward Call
        If those are not None, then it is ordinary supervised or unsupervised NaQ setting.
        '''
        features, adj = self.Data.features, self.Data.adj
        
        encodings = self.network(features, adj=adj)
        x_spt, x_query = encodings[id_spt], encodings[id_query]
        
        prototypes = x_spt.view(self.n, self.k, x_spt.size(1)).mean(dim=1)
        dists = self.euclidean_dist(x_query, prototypes)

        output = F.log_softmax(-dists, dim=1)

        # to take care of permutation during task-generation phase
        # 1shot setting or NaQ setting
        if (self.k == 1):
            label_new = torch.LongTensor([y_spt.tolist().index(i) for i in y_query.tolist()]).to(self.device)
        else:
            compressed = y_spt.detach().view(-1, self.k).float().mean(dim=1).long()
            label_new = torch.LongTensor([compressed.tolist().index(i) for i in y_query.tolist()]).to(self.device)
        loss = F.nll_loss(output, label_new)
        
        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()

        train_acc = self.accuracy(output.cpu().detach(), label_new.cpu().detach())

        return loss, train_acc
    

    def fine_tuning(self, id_spt, y_spt, id_query, y_query):
        # Load Features/Adjacency matrix
        features, adj= self.Data.features, self.Data.adj
        
        # Fine-tune the copied model to prevent model learning in test phase
        net_GNN = deepcopy(self.network)
        
        # Get Embeddings
        encodings = net_GNN(features, vars=None, adj=adj).detach().cpu().numpy()
        x_spt, x_query, y_spt, y_query = encodings[id_spt], encodings[id_query], y_spt.detach().cpu().numpy(), y_query.detach().cpu().numpy()

        clf = LogisticRegression(max_iter=1000, C=1/self.l2_penalty).fit(x_spt, y_spt)

        # delete copied net that we used
        del net_GNN

        # Get Test Accuracy
        test_acc = clf.score(x_query, y_query)

        return None, test_acc
   
        
    # for utils
    def euclidean_dist(self, x, y):
        assert x.size(1) == y.size(1)
        n, m, d = x.size(0), y.size(0), x.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x-y, 2).sum(dim=2)
    
    def accuracy(self, output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)





class UnsupMAML(nn.Module):
    def __init__(self, args, config, Data):
        super(UnsupMAML, self).__init__()
        self.config = config 
        self.Data = Data
        self.network = GNNEncoder(config) 
        self.dim_latent = args.latent # 64

        self.n = args.n_way
        self.k = args.k_shot
        self.q = args.q_query
        self.q_test = args.q_query_test
        self.meta_batch_size = args.meta_batch_size
        self.num_steps_meta = args.num_steps_meta
        self.inner_lr = args.inner_lr
        self.meta_update_lr = args.meta_update_lr

        self.classifier = LinearClassifier(config) # used in Meta-training
        self.l2_penalty = args.l2_penalty # used in Fine-tuning
        
        self.meta_optimizer = optim.Adam(list(self.network.parameters())+list(self.classifier.parameters()), lr=self.meta_update_lr)
        # meta_optimizer를 통해 두 모델의 파라미터를 동시에 업데이트

    def forward(self, id_spt, y_spt, id_query, y_query):
        # id_spt, y_spt: (meta_batch_size, n_way)
        # Load Features/Adjacency matrix
        features, adj = self.Data.features, self.Data.adj # adj: sparse tensor matrix 형식 (indicies, values, shape)

        query_size = self.n*self.q # 쿼리 총 개수  
        num_cls_params = len(list(self.classifier.parameters())) # linear classifier의 파라미터 수 
        if num_cls_params == 0:
            num_cls_params = -len(list(self.network.parameters()))  
 
        # losses_query[j] = validation loss(loss of query) after jth update in inner-loop (j = 0, ..., self.num_steps_meta)
        # num_steps_meta: # of parameter updates for each tasks during the meta-training phase(inner loop)
        # 0(업데이트하기 전)부터 self.num_steps_meta(num_steps_meta번 업데이트한 후)까지의 loss를 저장
        losses_query = [0 for _ in range(self.num_steps_meta+1)] 
        corrects = [0 for _ in range(self.num_steps_meta+1)] 
    
        #---------------- <Meta-Training & Loss recording phase(Inner-loop)> ----------------#
        for i in range(self.meta_batch_size): # 한 epoch 내 여러 meta-task 수행
            
            # Get Loss & Spt embeddings for ith task for support samples before training
            # 주의: support set에 대한 것
            encodings = self.network(features, vars=None, adj=adj) # GNNEncoder의 forward 함수, f_\theta
            x_spt = encodings[id_spt[i]] # encodings에서 해당 support sample만 추출, [n_way, args.latent]
            logits = self.classifier(x_spt, vars=None) # LinearClassifier의 forward 함수
            # logits: 각 샘플이 각 클래스에 속할 점수, [n_way(각 샘플), n_way(클래스)]
            loss = F.cross_entropy(logits, y_spt[i]) # y_spt[i]: [n_way]
            # 1. logits의 각 행에 대해 softmax를 적용하여 클래스별 확률 분포를 계산
            # 2. 한 행의 요소 중 정답 클래스에 해당하는 확률을 뽑아서 negative log likelihood를 계산 
            # 3. 모든 행에 대한 평균을 구해 loss를 계산
            grad = torch.autograd.grad(loss, list(self.network.parameters())+list(self.classifier.parameters())) # loss에 대한 gradient 계산 (각 파라미터에 대응하는 gradient)
            # 1st update of model parameters
            weights_updated = list(map(lambda f: f[1] - self.inner_lr*f[0], zip(grad, list(self.network.parameters())+list(self.classifier.parameters()))))
            # zip(~): [(grad_1, param_1), (grad_2, param_2), ..., (grad_n, param_n)] -> lambda의 input으로 들어감
            # lambda f: f[1] - self.inner_lr * f[0] 
            # weights_updated = [param_1_updated, param_2_updated, ..., param_n_updated]

            # Get Query embeddings & Record loss and accuracy for query samples before the 1st update for the meta-update phase
            # 주의: query set에 대한 것으로, 파라미터를 업데이트 하기 전 성능 평가 
            with torch.no_grad(): # 학습 과정이 아니므로, gradient 계산을 하지 않음
                x_query = encodings[id_query[i]] # 해당 query sample만 추출
                logits_query = self.classifier(x_query, vars=self.classifier.parameters()) # 학습된 파라미터 사용
                loss_query = F.cross_entropy(logits_query, y_query[i])
                losses_query[0] += loss_query # record loss
                
                pred_query = F.softmax(logits_query, dim=1).argmax(dim=1) # 각 쿼리 샘플에 대한 예측 클래스
                correct = torch.eq(pred_query, y_query[i]).sum().item() # 정답과 일치하는 개수
                corrects[0] = corrects[0] + correct # record accuracy
            
            # Get Query embeddings & Record loss and accuracy after the 1st update
            # 주의: query set에 대한 것으로, 파라미터를 업데이트 한 후 성능 평가
            encodings = self.network(features, vars=weights_updated[:-num_cls_params], adj=adj) # 학습된 파라미터 사용
            x_query = encodings[id_query[i]]
            logits_query = self.classifier(x_query, vars=weights_updated[-num_cls_params:])
            loss_query = F.cross_entropy(logits_query, y_query[i])
            losses_query[1] += loss_query # record loss
            
            with torch.no_grad():
                pred_query = F.softmax(logits_query, dim=1).argmax(dim=1)
                correct = torch.eq(pred_query, y_query[i]).sum().item()
                corrects[1] = corrects[1] + correct # record accuracy

            # 나머지에 대해서도 동일한 과정 반복(앞에서 이미 쿼리에 대해서 loss와 accuracy를 계산했으므로 2부터 시작)
            for j in range(2, self.num_steps_meta+1): 
                # 주의: support set에 대한 것, j-1번째 업데이트 이후의 성능 평가
                # (1) Get Spt embeddings & loss for ith task with weights after (j-1)th update (j = 2, ..., self.num_steps_meta)
                x_spt = encodings[id_spt[i]]
                logits = self.classifier(x_spt, vars=weights_updated[-num_cls_params:])
                loss = F.cross_entropy(logits, y_spt[i])
                # (2) Get gradient at current parameter
                grad = torch.autograd.grad(loss, weights_updated)
                # (3) jth update of model parameter
                weights_updated = list(map(lambda f: f[1] - self.inner_lr*f[0], zip(grad, weights_updated)))
                
                # (4) Record loss and accuracy after the jth update for the meta-update phase
                # 주의: query set에 대한 것으로, j번째 업데이트 이후 성능 평가
                encodings = self.network(features, vars=weights_updated[:-num_cls_params], adj=adj)
                x_query = encodings[id_query[i]]
                logits_query = self.classifier(x_query, vars=weights_updated[-num_cls_params:])
                loss_query = F.cross_entropy(logits_query, y_query[i])
                losses_query[j] += loss_query

                with torch.no_grad():
                    pred_query = F.softmax(logits_query, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_query, y_query[i]).sum().item()
                    corrects[j] = corrects[j] + correct
        #------------------------------------------------------------------------------------#

        #------------------------- <Meta Update Phase(Outer-loop)> --------------------------#
        # Use loss of query samples by using final updated parameter
        final_loss_query = losses_query[-1] / self.meta_batch_size  # inner-loop가 끝난 후 최종 query loss

        # Meta Update
        self.meta_optimizer.zero_grad() # gradient 초기화(역전파 전 필요)
        final_loss_query.backward() # backpropagation
        self.meta_optimizer.step() # parameter 업데이트

        # calculating training accuracy by using final updated parameter
        final_acc = corrects[-1] / (query_size*self.meta_batch_size) # inner-loop가 끝난 후 최종 query accuracy
        #------------------------------------------------------------------------------------#

        return final_loss_query, final_acc


    def fine_tuning(self, id_spt, y_spt, id_query, y_query):
        # 일반적인 supervised 분류로 fine-tuning 수행 -> 성능 평가 
        # Load Features/Adjacency matrix
        features, adj= self.Data.features, self.Data.adj

        assert id_spt.shape[0] != self.meta_batch_size # id_spt: (n_way*k_shot,)

        # Fine-tune the copied model to prevent model learning in test phase
        net_GNN = deepcopy(self.network) 
        # deepcopy: 내용을 수정하더라도 원래 모델에 영향을 주지 않음 
        # 중요: meta-training에서 학습된 모델을 가지고 fine-tuning을 수행
        
        # Get Embeddings
        encodings = net_GNN(features, vars=None, adj=adj).detach().cpu().numpy() # 학습된 모델 사용
        # detach(): 역전파를 위한 기울기 계산에서 제외(autograd에서 제외)
        # cpu(): Numpy 배열로 변환하려면 CPU 메모리에 텐서가 있어야 함
        x_spt, x_query, y_spt, y_query = encodings[id_spt], encodings[id_query], y_spt.detach().cpu().numpy(), y_query.detach().cpu().numpy()

        clf = LogisticRegression(max_iter=1000, C=1/self.l2_penalty).fit(x_spt, y_spt) # Logistic Regression 모델 학습(일반적인 supervised 분류 모델)

        # delete copied net that we used
        del net_GNN

        # Get Test Accuracy
        test_acc = clf.score(x_query, y_query) # test accuracy 계산

        return None, test_acc
    

