import os
import numpy as np
import scipy.sparse as sp
import torch
import random
import pickle

from copy import deepcopy

from utils import *



def downstream_task_generator(args, Data, n_way, k_shot, q_query=8):
    '''
    Downstream Task Generator to Consistent Comparison by saving them.

    Query is set to be 8. This is consistent for all baselines and all settings
    '''   
    # If saved downstream task for current setting exists, load the data and return it.
    if os.path.exists(f'./save/downstream_tasks/{args.dataset}/{n_way}way_{k_shot}shot.pkl'):
        with open(f'./save/downstream_tasks/{args.dataset}/{n_way}way_{k_shot}shot.pkl', 'rb') as tasks:
            valid_pool, test_pool = pickle.load(tasks)
        return valid_pool, test_pool
    
    # If saved downstream task for current setting does not exist, then we generate the tasks and save it.
    os.makedirs(f'./save/downstream_tasks/{args.dataset}/', exist_ok=True)
    
    valid_num, test_num = args.num_valid_tasks, args.num_downstream_tasks # 50, 500

    # supervised 방식으로 생성 
    # 학습은 unsupervised 방식으로 생성한 데이터를 가지고 진행하고, downstream task(FSNC)에서 평가할때는 supervised 방식으로 생성된 데이터를 사용하여 검증한다. 
    valid_pool = [supervised_task_generator(Data.id_by_class, Data.class_list_valid, Data.labels, n_way, k_shot, q_query) for _ in range(valid_num)]
    test_pool = [supervised_task_generator(Data.id_by_class, Data.class_list_test, Data.labels, n_way, k_shot, q_query) for _ in range(test_num)]

    tasks = (valid_pool, test_pool)
    with open(f'./save/downstream_tasks/{args.dataset}/{n_way}way_{k_shot}shot.pkl', 'wb') as f:
        pickle.dump(tasks, f)

    return valid_pool, test_pool 


# Task-generator for MAML-like algorithms (which require meta-batches)
def meta_batch_generator(args, Data):
    if args.setting == 'sup':
        id_by_class, class_list, labels = Data.id_by_class, Data.class_list_train, Data.labels
        id_spts, y_spts, id_queries, y_queries = [], [], [], []
        # id_spts: (meta_batch_size, n_way * k_shot)
        # id_queries: (meta_batch_size, n_way * q_query)
        
        for _ in range(args.meta_batch_size): # 한 meta task 내 여러개의 batch를 생성하자
            id_spt, y_spt, id_query, y_query = supervised_task_generator(id_by_class, class_list, labels, args.n_way, args.k_shot, args.q_query)
            # class_list_train 중 n_way개의 class를 랜덤하게 선택하여, 각 class별로 k_shot개의 sample을 spt로, 나머지를 query로 생성
            
            id_spts.append(id_spt)
            y_spts.append(y_spt.view(1, -1)) # 텐서를 2D 형태로 변환, 하나의 리스트에 torch.tensor를 추가하는 것. 
            id_queries.append(id_query)
            y_queries.append(y_query.view(1, -1)) 
        
        id_spts, id_queries = np.array(id_spts), np.array(id_queries)
        y_spts, y_queries = torch.cat(y_spts, 0).long(), torch.cat(y_queries, 0).long() # 0: 행 방향으로 쌓기
        # y_spts 리스트 안의 요소인 torch tensor에 대해 torch tensor의 행을 기준으로 쌓기
        # ex
        # y_spts = [torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]]), torch.tensor([[7, 8, 9]])]
        # torch.cat(y_spts, 0) 결과:
        # tensor([[1, 2, 3],
                # [4, 5, 6],
                # [7, 8, 9]])
        
        return id_spts, y_spts, id_queries, y_queries
    
    if args.setting == 'unsup':
        # supervised와 다르게 Data.class_list_train를 사용하지 않는다. 
        id_train = Data.id_train # 전체 train 데이터의 노드 인덱스
        diffusion = True
        spt_to_sample = 1 # only 'UMTRA' style initial spt generation is considered.
        # 1-shot support set: 샘플링된 support set 노드들이 서로 더 구별할 수 있도록 한다. 
        qry_to_gen = args.q_query
        
        # When 'neighbors' are utilized as queries
        if args.query_generation == 'NaQ':
            id_spts, y_spts, id_queries, y_queries = [], [], [], []
            
            for _ in range(args.meta_batch_size):
                # initial spt sample sampling
                id_spt, y_spt = spt_sampling(id_train, args.n_way, spt_to_sample) # y_spt: pseudo-labels

                # query generation by NaQ
                id_query, y_query = query_generation_NaQ(id_spt, y_spt, qry_to_gen, Data, diffusion)
    
                id_spts.append(id_spt) 
                y_spts.append(y_spt.view(1,-1))
                id_queries.append(id_query)
                y_queries.append(y_query.view(1,-1))

            id_spts, y_spts = np.array(id_spts), torch.cat(y_spts, 0).long()
            id_queries, y_queries = np.array(id_queries), torch.cat(y_queries, 0).long()

            return id_spts, y_spts, id_queries, y_queries



# Task-generator for ProtoNet-like algorithms
def proto_task_generator(args, Data):
    if args.setting == 'sup':
        id_by_class, class_list, labels = Data.id_by_class, Data.class_list_train, Data.labels
        id_spt, y_spt, id_query, y_query = supervised_task_generator(id_by_class, class_list, labels, args.n_way, args.k_shot, args.q_query)

        return id_spt, y_spt, id_query, y_query
    
    if args.setting == 'unsup':
        id_train = Data.id_train
        diffusion = True
        spt_to_sample = 1 # only 'UMTRA' style initial spt generation is considered.
        qry_to_gen = args.q_query

        # When NaQ query generation is utilized
        if args.query_generation == 'NaQ':
            id_spt, y_spt = spt_sampling(id_train, args.n_way, spt_to_sample)

            # Query generation by NaQ
            id_query, y_query = query_generation_NaQ(id_spt, y_spt, qry_to_gen, Data, diffusion) 
            # id_query: top-Q similar node의 인덱스, y_query: top-Q similar node의 pseudo-label
            
            return id_spt, y_spt, id_query, y_query
            # 정리
            # id_spt: support set의 노드 인덱스, y_spt: support set의 pseudo-label
            # id_query: query set의 노드 인덱스, y_query: query set의 pseudo-label


def spt_sampling(id_train, n_way, k_shot):
    '''
    Unsupervised Task Generator to be used in my algorithm
    Used in Training Phase
    Here, we only return support set indices/pseudo-labels(id_support and y_spt)
    '''
    # 1. Sample just sample indices
    id_support = random.sample(id_train, n_way*k_shot) # 차원: [n_way*k_shot,]
    # 2. Give them random labels so that satisfying n_way-k_shot setting
    # 2-1) making random labels
    rand_train_labels = [] # [n_way * k_shot,]
    for i in range(n_way):
        labels = [i for _ in range(k_shot)]
        rand_train_labels += labels
    # 2-2) assign pseudo-labels randomly
    random.shuffle(rand_train_labels) # pseudo-label 생성 
    y_spt = torch.LongTensor(rand_train_labels)

    return np.array(id_support), y_spt


def query_generation_NaQ(id_spt, y_spt, to_gen, Data, diffusion=True):
    '''
    Query generation algorithm based on Diffusion matrix.
    '''
    if diffusion is True:
        # Find top-q similar nodes
        if Data.Diffusion_matrix.layout != torch.sparse_coo:
            _, nbr_ix = Data.Diffusion_matrix[id_spt].topk(k=to_gen, dim=1) # id_spt의 각각의 노드에 대해서 top-Q similar nodes를 찾기 
            # nbr_ix: (n_way*1, to_gen)
        else:
            _, nbr_ix = sparse_topk_dim1(Data.Diffusion_matrix, id_spt, k=to_gen)
            # nbr_ix: (n_way*1, to_gen), nbr_ix = top-Q similar node의 인덱스 
        nbr_ix = nbr_ix.cpu().tolist()
        # Get query samples
        id_query, y_query = [], []
        for i in range(id_spt.shape[0]): # 노드 수만큼 반복
            id_query += nbr_ix[i] # 각 노드에 대해 top-Q similar node의 인덱스를 추가
            y_query += [y_spt[i].item() for _ in range(len(nbr_ix[i]))] # top-Q similar node의 개수만큼 pseudo-label 추가
        id_query = np.array(id_query)
        y_query = torch.LongTensor(y_query)

    else:
        raise NotImplementedError

    # id_query: (n_way*to_gen,) y_query: (n_way*to_gen,)
    # id_query: top-Q similar node의 인덱스, y_query: top-Q similar node의 pseudo-label
    return id_query, y_query


def supervised_task_generator(id_by_class, class_list, labels, n_way, k_shot, q_query):
    '''
    Usual supervised few-shot task generator from GPN code
    Used in Fine-tuning Phase in Unsupervised Setting
    '''
    # sample class indices
    class_selected = random.sample(class_list, n_way) # n_way classes 추출 
    # sample n_way-k-shot/q_query samples
    id_support = []
    id_query = []
    for cla in class_selected:
        temp = random.sample(id_by_class[cla], k_shot + q_query) # 추출된 class 각각에 대한 node 샘플링 (k_shot + q_query개 만큼)
        id_support.extend(temp[:k_shot])
        id_query.extend(temp[k_shot:])
    # New labels for Support/Query samples
    y_spt = torch.LongTensor([class_selected.index(i) for i in labels[np.array(id_support)]]) # Support set에 대한 새로운 label 생성
    y_query = torch.LongTensor([class_selected.index(i) for i in labels[np.array(id_query)]]) # Query set에 대한 새로운 label 생성
    # 샘플링된 각 클래스에 대해 0~(n_way-1)까지의 label을 부여하기 위함

    return np.array(id_support), y_spt, np.array(id_query), y_query


