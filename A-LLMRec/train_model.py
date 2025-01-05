import os
import torch
import random
import time
import os

from tqdm import tqdm

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from models.a_llmrec_model import *
from pre_train.sasrec.utils import data_partition, SeqDataset, SeqDataset_Inference


def setup_ddp(rank, world_size):
    # PyTorch의 분산 데이터 병렬(Distributed Data Parallel, DDP) 학습을 위한 초기 설정
    # 여러 GPU에서 병렬로 학습을 수행할 때 필요한 초기화 작업
    os.environ ["MASTER_ADDR"] = "localhost" 
    # MASTER_ADDR: 분산 학습을 할 때, 여러 프로세스가 서로 통신해야 하므로 마스터 노드(주로 학습을 시작하는 프로세스)의 주소를 설정
    # 여기서는 "localhost"로 설정되어 있어, 동일한 머신에서 실행된다는 의미
    os.environ ["MASTER_PORT"] = "12355" # 마스터 노드와 다른 프로세스들이 연결할 포트 번호를 설정
    init_process_group(backend="nccl", rank=rank, world_size=world_size) # 분산 학습을 위한 초기화 함수
    # rank는 0부터 시작하여 world_size - 1까지 할당
    # world_size는 전체 프로세스 수
    torch.cuda.set_device(rank) # 현재 프로세스에서 사용할 GPU 설정
    
def train_model_phase1(args): # pretrain_stage1
    print('A-LLMRec start train phase-1\n')
    if args.multi_gpu:
        world_size = torch.cuda.device_count() # 사용 가능한 GPU 수 저장 
        mp.spawn(train_model_phase1_, args=(world_size, args), nprocs=world_size) # 멀티 프로세싱을 활용한 학습을 시작
        # train_model_phase1_을 여러 프로세스에서 동시에 병렬적으로 실행함
        # 각 프로세스는 하나의 GPU에서 작업을 수행
    else:
        train_model_phase1_(0, 0, args) # 첫 번째 0은 GPU 인덱스, 두 번째 0은 프로세스 ID
        
def train_model_phase2(args):
    print('A-LLMRec strat train phase-2\n')
    if args.multi_gpu:
        world_size = torch.cuda.device_count()
        mp.spawn(train_model_phase2_, args=(world_size, args), nprocs=world_size)
    else:
        train_model_phase2_(0, 0, args)

def inference(args):
    print('A-LLMRec start inference\n')
    if args.multi_gpu:
        world_size = torch.cuda.device_count()
        mp.spawn(inference_, args=(world_size, args), nprocs=world_size)
    else:
        inference_(0,0,args)
  
def train_model_phase1_(rank, world_size, args):
    # rank: mp.spawn이 각 프로세스에 자동으로 전달하는 값, 0부터 nprocs - 1까지의 값을 가짐
    if args.multi_gpu:
        setup_ddp(rank, world_size) 
        args.device = 'cuda:' + str(rank)
        
    model = A_llmrec_model(args).to(args.device)
    
    # preprocess data
    dataset = data_partition(args.rec_pre_trained_data, path=f'./data/amazon/{args.rec_pre_trained_data}.txt') 
    [user_train, user_valid, user_test, usernum, itemnum] = dataset # 데이터셋 분할
    print('user num:', usernum, 'item num:', itemnum)
    num_batch = len(user_train) // args.batch_size1
    cc = 0.0 # sequence length의 합을 저장
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    # Init Dataloader, Model, Optimizer
    train_data_set = SeqDataset(user_train, usernum, itemnum, args.maxlen)
    if args.multi_gpu:
        train_data_loader = DataLoader(train_data_set, batch_size = args.batch_size1, sampler=DistributedSampler(train_data_set, shuffle=True), pin_memory=True)
        # train_data_set을 배치 단위로 로드
        # DistributedSampler: 분산 학습을 위한 데이터 로더
        # 분산학습: 여러 GPU가 동일한 모델을 학습하지만, 각 GPU는 전체 데이터셋의 일부만 처리
        # 분산학습: 각 GPU에서 모델의 복사본을 가지고 동기화된 학습을 진행. 학습이 끝난 후, 각 GPU에서 계산된 기울기를 동기화하여 모델을 업데이트
        # pin_memory=True: 데이터 로딩 속도를 높이기 위해 GPU 메모리로 고정된 메모리를 사용
        model = DDP(model, device_ids = [args.device], static_graph=True)
    else:
        train_data_loader = DataLoader(train_data_set, batch_size = args.batch_size1, pin_memory=True)        
        
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.stage1_lr, betas=(0.9, 0.98)) 
    
    epoch_start_idx = 1
    T = 0.0
    model.train()
    t0 = time.time()
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        if args.multi_gpu:
            train_data_loader.sampler.set_epoch(epoch) # 각 에폭마다 데이터 샘플링의 순서를 바꿔서 각 GPU에서 같은 데이터를 반복해서 학습하지 않도록 보장
        for step, data in enumerate(train_data_loader): # batch_num 만큼 반복 (전체=1000개, 배치 사이즈=100 이면 10번 반복)
            u, seq, pos, neg = data # 배치 사이즈 만큼 로드 
            u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
            model([u,seq,pos,neg], optimizer=adam_optimizer, batch_iter=[epoch,args.num_epochs + 1,step,num_batch], mode='phase1') 

            if step % max(10,num_batch//100) ==0: # 학습 중 일정 배치마다 모델을 저장
                if rank ==0: # rank가 0인 프로세스에서만 모델을 저장 (모델 저장을 여러 번 할 필요는 없음)
                    if args.multi_gpu: model.module.save_model(args, epoch1=epoch) # 모델 저장, (DDP)로 래핑되어 있어 실제 모델은 model.module에 있음
                    else: model.save_model(args, epoch1=epoch) # 모델 저장
        if rank == 0: # 각 에폭이 끝날 때 (배치 학습이 모두 끝났을 때) 저장 
            if args.multi_gpu: model.module.save_model(args, epoch1=epoch) 
            else: model.save_model(args, epoch1=epoch)

    print('train time :', time.time() - t0)
    if args.multi_gpu:
        destroy_process_group() # 분산 학습을 위한 프로세스 그룹을 제거
    return 

def train_model_phase2_(rank,world_size,args): # stage2
    if args.multi_gpu:
        setup_ddp(rank, world_size)
        args.device = 'cuda:'+str(rank)
    random.seed(0)

    model = A_llmrec_model(args).to(args.device)
    phase1_epoch = 10 # phase1에서 학습한 모델을 불러오기 위한 에폭 수
    model.load_model(args, phase1_epoch=phase1_epoch) # phase1에서 학습된 것 불러오기 -> stage 2에서도 사용됨 

    dataset = data_partition(args.rec_pre_trained_data, path=f'./data/amazon/{args.rec_pre_trained_data}.txt')
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    print('user num:', usernum, 'item num:', itemnum)
    num_batch = len(user_train) // args.batch_size2
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    
    # Init Dataloader, Model, Optimizer
    train_data_set = SeqDataset(user_train, usernum, itemnum, args.maxlen)
    if args.multi_gpu:
        train_data_loader = DataLoader(train_data_set, batch_size = args.batch_size2, sampler=DistributedSampler(train_data_set, shuffle=True), pin_memory=True)
        model = DDP(model, device_ids = [args.device], static_graph=True)
    else:
        train_data_loader = DataLoader(train_data_set, batch_size = args.batch_size2, pin_memory=True, shuffle=True)
        
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.stage2_lr, betas=(0.9, 0.98))
    
    epoch_start_idx = 1
    T = 0.0
    model.train()
    t0 = time.time()
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)):
        if args.multi_gpu:
            train_data_loader.sampler.set_epoch(epoch)
        for step, data in enumerate(train_data_loader): # 배치 단위로 데이터 로드
            u, seq, pos, neg = data
            u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
            model([u,seq,pos,neg], optimizer=adam_optimizer, batch_iter=[epoch,args.num_epochs + 1,step,num_batch], mode='phase2')
            if step % max(10,num_batch//100) ==0:
                if rank ==0:
                    if args.multi_gpu: model.module.save_model(args, epoch1=phase1_epoch, epoch2=epoch)
                    else: model.save_model(args, epoch1=phase1_epoch, epoch2=epoch)
        if rank == 0:
            if args.multi_gpu: model.module.save_model(args, epoch1=phase1_epoch, epoch2=epoch)
            else: model.save_model(args, epoch1=phase1_epoch, epoch2=epoch)
    
    print('phase2 train time :', time.time() - t0)
    if args.multi_gpu:
        destroy_process_group()
    return

def inference_(rank, world_size, args):
    if args.multi_gpu:
        setup_ddp(rank, world_size)
        args.device = 'cuda:' + str(rank)
        
    model = A_llmrec_model(args).to(args.device)
    phase1_epoch = 10
    phase2_epoch = 5
    model.load_model(args, phase1_epoch=phase1_epoch, phase2_epoch=phase2_epoch) # phase1, phase2에서 학습한 모델 불러오기

    dataset = data_partition(args.rec_pre_trained_data, path=f'./data/amazon/{args.rec_pre_trained_data}.txt')
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    print('user num:', usernum, 'item num:', itemnum)
    num_batch = len(user_train) // args.batch_size_infer 
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    model.eval() # 모델을 평가 모드로 설정
    
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000) # 10000명의 사용자만 추출
    else:
        users = range(1, usernum + 1) # 모든 사용자 추출
    
    user_list = []
    for u in users:
        if len(user_train[u]) < 1 or len(user_test[u]) < 1: continue # 사용자의 히스토리가 없거나 테스트 데이터가 없는 경우 제외
        user_list.append(u)

    inference_data_set = SeqDataset_Inference(user_train, user_valid, user_test, user_list, itemnum, args.maxlen) # validation 데이터를 train에 포함함.
    
    if args.multi_gpu:
        inference_data_loader = DataLoader(inference_data_set, batch_size = args.batch_size_infer, sampler=DistributedSampler(inference_data_set, shuffle=True), pin_memory=True)
        model = DDP(model, device_ids = [args.device], static_graph=True)
    else:
        inference_data_loader = DataLoader(inference_data_set, batch_size = args.batch_size_infer, pin_memory=True)
    
    for _, data in enumerate(inference_data_loader): # 배치 단위로 데이터 로드
        u, seq, pos, neg = data # pos는 test set의 아이템, neg는 seq와 상호작용이 없는 3개의 아이템 
        u, seq, pos, neg = u.numpy(), seq.numpy(), pos.numpy(), neg.numpy()
        model([u,seq,pos,neg, rank], mode='generate') # 모델을 사용하여 추론 수행