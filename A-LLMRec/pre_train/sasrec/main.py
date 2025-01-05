import os
import time
import torch
import argparse
import numpy as np
from model import SASRec
from data_preprocess import *
from utils import *

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True) # 필수 입력
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, action='store_true')
parser.add_argument('--state_dict_path', default=None, type=str)

args = parser.parse_args()

if __name__ == '__main__':
    
    # global dataset
    preprocess(args.dataset) # 전처리 및 데이터셋 생성
    dataset = data_partition(args.dataset) # train/valid/test set 생성(test set은 마지막 item만 사용, valid set은 마지막 이전의 1개 item 사용)

    [user_train, user_valid, user_test, usernum, itemnum] = dataset # dataset에서 반환된 값들을 unpacking
    print('user num:', usernum, 'item num:', itemnum)
    num_batch = len(user_train) // args.batch_size # batch size에 따른 batch 수 계산
    
    cc = 0.0 # sequence length의 합
    for u in user_train: # user_train: {user: [item1, item2, ...]}
        cc += len(user_train[u]) # user가 평가한 item 수
    print('average sequence length: %.2f' % (cc / len(user_train))) # user 평균 sequence length 출력
    
    # dataloader
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)       
    
    # model init
    model = SASRec(usernum, itemnum, args).to(args.device) 
    
    for name, param in model.named_parameters(): # 모델의 파라미터에 대해 이름과 값 출력
        try:
            torch.nn.init.xavier_normal_(param.data) # Xavier initialization
        except:
            pass
    
    model.train() # 학습 모드로 설정
     
    epoch_start_idx = 1 # 시작 epoch
    if args.state_dict_path is not None: # state_dict_path가 입력되었을 때
        try:
            kwargs, checkpoint = torch.load(args.state_dict_path, map_location=torch.device(args.device)) # state_dict_path로부터 모델의 파라미터를 불러옴
            # kwargs: 모델의 인자, checkpoint: 모델의 파라미터
            kwargs['args'].device = args.device # device 설정
            model = SASRec(**kwargs).to(args.device) # model init
            model.load_state_dict(checkpoint) # 모델의 파라미터를 불러옴
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:] # state_dict_path에서 'epoch=' 이후의 문자열을 tail로 설정
            epoch_start_idx = int(tail[:tail.find('.')]) + 1 # tail에서 '.' 이전의 문자열을 epoch_start_idx로 설정
            # 이전에 학습한 모델의 파라미터를 불러와서 이어서 학습 진행
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace() # pdb 디버거 실행
    
    if args.inference_only: # inference_only가 True일 때
        model.eval() # 평가 모드로 설정
        t_test = evaluate(model, dataset, args) # evaluate 함수를 통해 NDCG, HR 계산
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
    
    
    # 학습
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98)) # Adam optimizer
    
    T = 0.0
    t0 = time.time()
    
    for epoch in tqdm(range(epoch_start_idx, args.num_epochs + 1)): # 이어서 학습할 epoch부터 args.num_epochs까지 반복
        if args.inference_only: 
            break # inference_only가 True일 때 break
        
        for step in range(num_batch): # 배치 수만큼 반복
            u, seq, pos, neg = sampler.next_batch()  
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            pos_logits, neg_logits = model(u, seq, pos, neg) # model foward 실행
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)

            adam_optimizer.zero_grad() # optimizer gradient 초기화
            indices = np.where(pos != 0) # pos가 0이 아닌 index
            loss = bce_criterion(pos_logits[indices], pos_labels[indices]) # positive sample에 대한 loss 계산
            loss += bce_criterion(neg_logits[indices], neg_labels[indices]) # negative sample에 대한 loss 계산
            for param in model.item_emb.parameters():  
                loss += args.l2_emb * torch.norm(param) # L2 regularization (loss에 추가적인 페널티 항)
            loss.backward() # gradient 계산
            adam_optimizer.step()  # 파라미터 업데이트
            if step % 100 == 0: # 100번째 step마다 loss 출력
                print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
    
        if epoch % 20 == 0 or epoch == 1: # 20번째 epoch마다 또는 첫 epoch일 때
            model.eval() # 평가 모드로 설정
            t1 = time.time() - t0 # 시간 계산
            T += t1 # 시간 누적
            print('Evaluating', end='')
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print('\n')
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            print(str(t_valid) + ' ' + str(t_test) + '\n')
            t0 = time.time()
            model.train() # 학습 모드로 설정
    
        if epoch == args.num_epochs: # 마지막 epoch일 때 모델 저장
            folder = args.dataset 
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
            if not os.path.exists(os.path.join(folder, fname)): 
                try:
                    os.makedirs(os.path.join(folder))
                except:
                    print()
            torch.save([model.kwargs, model.state_dict()], os.path.join(folder, fname)) # 모델의 인자와 파라미터 저장
    
    sampler.close()
    print("Done")