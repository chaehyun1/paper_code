import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
import os
from datetime import datetime
from pytz import timezone
from torch.utils.data import Dataset

    
# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r) # l과 r 사이의 랜덤한 정수 생성
    while t in s:
        t = np.random.randint(l, r) # 생성된 정수가 s에 포함되어 있으면 다시 생성
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():
        user = np.random.randint(1, usernum + 1) # user id를 랜덤하게 선택
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1) # user가 평가한 item 수가 1개 이하이면 다시 선택

        seq = np.zeros([maxlen], dtype=np.int32) # user의 sequence
        pos = np.zeros([maxlen], dtype=np.int32) # positive item
        neg = np.zeros([maxlen], dtype=np.int32) # negative item
        nxt = user_train[user][-1] # user의 마지막 item
        idx = maxlen - 1 # sequence의 index

        ts = set(user_train[user]) 
        for i in reversed(user_train[user][:-1]): # user가 평가한 item 중 마지막 item을 제외한 item들에 대해
            seq[idx] = i # sequence에 item 추가
            pos[idx] = nxt # positive item 추가
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts) # 상호작용하지 않은 item 중 하나를 negative item으로 추가
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg) # user, sequence, positive item, negative item 반환

    np.random.seed(SEED) 
    while True: # 정해진 queue size만큼 데이터를 생성
        one_batch = [] 
        for i in range(batch_size): # 한 배치에 속하는 batch_size만큼의 데이터를 생성
            one_batch.append(sample())

        result_queue.put(zip(*one_batch)) # 생성된 데이터를 Queue에 추가


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        # sample_function을 여러 프로세스에서 병렬적으로 실행하여 데이터 샘플링 속도를 향상
        self.result_queue = Queue(maxsize=n_workers * 10) # 샘플링된 데이터를 프로세스 간 공유하기 위해 Queue를 사용
        self.processors = [] 
        for i in range(n_workers): 
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      ))) # sample_function을 여러 worker 프로세스에서 병렬로 수행하도록 설정, args는 sample_function의 인자들
            self.processors[-1].daemon = True # 데몬 프로세스로 설정된 마지막 프로세트는 메인 프로세스가 종료되면 함께 종료
            self.processors[-1].start() # 마지막 프로세스는 메인 프로세스와 밀접하게 연관된 보조적인 흐름을 관리하는 역할

    def next_batch(self):
        return self.result_queue.get() # Queue에서 데이터를 가져옴

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

# DataSet for ddp
class SeqDataset(Dataset):
    def __init__(self, user_train, num_user, num_item, max_len): # train set만 활용 
        self.user_train = user_train
        self.num_user = num_user
        self.num_item = num_item
        self.max_len = max_len
        print("Initializing with num_user:", num_user)

        
    def __len__(self): # 사용자 수 반환
        return self.num_user 
        
    def __getitem__(self, idx):
        user_id = idx + 1
        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)
    
        nxt = self.user_train[user_id][-1] # user의 마지막 item
        length_idx = self.max_len - 1 
        
        # user의 seq set
        ts = set(self.user_train[user_id]) 
        for i in reversed(self.user_train[user_id][:-1]): # 마지막 item을 제외한 item들에 대해
            seq[length_idx] = i 
            pos[length_idx] = nxt # 각 아이템의 후속 아이템을 저장
            if nxt != 0: # 마지막 아이템이 아닌 경우
                neg[length_idx] = random_neq(1, self.num_item + 1, ts) # 상호작용하지 않은 item 중 하나를 negative item으로 추가
            nxt = i
            length_idx -= 1
            if length_idx == -1: break # 정해진 길이를 초과하면 종료

        # 해당 사용자의 seq, seq 내 각 아이템의 next 아이템(pos), 상호작용하지 않은 neg를 생성하여 반환
        return user_id, seq, pos, neg

class SeqDataset_Inference(Dataset):
    def __init__(self, user_train, user_valid, user_test, use_user, num_item, max_len): 
        self.user_train = user_train
        self.user_valid = user_valid
        self.user_test = user_test
        self.num_user = len(use_user) 
        self.num_item = num_item
        self.max_len = max_len
        self.use_user = use_user
        print("Initializing with num_user:", self.num_user)

    def __len__(self):
        return self.num_user
        
    def __getitem__(self, idx):
        user_id = self.use_user[idx] # 사용자 id
        seq = np.zeros([self.max_len], dtype=np.int32) 
        idx = self.max_len -1
        seq[idx] = self.user_valid[user_id][0] # valid set의 item을 sequence에 추가
        idx -=1
        for i in reversed(self.user_train[user_id]):
            seq[idx] = i
            idx -=1
            if idx ==-1: break
        # 즉, seq에 valid set item 및 최신의 train set item을 추가 (max_len만큼)
        rated = set(self.user_train[user_id]) 
        rated.add(0) # 패딩을 위한 0 추가
        pos = self.user_test[user_id][0] # test set의 item
        neg = []
        for _ in range(3): # negative item 3개 생성
            t = np.random.randint(1,self.num_item+1) 
            while t in rated: t = np.random.randint(1,self.num_item+1)
            neg.append(t)
        neg = np.array(neg) 
        return user_id, seq, pos, neg

# train/val/test data generation
def data_partition(fname, path=None):
    usernum = 0
    itemnum = 0
    User = defaultdict(list) # {user: [item1, item2, ...]}, userid가 처음 등장하면 defaultdict가 자동으로 빈 리스트를 생성
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    
    # f = open('./pre_train/sasrec/data/%s.txt' % fname, 'r')
    if path == None:
        f = open('../../data/amazon/%s.txt' % fname, 'r')
    else:
        f = open(path, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u) # user id
        i = int(i) # item id
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i) 

    for user in User: # user: user id
        nfeedback = len(User[user]) # user가 평가한 item 수
        if nfeedback < 3: # user가 평가한 item 수가 3개 미만이면 valid, test set을 생성할 수 없음
            user_train[user] = User[user] # user가 평가한 item을 모두 train set으로 사용
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2] # user가 평가한 item 중 마지막 2개를 제외한 item을 train set으로 사용
            user_valid[user] = []
            user_valid[user].append(User[user][-2]) # user가 평가한 item 중 마지막 2번째 item을 valid set으로 사용
            user_test[user] = []
            user_test[user].append(User[user][-1]) # user가 평가한 item 중 마지막 item을 test set으로 사용
    return [user_train, user_valid, user_test, usernum, itemnum] # user_train: {user: [item1, item2, ...]}, user_valid: {user: [item]}, user_test: {user: [item]}

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset) # data unpacking

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0 

    if usernum > 10000: 
        users = random.sample(range(1, usernum + 1), 10000) # user 수가 10000명 이상이면 10000명만 샘플링
    else:
        users = range(1, usernum + 1) # user 수가 10000명 미만이면 모든 user 사용
    
    for u in users: # user id에 대해
        if len(train[u]) < 1 or len(test[u]) < 1: continue # user의 item 수가 1개 미만이면 다음 user로 넘어감

        seq = np.zeros([args.maxlen], dtype=np.int32) # user의 sequence
        idx = args.maxlen - 1 # sequence의 index
        seq[idx] = valid[u][0] # valid set의 item을 sequence에 추가
        idx -= 1 
        for i in reversed(train[u]): # user가 평가한 item 중 마지막 item부터 역순으로 sequence에 추가
            seq[idx] = i
            idx -= 1
            if idx == -1: break
            
        rated = set(train[u]) 
        rated.add(0) # 패딩을 위한 0 추가
        item_idx = [test[u][0]]  # test set: 1 positive item, 19 negative items
        for _ in range(19): # negative item 19개 생성
            t = np.random.randint(1, itemnum + 1) # 1부터 itemnum까지 랜덤한 정수 생성
            while t in rated: t = np.random.randint(1, itemnum + 1) # 생성된 정수가 rated에 포함되어 있으면 다시 생성
            item_idx.append(t) # item_idx에 neg item 추가

        # [u]: user, [seq]: sequence(train item + valid item), [item_idx]: item index(test item + negative items)
        # 테스트 아이템을 neg 샘플들 사이에서 얼마나 잘 식별하는지 평가
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]]) 
        # 각 반복마다 [u], [seq], [item_idx]를 차례대로 가져옴 = l
        # array로 변환 후 unpacking하여 predict 함수에 입력
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item() 
        # 예시: ["-0.9", -0.3, -0.7, -0.2] -> ["0", 2, 1, 3] -> ["0", 2, 1, 3]
        # test item은 item_idx에서 첫 번째에 위치함 
        # 모델을 통해 예측하였을 때, test item의 점수가 가장 낮아야 함. (-가 붙어서)
        # 가장 낮다면 rank가 0이 될 것이다. -> 이후 hit, ndcg가 증가함. (잘 맞췄기 때문에)
        # 결론: test item의 rank를 구한다. (test item은 neg item들 사이에서 가장 낮은 점수를 가져야 함)

        valid_user += 1 

        if rank < 1: # 예측된 아이템이 1위인 경우에만 점수 추가 
            NDCG += 1 / np.log2(rank + 2) 
            HT += 1 
        if valid_user % 100 == 0: 
            print('.', end="")
            sys.stdout.flush() # 100명의 user에 대해 평가할 때마다 . 출력

    return NDCG / valid_user, HT / valid_user 


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
        
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        
        for _ in range(100): # 100개 neg 샘플 생성
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item() 
        valid_user += 1

        if rank < 10: # 상위 10위 순위에 대해 NDCG, HT 계산
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()
    return NDCG / valid_user, HT / valid_user