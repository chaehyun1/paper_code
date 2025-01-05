import random
import pickle

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import numpy as np

from models.recsys_model import *
from models.llm4rec import *
from sentence_transformers import SentenceTransformer


class two_layer_mlp(nn.Module): 
    def __init__(self, dims):
        super().__init__()
        self.fc1 = nn.Linear(dims, 128) # 인코더 
        # input dim: batch_size x dims
        # W: 128 x dims
        # bias: 128
        self.fc2 = nn.Linear(128, dims) # 디코더 
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x1 = self.fc2(x)
        return x, x1 # x: matching, x1: projection

class A_llmrec_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        rec_pre_trained_data = args.rec_pre_trained_data 
        self.args = args
        self.device = args.device
        
        with open(f'./data/amazon/{args.rec_pre_trained_data}_text_name_dict.json.gz','rb') as ft: 
            self.text_name_dict = pickle.load(ft) 
        
        self.recsys = RecSys(args.recsys, rec_pre_trained_data, self.device) # pre-trained recommendation system
        self.item_num = self.recsys.item_num # number of items
        self.rec_sys_dim = self.recsys.hidden_units  # hidden units
        self.sbert_dim = 768
        
        # 참고: 각 인코더가 똑같은 mlp를 쓰는 것 아님, 각각의 mlp를 쓰는 것이다. 객체 다르게 생성함! (코드상 같은 코드를 써서 헷갈릴 수 있음)
        self.mlp = two_layer_mlp(self.rec_sys_dim) # MLP for cf recsys
        if args.pretrain_stage1:
            self.sbert = SentenceTransformer('nq-distilbert-base-v1') # pre-trained SBERT
            self.mlp2 = two_layer_mlp(self.sbert_dim) # MLP for text
        
        self.mse = nn.MSELoss()
        
        self.maxlen = args.maxlen
        self.NDCG = 0
        self.HIT = 0
        self.rec_NDCG = 0
        self.rec_HIT = 0
        self.lan_NDCG=0
        self.lan_HIT=0
        self.num_user = 0
        self.yes = 0
        
        self.bce_criterion = torch.nn.BCEWithLogitsLoss() 
        
        if args.pretrain_stage2 or args.inference:
            self.llm = llm4rec(device=self.device, llm_model=args.llm)
            
            # project collaborative knowledge onto the token space of LLM
            # 1. project user representation
            self.log_emb_proj = nn.Sequential(
                nn.Linear(self.rec_sys_dim, self.llm.llm_model.config.hidden_size), 
                nn.LayerNorm(self.llm.llm_model.config.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.llm.llm_model.config.hidden_size, self.llm.llm_model.config.hidden_size)
            ) # log_emb_proj(=F_U): project user reprentations 
            nn.init.xavier_normal_(self.log_emb_proj[0].weight) 
            nn.init.xavier_normal_(self.log_emb_proj[3].weight) # 첫 번째 linear layer, 두 번째 linear layer 가중치 초기화

            # 2. project joint collaborative text embedding
            self.item_emb_proj = nn.Sequential(
                nn.Linear(128, self.llm.llm_model.config.hidden_size),
                nn.LayerNorm(self.llm.llm_model.config.hidden_size),
                nn.GELU(),
                nn.Linear(self.llm.llm_model.config.hidden_size, self.llm.llm_model.config.hidden_size)
            ) # item_emb_proj(=F_I): project joint collaborative text embedding e_i
            nn.init.xavier_normal_(self.item_emb_proj[0].weight)
            nn.init.xavier_normal_(self.item_emb_proj[3].weight)
            
    def save_model(self, args, epoch1=None, epoch2=None):
        out_dir = f'./models/saved_models/'
        create_dir(out_dir) # create directory if not exist
        out_dir += f'{args.rec_pre_trained_data}_{args.recsys}_{epoch1}_'
        if args.pretrain_stage1:
            torch.save(self.sbert.state_dict(), out_dir + 'sbert.pt')
            torch.save(self.mlp.state_dict(), out_dir + 'mlp.pt') # cf-recsys 
            torch.save(self.mlp2.state_dict(), out_dir + 'mlp2.pt') # text
        
        out_dir += f'{args.llm}_{epoch2}_'
        if args.pretrain_stage2:
            torch.save(self.log_emb_proj.state_dict(), out_dir + 'log_proj.pt')
            torch.save(self.item_emb_proj.state_dict(), out_dir + 'item_proj.pt')
            
    def load_model(self, args, phase1_epoch=None, phase2_epoch=None):
        out_dir = f'./models/saved_models/{args.rec_pre_trained_data}_{args.recsys}_{phase1_epoch}_' # stage-1에서 학습한 모델 불러오기
        
        mlp = torch.load(out_dir + 'mlp.pt', map_location = args.device) 
        self.mlp.load_state_dict(mlp) # mlp에 불러온 'mlp' 대입
        del mlp
        for name, param in self.mlp.named_parameters(): # mlp의 파라미터에 대해 반복
            param.requires_grad = False # 학습하지 않음

        if args.inference:
            out_dir += f'{args.llm}_{phase2_epoch}_'
            
            log_emb_proj_dict = torch.load(out_dir + 'log_proj.pt', map_location = args.device)
            self.log_emb_proj.load_state_dict(log_emb_proj_dict)
            del log_emb_proj_dict
            
            item_emb_proj_dict = torch.load(out_dir + 'item_proj.pt', map_location = args.device)
            self.item_emb_proj.load_state_dict(item_emb_proj_dict)
            del item_emb_proj_dict

    def find_item_text(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'
        if title_flag and description_flag: # title과 description 모두 사용
            return [f'"{self.text_name_dict[t].get(i,t_)}, {self.text_name_dict[d].get(i,d_)}"' for i in item] # item에 대한 title과 description 반환
        elif title_flag and not description_flag:
            return [f'"{self.text_name_dict[t].get(i,t_)}"' for i in item]
        elif not title_flag and description_flag:
            return [f'"{self.text_name_dict[d].get(i,d_)}"' for i in item]
    
    def find_item_text_single(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'
        if title_flag and description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}, {self.text_name_dict[d].get(item,d_)}"'
        elif title_flag and not description_flag:
            return f'"{self.text_name_dict[t].get(item,t_)}"'
        elif not title_flag and description_flag:
            return f'"{self.text_name_dict[d].get(item,d_)}"'
        
    def get_item_emb(self, item_ids):
        with torch.no_grad(): # gradient 계산 안하고 forward만 진행: stage-2에서는 enc 동결
            item_embs = self.recsys.model.item_emb(torch.LongTensor(item_ids).to(self.device)) # item_ids에 대한 임베딩을 얻음
            
            # 중요: self.mlp는 enc_I에 대한 것으로 stage-2에서 joint collaborative-text embedding e_i를 얻기 위해 사용
            # 이미 학습이 완료된 것이다. 
            item_embs, _ = self.mlp(item_embs) # item_embs를 1-layer MLP에 통과시킴 = e_i = joint collaborative text embedding
         
        return item_embs
    
    def forward(self, data, optimizer=None, batch_iter=None, mode='phase1'):
        if mode == 'phase1':
            self.pre_train_phase1(data, optimizer, batch_iter)
        if mode == 'phase2':
            self.pre_train_phase2(data, optimizer, batch_iter)
        if mode =='generate':
            self.generate(data)

    def pre_train_phase1(self, data, optimizer, batch_iter): # stage1 학습시키는 부분 
        epoch, total_epoch, step, total_step = batch_iter
        # epoch: 현재 epoch, total_epoch: 총 epoch, step: 현재 step, total_step: 총 step
        
        self.sbert.train() # SBERT fine-tuning 가능
        optimizer.zero_grad() # optimizer initialization

        u, seq, pos, neg = data 
        indices = [self.maxlen*(i+1)-1 for i in range(u.shape[0])] # 각 배치의 유저 수 만큼 반복하여 마지막 인덱스를 가져옴, ex) [49, 99, 149, ...]
        
        with torch.no_grad(): # gradient 계산 안하고 forward만 진행
            log_emb, pos_emb, neg_emb = self.recsys.model(u, seq, pos, neg, mode='item')
            # sasrec model의 forward에서 배치 유저의 seq, pos, neg에 대한 아이템 임베딩을 반환
            # seq를 넣어 얻은 log_emb = user representation = x
            # pre-train된 sasrec model을 이용
            
        log_emb_ = log_emb[indices] # 배치 유저의 마지막 상호작용 아이템의 임베딩
        pos_emb_ = pos_emb[indices] # 배치 유저의 마지막 positive 아이템(마지막 상호작용 아이템의 next item) 임베딩
        neg_emb_ = neg_emb[indices] # 배치 유저의 마지막 negative 아이템(상호작용X) 임베딩
        pos_ = pos.reshape(pos.size)[indices] # 배치 유저의 마지막 positive 아이템 인덱스
        neg_ = neg.reshape(neg.size)[indices] # 배치 유저의 마지막 negative 아이템 인덱스
        # 학습 시 시퀀스 내의 마지막 아이템만 고려 
        
        start_inx = 0 
        end_inx = 60 # 한 번에 60개의 유저를 학습하기 위해
        iterss = 0 
        mean_loss = 0
        bpr_loss = 0
        gt_loss = 0 
        rc_loss = 0 
        text_rc_loss = 0
        original_loss = 0
        while start_inx < len(log_emb_): # log_emb_의 길이(user 수)만큼 반복
            # 아이템 임베딩을 위한 과정
            log_emb = log_emb_[start_inx:end_inx] # 60명 유저의 마지막 아이템의 log_emb = user representation = x
            pos_emb = pos_emb_[start_inx:end_inx] # _emb = E_i
            neg_emb = neg_emb_[start_inx:end_inx] # _emb = E_i
            
            # 텍스트 임베딩을 위한 과정 
            pos__ = pos_[start_inx:end_inx] # pos_에서 start_inx부터 end_inx까지의 아이템 인덱스
            neg__ = neg_[start_inx:end_inx]
            
            start_inx = end_inx # start_inx를 end_inx로 변경
            end_inx += 60 
            iterss +=1
            
            # text embedding을 얻는 과정 
            pos_text = self.find_item_text(pos__) # item에 대한 title과 description, 각 아이템을 요소로 한 리스트 반환 
            neg_text = self.find_item_text(neg__)
            
            pos_token = self.sbert.tokenize(pos_text) # SBERT로 토큰화
            pos_text_embedding= self.sbert({'input_ids':pos_token['input_ids'].to(self.device),
                                            'attention_mask':pos_token['attention_mask'].to(self.device)})['sentence_embedding']
            # SBERT로 text embedding을 얻음
            neg_token = self.sbert.tokenize(neg_text)
            neg_text_embedding= self.sbert({'input_ids':neg_token['input_ids'].to(self.device),
                                            'attention_mask':neg_token['attention_mask'].to(self.device)})['sentence_embedding']
            # _text_embedding = Q_i
            
            pos_text_matching, pos_proj = self.mlp(pos_emb) # positive 아이템 임베딩을 MLP에 통과시킴
            neg_text_matching, neg_proj = self.mlp(neg_emb) # negative 아이템 임베딩을 MLP에 통과시킴
            # matching: 햐나의 MLP layer를 통과시킨 것 (인코더 통과) e_i
            # proj: 두 번째 MLP layer를 통과시킨 것 (디코더 통과)
            
            pos_text_matching_text, pos_text_proj = self.mlp2(pos_text_embedding) # positive 아이템 텍스트 임베딩을 MLP에 통과시킴
            neg_text_matching_text, neg_text_proj = self.mlp2(neg_text_embedding) # negative 아이템 텍스트 임베딩을 MLP에 통과시킴
            # matching: 햐나의 MLP layer를 통과시킨 것 (인코더 통과) q_i
            # proj: 두 번째 MLP layer를 통과시킨 것 (디코더 통과)
            
            # 이전 선호도를 바탕으로 다음 아이템의 선호도 점수를 내적 계산: 이러한 이유로 인해 seq는 처음부터 끝-1까지, pos는 1부터 끝까지 데이터가 만들어짐. 
            # neg는 이전 선호도와 상호작용이 없는 것으로 구성
            # L_rec 계산 논문 식 (5) 
            pos_logits, neg_logits = (log_emb*pos_proj).mean(axis=1), (log_emb*neg_proj).mean(axis=1) # 내적 계산: x * dec(~)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=pos_logits.device), torch.zeros(neg_logits.shape, device=pos_logits.device) # positive, negative 아이템에 대한 레이블 생성
            loss = self.bce_criterion(pos_logits, pos_labels) # 모델은 positive 아이템을 높게, negative 아이템을 낮게 예측하도록 학습됨
            loss += self.bce_criterion(neg_logits, neg_labels)
            
            # 세 가지 loss 계산 
            matching_loss = self.mse(pos_text_matching,pos_text_matching_text) + self.mse(neg_text_matching,neg_text_matching_text) # 논문 식 (2)
            reconstruction_loss = self.mse(pos_proj,pos_emb) + self.mse(neg_proj,neg_emb) # 논문 식 (3)
            text_reconstruction_loss = self.mse(pos_text_proj,pos_text_embedding.data) + self.mse(neg_text_proj,neg_text_embedding.data) # 논문 식 (4)
            
            total_loss = loss + matching_loss + 0.5*reconstruction_loss + 0.2*text_reconstruction_loss # 총 loss 계산
            
            total_loss.backward() # gradient 계산
            optimizer.step() # 가중치 업데이트
            
            # loss 반영 
            mean_loss += total_loss.item()  
            bpr_loss += loss.item() 
            gt_loss += matching_loss.item()
            rc_loss += reconstruction_loss.item()
            text_rc_loss += text_reconstruction_loss.item()
        
        # 한 에포크 내 배치별 평균 loss 출력    
        print("loss in epoch {}/{} iteration {}/{}: {} / BPR loss: {} / Matching loss: {} / Item reconstruction: {} / Text reconstruction: {}".format(epoch, total_epoch, step, total_step, mean_loss/iterss, bpr_loss/iterss, gt_loss/iterss, rc_loss/iterss, text_rc_loss/iterss))
    
    def make_interact_text(self, interact_ids, interact_max_num):
        interact_item_titles_ = self.find_item_text(interact_ids, title_flag=True, description_flag=False) # item에 대한 title을 모은 리스트
        interact_text = [] # 상호작용 아이템의 title을 저장할 리스트
        if interact_max_num == 'all': 
            for title in interact_item_titles_:
                interact_text.append(title + '[HistoryEmb]') # '[HistoryEmb]'를 붙여서 아이템 title 저장
        else: 
            for title in interact_item_titles_[-interact_max_num:]: # 상호작용 아이템의 title을 최대 interact_max_num개까지만 저장
                interact_text.append(title + '[HistoryEmb]') 
            interact_ids = interact_ids[-interact_max_num:] # 상호작용 아이템의 id를 최대 interact_max_num개까지만 저장
            
        interact_text = ','.join(interact_text) # 상호작용 아이템의 title을 쉼표로 구분하여 하나의 문자열로 반환
        return interact_text, interact_ids
    
    def make_candidate_text(self, interact_ids, candidate_num, target_item_id, target_item_title):
        # neg_item_id: 상호작용하지 않은 아이템의 id를 저장할 리스트
        neg_item_id = []
        while len(neg_item_id)<50:
            t = np.random.randint(1, self.item_num+1) 
            if not (t in interact_ids or t in neg_item_id): # 상호작용한 아이템이나 이미 선택된 아이템이 아닌 경우
                neg_item_id.append(t) # neg_item_id에 추가
        random.shuffle(neg_item_id) # neg_item_id를 섞음
         
        candidate_ids = [target_item_id] # 후보 아이템의 id를 저장할 리스트
        candidate_text = [target_item_title + '[CandidateEmb]'] # 후보 아이템의 title을 저장할 리스트

        for neg_candidate in neg_item_id[:candidate_num - 1]: # 후보 아이템의 개수만큼 반복
            candidate_text.append(self.find_item_text_single(neg_candidate, title_flag=True, description_flag=False) + '[CandidateEmb]') 
            candidate_ids.append(neg_candidate) # 후보 아이템의 title을 저장하고 후보 아이템의 id를 저장
        
        # 후보군 랜덤하게 섞기        
        random_ = np.random.permutation(len(candidate_text)) # len(candidate_text)만큼 숫자를 섞음
        candidate_text = np.array(candidate_text)[random_] # candidate_text를 섞음
        candidate_ids = np.array(candidate_ids)[random_] # candidate_ids를 섞음
            
        return ','.join(candidate_text), candidate_ids 
    
    def pre_train_phase2(self, data, optimizer, batch_iter):
        epoch, total_epoch, step, total_step = batch_iter
        
        optimizer.zero_grad() # optimizer 초기화
        u, seq, pos, neg = data # pos: seq의 각 아이템의 다음 아이템
        mean_loss = 0
        
        text_input = [] # 각 유저별 prompt를 모아둔 리스트
        text_output = [] # 각 유저별 정답 아이템의 title을 모아둔 리스트
        interact_embs = []
        candidate_embs = []
        self.llm.eval() # LLM은 학습 안함
        
        with torch.no_grad(): # gradient 계산 안하고 forward만 진행
            log_emb = self.recsys.model(u,seq,pos,neg, mode = 'log_only') # sasrec의 forward 통해 각 유저의 마지막 아이템에 대한 user representation 얻기
            
        for i in range(len(u)): # batch size만큼 반복
            target_item_id = pos[i][-1] # 상호작용한 마지막 아이템 
            target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False) # 마지막 아이템의 title
            
            # 상호작용한 아이템의 title과 그 id를 얻음
            interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10) # seq[i][seq[i] > 0]: 해당 유저 시퀀스에서 0이 아닌 값만 가져옴
            
            # 후보 아이템은 실제로 추천할 정답 아이템 + 아직 상호작용하지 않은 아이템들
            # 1개의 Positive 아이템(정답)과 20개의 Negative 아이템을 사용
            candidate_num = 20 # 후보 아이템 개수
            candidate_text, candidate_ids = self.make_candidate_text(seq[i][seq[i]>0], candidate_num, target_item_id, target_item_title) 
            
            # 논문 figure 3
            input_text = '' # input_text는 문장 형식에 쓰이는 string
            input_text += ' is a user representation.' # user representation을 먼저 넣음 (soft prompt와 유사)
                
            if self.args.rec_pre_trained_data == 'Movies_and_TV':
                input_text += 'This user has watched '
            elif self.args.rec_pre_trained_data == 'Video_Games':
                input_text += 'This user has played '
            elif self.args.rec_pre_trained_data == 'Luxury_Beauty' or self.args.rec_pre_trained_data == 'Toys_and_Games':
                input_text += 'This user has bought '
                
            input_text += interact_text # 이전까지 상호작용한 아이템들의 title을 넣음
            
            if self.args.rec_pre_trained_data == 'Movies_and_TV':
                input_text +=' in the previous. Recommend one next movie for this user to watch next from the following movie title set, '
            elif self.args.rec_pre_trained_data == 'Video_Games':
                input_text +=' in the previous. Recommend one next game for this user to play next from the following game title set, '            
            elif self.args.rec_pre_trained_data == 'Luxury_Beauty' or self.args.rec_pre_trained_data == 'Toys_and_Games':
                input_text +=' in the previous. Recommend one next item for this user to buy next from the following item title set, '
                    
            input_text += candidate_text # 후보 아이템들의 title을 넣음
            input_text += '. The recommendation is '

            text_input.append(input_text) # input_text를 text_input에 추가
            text_output.append(target_item_title) # target_item_title(정답 아이템 제목)을 text_output에 추가 

            # O_i = F_I(e_i) 계산 (toekn space로 project)
            interact_embs.append(self.item_emb_proj(self.get_item_emb(interact_ids))) 
            candidate_embs.append(self.item_emb_proj(self.get_item_emb(candidate_ids))) 
        # 정리: 모든 유저(batch_size)에 대해서 prompt를 만드는데, 해당 유저가 이전까지 상호작용한 아이템들의 title과 추천할 후보 아이템들의 title을 넣음.
        
        # prompt에 중간에 임베딩을 넣어야 한다.     
        samples = {'text_input': text_input, 'text_output': text_output, 'interact': interact_embs, 'candidate':candidate_embs}
        log_emb = self.log_emb_proj(log_emb) # O_u = F_U(x^u) (projected embeddings of user representation)
        loss_rm = self.llm(log_emb, samples) # LLM fowrard 진행 -> next item title을 예측하는 loss 계산, 한 배치에 대해서 한 것임
        loss_rm.backward()  # gradient 계산
        optimizer.step() # 가중치 업데이트
        mean_loss += loss_rm.item() # loss 계산
        print("A-LLMRec model loss in epoch {}/{} iteration {}/{}: {}".format(epoch, total_epoch, step, total_step, mean_loss))
        
    def generate(self, data):
        u, seq, pos, neg, rank = data 
        # 참고: neg는 학습 시 사용되는 것. 
        
        answer = []
        text_input = []
        interact_embs = []
        candidate_embs = []
        with torch.no_grad(): # 학습 안함. 
            log_emb = self.recsys.model(u,seq,pos,neg, mode = 'log_only')
            for i in range(len(u)):
                target_item_id = pos[i]
                target_item_title = self.find_item_text_single(target_item_id, title_flag=True, description_flag=False) # target item의 title
                
                interact_text, interact_ids = self.make_interact_text(seq[i][seq[i]>0], 10) # 상호작용 아이템의 title과 id를 얻음
                
                # 추천 task 후보군 생성
                candidate_num = 20
                candidate_text, candidate_ids = self.make_candidate_text(seq[i][seq[i]>0], candidate_num, target_item_id, target_item_title)
                
                input_text = ''
                input_text += ' is a user representation.'
                if self.args.rec_pre_trained_data == 'Movies_and_TV':
                    input_text += 'This user has watched '
                elif self.args.rec_pre_trained_data == 'Video_Games':
                    input_text += 'This user has played '
                elif self.args.rec_pre_trained_data == 'Luxury_Beauty' or self.args.rec_pre_trained_data == 'Toys_and_Games':
                    input_text += 'This user has bought '
                    
                input_text += interact_text
                
                if self.args.rec_pre_trained_data == 'Movies_and_TV':
                    input_text +=' in the previous. Recommend one next movie for this user to watch next from the following movie title set, '
                elif self.args.rec_pre_trained_data == 'Video_Games':
                    input_text +=' in the previous. Recommend one next game for this user to play next from the following game title set, '            
                elif self.args.rec_pre_trained_data == 'Luxury_Beauty' or self.args.rec_pre_trained_data == 'Toys_and_Games':
                    input_text +=' in the previous. Recommend one next item for this user to buy next from the following item title set, '
                
                input_text += candidate_text
                input_text += '. The recommendation is '
                
                answer.append(target_item_title)
                text_input.append(input_text)
                
                interact_embs.append(self.item_emb_proj(self.get_item_emb(interact_ids))) # O_i = F_I(e_i)
                candidate_embs.append(self.item_emb_proj(self.get_item_emb(candidate_ids)))
        
        log_emb = self.log_emb_proj(log_emb) # O_u = F_U(x^u)
        atts_llm = torch.ones(log_emb.size()[:-1], dtype=torch.long).to(self.device) 
        atts_llm = atts_llm.unsqueeze(1)
        log_emb = log_emb.unsqueeze(1)
        
        with torch.no_grad(): # gradient 계산 안하고 forward만 진행
            self.llm.llm_tokenizer.padding_side = "left" # padding을 left로 설정
            llm_tokens = self.llm.llm_tokenizer( # LLM tokenizer로 prompt를 토큰화
                text_input,
                padding="longest",
                return_tensors="pt"
            ).to(self.device)
            
            with torch.cuda.amp.autocast():
                inputs_embeds = self.llm.llm_model.get_input_embeddings()(llm_tokens.input_ids) # token embedding
                
                llm_tokens, inputs_embeds = self.llm.replace_hist_candi_token(llm_tokens, inputs_embeds, interact_embs, candidate_embs) # 상호작용 아이템과 후보 아이템의 임베딩을 prompt에 추가
                    
                attention_mask = llm_tokens.attention_mask
                inputs_embeds = torch.cat([log_emb, inputs_embeds], dim=1) # user representation과 나머지 합치기 
                attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)
                    
                outputs = self.llm.llm_model.generate( # LLM으로 text 생성
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=False,
                    top_p=0.9,
                    temperature=1,
                    num_beams=1,
                    max_length=512,
                    min_length=1,
                    pad_token_id=self.llm.llm_tokenizer.eos_token_id,
                    repetition_penalty=1.5,
                    length_penalty=1,
                    num_return_sequences=1,
                )

            outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
            output_text = self.llm.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True) # token을 text로 디코딩
            output_text = [text.strip() for text in output_text] # text의 앞뒤 공백 제거

        for i in range(len(text_input)):
            f = open(f'./recommendation_output.txt','a')
            f.write(text_input[i])
            f.write('\n\n')
            
            f.write('Answer: '+ answer[i])
            f.write('\n\n')
            
            f.write('LLM: '+str(output_text[i]))
            f.write('\n\n')
            f.close()

        return output_text # 평가 시 학습 때처럼 hitory에 대한 item embedding과 후보 아이템 embedding을 넣을 필요가 없다. 왜냐하면 모델이 이미 학습을 통해 collaborative knowledge를 학습했기 때문이다.