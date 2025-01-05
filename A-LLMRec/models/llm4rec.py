import torch
import torch.nn as nn

from transformers import AutoTokenizer, OPTForCausalLM

class llm4rec(nn.Module):
    def __init__(
        self,
        device,
        llm_model="",
        max_output_txt_len=256,
    ):
        super().__init__()
        self.device = device
        
        # flan_t5, opt, vicuna
        if llm_model == 'opt': 
            # pre-trained llm model
            self.llm_model = OPTForCausalLM.from_pretrained("facebook/opt-6.7b", torch_dtype=torch.float16, load_in_8bit=True, device_map=self.device)
            # 해당 모델에 맞는 토크나이저를 사용하여 입력 데이터를 토큰화
            self.llm_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b", use_fast=False)
            # self.llm_model = OPTForCausalLM.from_pretrained("facebook/opt-6.7b", torch_dtype=torch.float16, device_map=self.device)
        else:
            raise Exception(f'{llm_model} is not supported')
            
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # 패딩 토큰 추가
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'}) # 문장의 시작을 나타내는 토큰 추가
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'}) # 문장의 끝을 나타내는 토큰 추가
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'}) # 모르는 단어를 나타내는 토큰 추가
        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': ['[UserRep]','[HistoryEmb]','[CandidateEmb]']}) # 사용자 임베딩, 상호작용 임베딩, 후보 아이템 임베딩을 나타내는 토큰 추가
        # LLM 프롬프트에서 embedding을 값을 대체하기 위한 특수 토큰 추가

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer)) # 특수 토큰을 추가한 후, 토크나이저의 토큰 수가 변경 -> 모델의 토큰 임베딩 크기를 토크나이저에 맞게 조정
        
        for _, param in self.llm_model.named_parameters(): 
            param.requires_grad = False # 모든 파라미터를 학습하지 않음
            
        self.max_output_txt_len = max_output_txt_len # 최대 출력 텍스트 길이

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        # 입력 텍스트(input)와 출력 텍스트(output)를 결합하여 하나의 시퀀스로 만든다.
        # input_ids: 각 유저의 입력 프롬프트를 토큰 ID로 변환한 값
        # output_ids: 각 유저의 정답 아이템 제목을 토큰 ID로 변환한 값
        input_part_targets_len = [] # 각 유저별로 입력 프롬프트의 유효한 토큰 수를 저장할 리스트
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)): # batch size만큼 반복
            this_input_ones = input_atts[i].sum() # 각 유저별 입력 프롬프트의 유효한 토큰 수
            input_part_targets_len.append(this_input_ones) # 입력 프롬프트의 유효한 토큰 수를 저장
            
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones], # 입력의 유효 토큰 부분 
                    output_ids[i][1:], # BOS 토큰을 제외한 출력의 모든 부분 
                    input_ids[i][this_input_ones:] # 입력의 패딩된 부분
                ]) # 입력 텍스트와 출력 텍스트를 토큰 id를 바탕으로 결합
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids']) # 리스트를 텐서로 변환
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask']) 
        return llm_tokens, input_part_targets_len # 결합된 정보와 입력 프롬프트의 유효한 토큰 수 리스트 반환

    def replace_hist_candi_token(self, llm_tokens, inputs_embeds, interact_embs, candidate_embs):
        if len(interact_embs) == 0: # 상호작용 임베딩이 없는 경우
            return llm_tokens, inputs_embeds 
        
        # "[HistoryEmb]"라는 특별한 토큰의 ID 찾기. 이 토큰을 찾아서 상호작용 아이템 임베딩으로 대체할 것이기 때문
        history_token_id = self.llm_tokenizer("[HistoryEmb]", return_tensors="pt", add_special_tokens=False).input_ids.item() 
        
        # "[CandidateEmb]"라는 특별한 토큰의 ID 찾기. 이 토큰을 찾아서 후보 아이템 임베딩으로 대체할 것이기 때문
        candidate_token_id = self.llm_tokenizer("[CandidateEmb]", return_tensors="pt", add_special_tokens=False).input_ids.item()
        
        for inx in range(len(llm_tokens["input_ids"])): # batch size만큼 반복
            idx_tensor=(llm_tokens["input_ids"][inx]==history_token_id).nonzero().view(-1) # 입력 토큰 ID 중 "[HistoryEmb]" 토큰의 위치 찾기
            # nonzero() 함수는 0이 아닌 값의 위치를 찾아주는 함수
            # view(-1)로 1차원으로 변환
            for idx, item_emb in zip(idx_tensor, interact_embs[inx]): # 찾은 위치에 상호작용 임베딩을 대체
                inputs_embeds[inx][idx]=item_emb
        
            idx_tensor=(llm_tokens["input_ids"][inx]==candidate_token_id).nonzero().view(-1) # 입력 토큰 ID 중 "[CandidateEmb]" 토큰의 위치 찾기
            for idx, item_emb in zip(idx_tensor, candidate_embs[inx]): # 찾은 위치에 후보 아이템 임베딩을 대체
                inputs_embeds[inx][idx]=item_emb
        return llm_tokens, inputs_embeds # 대체된 정보(inputs_embeds)를 반환
    
    def forward(self, log_emb, samples):
        # log_emb = O_u
        atts_llm = torch.ones(log_emb.size()[:-1], dtype=torch.long).to(self.device) # 마지막 차원(embedding_dim)을 제외한 차원들로 텐서를 생성, (batch_size, seq_len)
        atts_llm = atts_llm.unsqueeze(1) # 차원 추가: (batch_size, 1, seq_len)
        
        # 토큰화: 토큰(텍스트) -> 토큰 ID로 변환    
        text_output_tokens = self.llm_tokenizer( # 각 정답 아이템 제목을 토큰화
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']], # 각 정답 아이템 제목 끝에 문장의 끝을 나타내는 토큰 추가
            return_tensors="pt", 
            padding="longest", # 가장 긴 시퀀스 길이에 맞춰 패딩
            truncation=False,
        ).to(self.device)
        
        text_input_tokens = self.llm_tokenizer( # 각 입력 프롬프트를 토큰화
            samples['text_input'], 
            return_tensors="pt",
            padding="longest",
            truncation=False,
        ).to(self.device)
        
        llm_tokens, input_part_targets_len = self.concat_text_input_output( # 입력 프롬프트 토큰 ID와 정답 아이템 제목 토큰 ID를 결합
            text_input_tokens.input_ids, # 텍스트를 모델이 이해할 수 있는 정수(ID) 시퀀스로 변환한 값
            text_input_tokens.attention_mask, # 각 위치가 유효한 토큰인지 아닌지를 나타내는 마스크, 1은 유효한 토큰(모델이 처리해야 할 부분), 0은 패딩된 부분(무시).
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )

        targets = llm_tokens['input_ids'].masked_fill(llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100) # llm_tokens['input_ids'] 텐서에서 패딩 토큰의 ID 값을 -100으로 대체
        # 패딩 토큰을 -100으로 설정하여, 패딩된 부분은 loss 계산에서 제외하도록 지정 
        # llm_tokens['input_ids'] = 입력 프롬프트와 정답 아이템 제목을 결합한 토큰 ID

        for i, l in enumerate(input_part_targets_len): # 각 사용자의 입력 프롬프트의 유효한 토큰 수 = l, 사용자수 만큼 반복
            targets[i][:l] = -100 # 입력 프롬프트의 유효한 토큰 부분을 -100으로 대체, loss 계산에서 제외
        # 즉, targets에서 출력 텍스트 부분만 남기고, 입력 텍스트와 패딩 부분은 -100으로 대체하여 loss 계산에서 제외
        # 입력 프롬프트의 경우에는 그냥 프롬프트 형식의 문자열이기 때문에 추가적으로 학습이 필요하지 않음. 
        # output에 해당하는 정답 아이템 제목 부분은 남겨두기 
        
        empty_targets = (torch.ones(atts_llm.size(), dtype=torch.long).to(self.device).fill_(-100)) # atts_llm과 같은 크기의 텐서를 생성하고 -100으로 채움
        # user representation은 학습할 필요가 없기 때문에 -100으로 채움

        targets = torch.cat([empty_targets, targets], dim=1) # 빈 타겟과 실제 타겟을 결합 -> 학습해야 하는 부분인 정답 아이템 제목 부분만 그대로 남김 
        
        # llm_tokens['input_ids']: (batch_size, sequence_length) sequence_length는 토큰화한 것의 최대 길이 
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids']) # 각 유저에 대해서 결합한 토큰 ID를 임베딩으로 변환, (batch_size, sequence_length, embedding_dim)
        
        # 중요: 이전의 상호작용 아이템 임베딩과 후보 아이템 임베딩 값으로 대체하기 
        # 아직 hitory의 item embedding과 candidate의 item embedding을 제대로 넣지 않음. 
        # 결합한 토큰 ID를 임베딩으로 변환한 상태에서 상호작용 아이템 임베딩과 후보 아이템 임베딩으로 대체해야 함. 
        llm_tokens, inputs_embeds = self.replace_hist_candi_token(llm_tokens, inputs_embeds, samples['interact'], samples['candidate']) 
        attention_mask = llm_tokens['attention_mask'] # 유효한 토큰인지 아닌지를 나타내는 마스크
        
        log_emb = log_emb.unsqueeze(1) # 차원 추가: (batch_size, 1, embedding_dim)
        inputs_embeds = torch.cat([log_emb, inputs_embeds], dim=1) # user representation과 나머지(프롬프트+결과) 결합
        attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1) # attention_mask에서 -100으로 채워진 부분은 학습에서 제외
        
        # 모델은 정답 아이템 제목을 예측하도록 학습된다. 
        with torch.cuda.amp.autocast(): # 자동 혼합 정밀도(AMP)를 사용하여 모델을 가속화
            outputs = self.llm_model( 
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True, # 출력을 딕셔너리 형태로 반환
                labels=targets, # 정답 아이템 제목 ID
            )
        loss = outputs.loss # 손실 값

        return loss # 정리: 정답 1개와 나머지 neg item 20개 중 정답 1개(다음에 추천될 아이템 제목)을 잘 예측하도록 학습됨 