import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import pickle
import os

class ST_SGG(nn.Module):
    def __init__(self, 
                 cfg
                 ):
        super(ST_SGG, self).__init__()
        self.cfg = cfg
        self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES # Number of object classes, 151
        self.num_rel_cls = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES # Number of predicate classes, 51
        self.alpha = cfg.MODEL.ROI_RELATION_HEAD.STSGG_MODULE.ALPHA_INC # 0.8
        self.alpha_decay = cfg.MODEL.ROI_RELATION_HEAD.STSGG_MODULE.ALPHA_DEC # 0.4
        self.save_period = cfg.MODEL.ROI_RELATION_HEAD.STSGG_MODULE.SAVE_CUMULATIVE_PSEUDO_LABEL_INFO_PERIOD
        self.use_gsl_output = cfg.MODEL.ROI_RELATION_HEAD.STSGG_MODULE.USE_GSL_OUTPUT
        
        # statistics: information of N predicate class
        statistics = torch.load("initial_data/obj_pred_info/obj_pred_info_1800") if self.num_rel_cls > 200 else torch.load("initial_data/obj_pred_info/obj_pred_info_50")
        fg_matrix = statistics['fg_matrix'] # 객체 간 관계 빈도를 저장한 행렬.
        # (num_obj_classes, num_obj_classes, num_rel_classes)
        # fg_matrix[i, j, k] = i번째 객체와 j번째 객체 사이의 k번째 관계가 등장한 횟수.
        
        pred_count = fg_matrix.sum(0).sum(0)[1:] # N_c: 각 관계(Predicate) 클래스가 등장한 총 횟수
        # [1:]: 0번 관계는 일반적으로 Background Relation 또는 No Relation을 의미.
        # 유의미한 관계는 1번부터 시작.
        
        pred_count = torch.hstack([torch.LongTensor([0]), pred_count]) # 배경 관계(0번 ID)를 포함.
        self.pred_count =pred_count.cuda()
        
        num_max = pred_count.max() # N_1: 데이터셋에서 가장 자주 등장한 관계(Predicate)의 등장 횟수.
        
        # Calculate the lambda^{inc}
        temp = (1/(np.exp((np.log(1.0)/self.alpha))))*num_max - num_max # temp=0
        self.lambda_inc = (pred_count / (num_max+temp)) ** (self.alpha) 
        
        # Calculate the lambda^{dec}
        self.lambda_dec = torch.zeros(len(self.lambda_inc)).cuda()
        val, ind = torch.sort(self.lambda_inc) # 오름차순으로 정렬, val: 정렬된 값, ind: 원래 인덱스
        temp = (1/(np.exp((np.log(1.0)/self.alpha_decay))))*num_max - num_max # temp=0
        temp_lambda_inc = (pred_count / (num_max+temp)) ** (self.alpha_decay)
        for i, (_, idx) in enumerate(zip(val, ind)):
            if idx == 0: continue # bg는 제외
            self.lambda_dec[idx] = temp_lambda_inc[ind[len(temp_lambda_inc)-i]]                          
        
        self.pred_threshold = (torch.zeros(self.num_rel_cls) + 1e-6).cuda() 
        
        # For saving the threshold info
        self.n_pseudo_info = {}
        self.n_cumulative_class = torch.zeros(self.num_rel_cls, dtype=torch.int).cuda()
        self.forward_time = 0
        self.pseudo_label_info = {i:[] for i in np.arange(self.num_rel_cls)}
        self.batch_decrease_conf_dict = {i:[] for i in range(self.num_rel_cls)}
        self.batch_increase_conf_dict = {i:[] for i in range(self.num_rel_cls)}
        
    def box_filter(self, boxes):
        overlaps = self.box_overlaps(boxes, boxes) > 0 # 오버랩 면적이 0보다 큰 경우 True, 그렇지 않으면 False.
        overlaps.fill_diagonal_(0) # 같은 박스끼리(box[i] == box[i]) 비교하면 항상 겹치므로 대각선 값은 0으로 설정.
        return torch.stack(torch.where(overlaps)).T # (num_overlapping_pairs, 2)
    
    def box_overlaps(self, box1, box2):
        num_box1 = box1.shape[0]
        num_box2 = box2.shape[0]
        lt = torch.maximum(box1.reshape([num_box1, 1, -1])[:,:,:2], box2.reshape([1, num_box2, -1])[:,:,:2])
        rb = torch.minimum(box1.reshape([num_box1, 1, -1])[:,:,2:], box2.reshape([1, num_box2, -1])[:,:,2:])
        wh = (rb - lt).clip(min=0) # 오버랩 영역의 너비와 높이 계산.
        inter = wh[:,:,0] * wh[:,:,1] # 겹치는 영역의 면적(intersection)을 계산.
        return inter
    
    
    def update_class_threshold(self, rel_pesudo_labels, one_hot_gt_or_pseudo):
        """
        Adaptive Thresholding: EMA
        """
        concat_pseudo_labels = torch.cat(rel_pesudo_labels)
        pseudo_label_set = torch.unique(concat_pseudo_labels[torch.nonzero(one_hot_gt_or_pseudo)]) # pseudo-label로 선택된 클래스만 선택
        
        for p in np.arange(self.num_rel_cls): # 모든 관계 클래스에 대해 반복
            if p == 0: continue # bg class 제외 
            # Descent
            if p not in pseudo_label_set: # pseudo-label로 선택되지 않은 클래스이면
                if len(self.batch_decrease_conf_dict[p]) > 0: # pseudo-label 후보 중 pseudo-label로 선택되지 않은 클래스의 신뢰도가 존재하면
                    decay_pred_conf = np.mean(self.batch_decrease_conf_dict[p]) # 해당 신뢰도 평균 계산
                    self.pred_threshold[p] = self.pred_threshold[p] * (1-self.lambda_dec[p]) + decay_pred_conf * self.lambda_dec[p] # 논문 식 (2): 두번째 줄
                    # self.pred_threshold[p]: tau_c^{t-1}
                    # self.lambda_dec[p]: lambda^{dec}
                    # decay_pred_conf: E[\hat{q}_i}]
            # Ascent
            else: # pseudo-label로 선택된 클래스이면
                if len(self.batch_increase_conf_dict[p]) > 0: # pseudo-label로 선택된 클래스의 신뢰도가 존재하면
                    mean_conf = np.mean(self.batch_increase_conf_dict[p]) # 해당 신뢰도 평균 계산
                    self.pred_threshold[p] = (1-self.lambda_inc[p])* self.pred_threshold[p] + self.lambda_inc[p] * mean_conf # 논문 식 (2): 첫번째 줄
            # set of predicates in unannotated triplets이 아니면, 신뢰도를 업데이트하지 않음.
            
        # Clear the list       
        # 다음 배치를 위한 새로운 임계값을 준비하기 위해 이전 배치에서 쌓인 데이터 초기화 
        for i in range(self.num_rel_cls):
            self.batch_increase_conf_dict[i].clear() # pseudo-label로 선택된 클래스의 신뢰도 저장 리스트 초기화
            self.batch_decrease_conf_dict[i].clear() # pseudo-label로 선택되지 않은 클래스의 신뢰도 저장 리스트 초기화
        
        
    def save_threshold_info(self):
        """
        Save the pseudo-label info: threshold, class
        """
        if self.save_period > 0:
            self.n_pseudo_info[self.forward_time] = {}
            self.n_pseudo_info[self.forward_time]['n_class'] = np.array(self.n_cumulative_class.cpu(), dtype=np.int32)
            self.n_pseudo_info[self.forward_time]['threshold'] = np.array(self.pred_threshold.cpu(), dtype=np.float16)

            if self.forward_time % self.save_period == 0:
                previous_path = f"{self.cfg.OUTPUT_DIR}/pseudo_info_{self.forward_time-self.save_period}.pkl"
                if os.path.isfile(previous_path):
                    os.remove(previous_path)
                    
                with open(f"{self.cfg.OUTPUT_DIR}/pseudo_info_{self.forward_time}.pkl", 'wb') as f:
                    pickle.dump(self.n_pseudo_info, f)
        self.forward_time += 1
    
    
    def forward(self, rel_pair_idxs, inst_proposals, rel_labels, pred_rel_logits, gsl_outputs=None):
        # rel_pair_idxs: 관계 추론 대상이 되는 객체 쌍의 인덱스 정보.
        # inst_proposals: proposals
        # rel_labels: GT Relation Labels
        # pred_rel_logits: 각 관계 클래스에 대한 확률값
        
        rel_pseudo_labels = []
        n_class = torch.zeros(self.num_rel_cls, dtype=torch.float) 
        gt_or_pseudo_list = [] # 1: assign pseudo-label, 0: No assign
        
        for i, (rel_pair, pred_rel_logit) in enumerate(zip(rel_pair_idxs, pred_rel_logits)):
            # 관계 후보 쌍(rel_pair_idxs)과 그에 대한 예측 logit(pred_rel_logits)을 하나씩 가져오는 반복문.
            rel_pair = rel_pair.long() 
            n_annotate_label = torch.nonzero(rel_labels[i]).shape[0] # 실제로 관계 라벨이 있는 것의 개수.
            pred_rel_logit = F.softmax(pred_rel_logit, -1).detach() # 각 row는 model prediction인 \hat{p}^p, 차원: (객체 i와 j의 num_rel_pairs, num_rel_classes)
            
            # Filter the non-overlapped pair
            # 관계 후보 중 실제로 객체 bbox가 겹치는 경우만 선택.
            overlap_idx = self.box_filter(inst_proposals[i].bbox)
            overlap_idx = ((overlap_idx.T[...,None][...,None] == rel_pair[None,...][None,...]).sum(0).sum(-1) == 2).any(0)[n_annotate_label:] # True: 해당 관계 후보(rel_pair)가 실제로 bbox가 겹치는 관계임.

            rel_confidence, pred_rel_class = pred_rel_logit[:,1:].max(1) 
            # rel_confidence: 각 관계 예측에서 신뢰도가 가장 높은 확률값. = \hat{q} (각 row마다 \hat{q} 존재)
            # pred_rel_class: 각 관계 예측에서 가장 높은 확률을 가진 클래스.
            # 차원: (num_rel_pairs,), list
            
            pred_rel_class += 1 # 0번 클래스는 bg class이므로 1을 더해줌.
            rel_confidence_threshold = self.pred_threshold[pred_rel_class] # pred_rel_class에 해당하는 임계값 가져오기. 
            
            if gsl_outputs is not None and self.use_gsl_output: # 논문 4.3 graph structure learner
                # MPNN 기반 ST-SGG를 사용하는 경우 적용. 
                # GSL: learn relevant and irrelevant relations between entities
                # use graph strcuture learner to give the confident pseudo-labels
                gsl_output = gsl_outputs[i].detach() # 현재 이미지의 그래프 구조 학습 결과, backpropagation을 통해 학습되지 않음.
                gsl_output = gsl_output[rel_pair[:,0], rel_pair[:,1]][n_annotate_label:] # 해당 관계에 대한 GSL 예측값을 가져오고, 새롭게 생성할 pseudo-label 후보만 고려.
                valid_pseudo_label_idx = (rel_confidence >= rel_confidence_threshold)[n_annotate_label:] # 신뢰도가 임계값 이상인 경우만 고려.
                valid_pseudo_label_idx = valid_pseudo_label_idx & (gsl_output == 1) # GSL이 유효한 관계라고 판단한 경우만 고려. 
                # 정리: valid_pseudo_label_idx은 GSL이 유효하다고 판단하고, 동시에 신뢰도가 충분히 높은 관계만 고려. 
                
                no_valid_pseudo_label_idx = (rel_confidence < rel_confidence_threshold)[n_annotate_label:] # 신뢰도가 임계값 미만인 경우만 고려.
                no_valid_pseudo_label_idx = no_valid_pseudo_label_idx & (gsl_output == 1) # GSL이 유효한 관계라고 판단한 경우만 고려.
            else: # graph structure learner을 사용하지 않는 경우
                valid_pseudo_label_idx = (rel_confidence >= rel_confidence_threshold)[n_annotate_label:]
                no_valid_pseudo_label_idx = (rel_confidence < rel_confidence_threshold)[n_annotate_label:]
            
            
            # Filter the non-overlap
            # 관계 후보 중에서 bbox가 겹치는 경우만 선택
            valid_pseudo_label_idx = valid_pseudo_label_idx & overlap_idx # 오버랩이 있는 객체 관계 중에서만 pseudo-label을 선택
            no_valid_pseudo_label_idx = no_valid_pseudo_label_idx & overlap_idx # 오버랩이 있는 관계들 중에서도 신뢰도가 낮은 관계.
                    
            
            # For pseudo-labeling and increasing the threshold
            # pseudo-label 후보 중 신뢰도가 높은 순서대로 정렬
            max_class_thres = torch.zeros(self.num_rel_cls) # pseudo-label로 선택된 관계 클래스의 개수를 저장.
            valid_rel_confidence = rel_confidence[n_annotate_label:][valid_pseudo_label_idx] # pseudo-label의 신뢰도.
            valid_rel_confidence, sort_ind = torch.sort(valid_rel_confidence, descending=True) # 신뢰도 내림차순 정렬, sort_ind: 정렬된 인덱스 정보.
            valid_pseudo_label = pred_rel_class[n_annotate_label:][valid_pseudo_label_idx][sort_ind] # pseudo-label의 클래스 정보 (신뢰도 높은 순서로 정렬됨).
            relative_idx = torch.nonzero(valid_pseudo_label_idx).view(-1)[sort_ind] # pseudo-label로 선택된 관계들의 원래 인덱스 정보 (신뢰도 높은 순서로 정렬됨)
            
            for p, c, rel_idx in zip(valid_pseudo_label, valid_rel_confidence, relative_idx):
                # p: pseudo-label로 선택된 관계의 클래스
                # c: pseudo-label로 선택된 관계의 신뢰도
                # 이미지 내에서 특정 클래스만 과도하게 pseudo-label로 선택되는 것 방지 (confirmation bias)
                if (self.pred_threshold[p] <= c).item(): # 신뢰도가 임계값 이상인 경우
                    # Constraint to the number of pseudo-label per image for preventing the confirmation bias
                    max_class_thres[p.item()] += 1 # 해당 클래스의 pseudo-label 개수 증가
                    self.batch_increase_conf_dict[p.item()].append(c.item()) # 신뢰도가 임계값 이상인 pseudo-label의 신뢰도 저장
                    
                    if max_class_thres[p.item()] > 3: # 같은 관계 클래스의 pseudo-label이 3개를 초과하면 더 이상 추가하지 않음.
                        valid_pseudo_label_idx[rel_idx] = False
                        continue
                    n_class[p] += 1 
                else: # 신뢰도가 임계값 미만인 경우
                    valid_pseudo_label_idx[rel_idx] = False # pseudo-label로 선택되지 않음.

            # For decaying the threshold  
            # 낮은 신뢰도를 가진 관계들은 임계값을 낮춰줘야 더 많은 관계가 pseudo-label로 선택될 수 있음.
            no_valid_pseudo_label = pred_rel_class[n_annotate_label:][no_valid_pseudo_label_idx] # pseudo-label이 되지 못한 클래스 추출
            no_valid_confidence = rel_confidence[n_annotate_label:][no_valid_pseudo_label_idx]
            
            for p, c in zip(no_valid_pseudo_label, no_valid_confidence):
                self.batch_decrease_conf_dict[p.item()].append(c.item()) # pseudo-label이 되지 못한 클래스의 신뢰도 저장.
                
            rel_pseudo_label = deepcopy(rel_labels[i].clone()) # GT Relation Label 복사
            rel_pseudo_label[n_annotate_label:][valid_pseudo_label_idx] = pred_rel_class[n_annotate_label:][valid_pseudo_label_idx] # pseudo-label을 할당해야 하는 관계들만 업데이트.
            rel_pseudo_labels.append(rel_pseudo_label) 

            gt_or_pseudo = torch.zeros((len(rel_pair)), dtype = torch.long)
            gt_or_pseudo[n_annotate_label:][valid_pseudo_label_idx] = 1
            gt_or_pseudo_list.append(gt_or_pseudo) 
            # pseudo-label인지 아닌지를 저장
            # 1: pseudo-label임
            
        if len(rel_pseudo_labels) == 0:
            rel_pseudo_labels = None
            
        for i in range(self.num_rel_cls):
            if i == 0 or n_class[i].item() == 0: continue 
            # n_class[i]: 현재 배치에서 클래스 i가 몇 개의 pseudo-label로 선택되었는지
            self.n_cumulative_class[i] += int(n_class[i].item()) # 전체 학습 동안 누적된 pseudo-label 개수

        return rel_pseudo_labels, torch.cat(gt_or_pseudo_list).cuda()
        # rel_pseudo_labels: pseudo-label이 반영된 관계 라벨 리스트
        # torch.cat(gt_or_pseudo_list).cuda(): pseudo-label인지 여부를 GPU로 반환
  
        