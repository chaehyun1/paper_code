import contextlib
import logging
import os
import glob

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from utils import *
from pre_train.sasrec.model import SASRec


def load_checkpoint(recsys, pre_trained):
    path = f'pre_train/{recsys}/{pre_trained}/' # 해당 데이터에 대해 사전 학습된 추천 모델의 경로
    
    pth_file_path = find_filepath(path, '.pth') # path안에 있는 pth 파일의 경로를 찾음
    assert len(pth_file_path) == 1, 'There are more than two models in this dir. You need to remove other model files.\n'
    # 해당 경로에 있는 pth 파일이 1개인지 확인
    kwargs, checkpoint = torch.load(pth_file_path[0], map_location="cpu") 
    logging.info("load checkpoint from %s" % pth_file_path[0]) # 파일 로드가 성공했음을 기록

    return kwargs, checkpoint

class RecSys(nn.Module):
    def __init__(self, recsys_model, pre_trained_data, device):
        super().__init__()
        kwargs, checkpoint = load_checkpoint(recsys_model, pre_trained_data) # 해당 데이터에 대해 사전 학습된 추천 모델의 인자와 체크포인트를 불러옴
        kwargs['args'].device = device
        model = SASRec(**kwargs) # 모델 초기화 
        model.load_state_dict(checkpoint) # 체크포인트 불러오기
            
        for p in model.parameters(): # 모델의 파라미터를 고정시킴 (동결)
            p.requires_grad = False
            
        self.item_num = model.item_num # 추천 모델의 아이템 수
        self.user_num = model.user_num # 추천 모델의 유저 수
        self.model = model.to(device)
        self.hidden_units = kwargs['args'].hidden_units 
        
    def forward():
        print('forward')