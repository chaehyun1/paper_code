# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.cfg = cfg.clone()
        self.backbone = build_backbone(cfg) # 백본 생성, 역할: 입력 이미지를 처리하고 기본적인 feature map을 생성.
        self.rpn = build_rpn(cfg, self.backbone.out_channels) # Region Proposal Network, 백본 네트워크의 출력을 입력으로 받음, 관심 영역(Region Proposals) 생성.
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels) # Region of Interest Heads 생성 
        # ROI Heads는 RPN에서 제안한 영역을 정교하게 처리하여 최종 출력(객체 클래스, 바운딩 박스 등)을 생성.

    def forward(self, images, targets=None, logger=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None: 
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images) # images를 ImageList로 변환
        features = self.backbone(images.tensors) # 백본 네트워크에 images(패딩된 것)를 입력하여 feature map을 생성
        proposals, proposal_losses = self.rpn(images, features, targets) # RPN 네트워크에 feature map을 입력하여 Region Proposals 생성
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets, logger) # CombinedROIHeads의 forward 함수 호출
            # ROI에서 객체 특징을 추출하고, 이를 기반으로 Object Detection과 관계 예측을 수행하여 최종 예측 결과와 loss을 반환

        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals # roi_heads가 없는 경우, RPN 결과(proposals)가 최종 출력.
            detector_losses = {}

        if self.training: # 학습 시 
            losses = {}
            losses.update(detector_losses)
            if not self.cfg.MODEL.RELATION_ON: 
                # During the relationship training stage, the rpn_head should be fixed, and no loss. 
                losses.update(proposal_losses)
            return losses # dict 형태의 losses 반환

        return result # 테스트 시, 탐지 결과를 반환
