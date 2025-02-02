# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from ..attribute_head.roi_attribute_feature_extractors import make_roi_attribute_feature_extractor
from ..box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_relation_feature_extractors import make_roi_relation_feature_extractor
from .roi_relation_predictors import make_roi_relation_predictor
from .inference import make_roi_relation_post_processor
from .loss import make_roi_relation_loss_evaluator, make_weaksup_relation_loss_evaluator
from .sampling import make_roi_relation_samp_processor, make_weaksup_roi_relation_sample_processor
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_kern import (
    to_onehot,
)


class ROIRelationHead(torch.nn.Module):
    """
    Generic Relation Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIRelationHead, self).__init__()
        self.cfg = cfg.clone()
        self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES # Object Detection에서 사용할 클래스 개수.
        self.num_rel_cls = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES # Relation Detection에서 사용할 클래스 개수.
        # same structure with box head, but different parameters
        # these param will be trained in a slow learning rate, while the parameters of box head will be fixed
        # Note: there is another such extractor in uniton_feature_extractor
        self.union_feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels) # 객체 간 관계 예측을 위해 두 객체의 Union Box에서 특징을 추출
        
        if cfg.MODEL.ATTRIBUTE_ON: # 실행 안함 
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=True)
            self.att_feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=True)
            feat_dim = self.box_feature_extractor.out_channels * 2
        else:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels) # ROI의 특징을 추출
            feat_dim = self.box_feature_extractor.out_channels
            
        self.predictor = make_roi_relation_predictor(cfg, feat_dim) # NOTE: 중요, MotifPredictor_STSGG의 인스턴스 생성
        self.post_processor = make_roi_relation_post_processor(cfg)
        self.object_cls_refine = cfg.MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE # True
        self.pass_obj_recls_loss = cfg.MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS # True
        self.weak_sup = False 
        if cfg.WSUPERVISE.METHOD == "VG_DS": # weak supervised인 경우 
            self.weak_sup = True
            self.wsup_samp_processor = make_weaksup_roi_relation_sample_processor(cfg)
            self.wsup_loss_evaluator = make_weaksup_relation_loss_evaluator(cfg) 
        # statistics = get_dataset_statistics(cfg)
        self.obj_mapping = None
        # For VG-1800
        # if len(statistics['obj_classes']) > 10000:
        #     print(str("/home/public/Datasets/CV/vg_1800/{}.pt".format(len(statistics['obj_classes']))))
        #     self.obj_mapping = torch.load(str("/home/public/Datasets/CV/vg_1800/{}.pt".format(len(statistics['obj_classes']))))

        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX: # True
            if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL: # True
                self.mode = "predcls"
            else:
                self.mode = "sgcls"
        else:
            self.mode = "sgdet"

        self.em_E_step = cfg.EM.MODE == "E"
        self.eval_only_acc = cfg.TEST.ONLY_ACC # False

        self.loss_evaluator = make_roi_relation_loss_evaluator(cfg) # RelationLossComputation의 인스턴스 생성, 논문 식 loss가 포함된 클래스
        self.samp_processor = make_roi_relation_samp_processor(cfg)
        self.rwt_dict = torch.zeros(self.num_rel_cls).cuda()  
        # parameters
        self.use_union_box = self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION # True
        
        # if len(statistics['obj_classes']) > 10000:
        #     print(str("/home/public/Datasets/CV/vg_1800/{}.pt".format(len(statistics['obj_classes']))))
        #     self.obj_mapping = torch.load(str("/home/public/Datasets/CV/vg_1800/{}.pt".format(len(statistics['obj_classes']))))


        
    def forward(self, features, proposals, targets=None, logger=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes. Note: it has been post-processed (regression, nms) in sgdet mode
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        rel_pn_labels = None # For learning the structure
        if self.training:
            # relation subsamples and assign ground truth label during training
            with torch.no_grad(): 
                # Faster R-CNN이 탐지한 객체들 중에서 GT 관계가 있는 객체 쌍(Positive Pairs)과 관계가 없는 쌍(Negative Pairs)을 샘플링.
                if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX: # True
                    if self.weak_sup and self.training: # False
                        proposals, rel_labels, rel_pair_idxs, rel_binarys = self.wsup_samp_processor.gtbox_relsample(proposals, targets)
                    else:
                        proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.gtbox_relsample(proposals, targets) 
                    rel_pn_labels = rel_labels 
                else:
                    if self.weak_sup and self.training:
                        proposals, rel_labels, rel_pair_idxs, rel_binarys = self.wsup_samp_processor.detect_relsample(proposals, targets)
                        rel_pn_labels = rel_labels
                    else:
                        proposals, rel_labels, rel_pair_idxs, rel_binarys, rel_labels_all = self.samp_processor.detect_relsample(proposals, targets)
                        rel_pn_labels = rel_labels_all
        elif self.em_E_step:
            rel_labels, rel_binarys = None, None
            rel_pair_idxs = [t.get_field("relation_pair_idxs") for t in targets]
        elif self.eval_only_acc:
            rel_labels, rel_binarys = None, None
            rel_pair_idxs = []
            for t in targets:
                idxs = t.get_field("relation").nonzero()
                if len(idxs) > 0:
                    rel_pair_idxs.append(idxs.long())
                else:
                    rel_pair_idxs.append(torch.zeros((1, 2), dtype=torch.int64, device=features[0].device))
        else:
            rel_labels, rel_binarys = None, None
            rel_pair_idxs = self.samp_processor.prepare_test_pairs(features[0].device, proposals)
            
        # For predcls
        if self.mode == "predcls":
            for proposal in proposals:
                obj_labels = proposal.get_field("labels")
                # VG1800
                if self.obj_mapping is not None:
                    proposal.add_field("predict_logits", to_onehot(obj_labels, 70099))
                else:
                    proposal.add_field("predict_logits", to_onehot(obj_labels, self.num_obj_cls))
                proposal.add_field("pred_scores", torch.ones(len(obj_labels)).to(features[0].device))
                proposal.add_field("pred_labels", obj_labels)
        # else:
        #     if not self.object_cls_refine:
        #         for each in proposals:
        #             each.extra_fields['pred_labels'] = torch.max(each.extra_fields['predict_logits'][:,1:], 1).indices + 1            

        roi_features = self.box_feature_extractor(features, proposals)

        if self.use_union_box:
            union_features = self.union_feature_extractor(features, proposals, rel_pair_idxs)
        else:
            union_features = None

     
        refine_logits, relation_logits, add_losses, rel_pesudo_labels, one_hot_gt_or_pseudo = self.predictor(proposals, rel_pair_idxs, rel_pn_labels, rel_binarys, roi_features, union_features, logger)
        # MotifPredictor_STSGG forward 함수 호출

        # for em_E_step
        if self.em_E_step:
            return None, relation_logits, {}

        # for test
        if not self.training:
            if not self.object_cls_refine:
                refine_logits = [prop.extra_fields['predict_logits'] for prop in proposals]
            result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, proposals)
            return roi_features, result, {}


        # # Change to pseudo label
        if rel_pesudo_labels is not None:
            new_label = []
            if self.cfg.IETRANS.RWT:
                for i in range(len(rel_labels)):
                    label = rel_labels[i].clone()
                    label[rel_pesudo_labels[i] > 0, 0] = -self.rwt_dict[rel_pesudo_labels[i][rel_pesudo_labels[i] > 0]]
                    label[rel_pesudo_labels[i] > 0, rel_pesudo_labels[i][rel_pesudo_labels[i] > 0]] = 1
                    new_label.append(label)
            else:
                for i in range(len(rel_labels)):
                    label = rel_labels[i].clone()
                    label[rel_pesudo_labels[i] > 0, rel_pesudo_labels[i][rel_pesudo_labels[i] > 0]] = 1
                    label[rel_pesudo_labels[i] > 0, 0] = 0
                    new_label.append(label)
            
            rel_labels = new_label



        if self.weak_sup and self.training:
            attn_score = None
            if "attn_score" in add_losses:
                attn_score = add_losses.pop("attn_score")
            loss_relation, loss_refine = self.wsup_loss_evaluator(proposals, rel_labels, relation_logits, refine_logits, attn_score, one_hot_gt_or_pseudo)
        else:
            loss_relation, loss_refine = self.loss_evaluator(proposals, rel_labels, relation_logits, refine_logits, one_hot_gt_or_pseudo)
            # 논문 식 (1) loss 계산 포함
            # 관계 예측 loss, 객체 분류 loss

        if self.cfg.MODEL.ATTRIBUTE_ON and isinstance(loss_refine, (list, tuple)):
            output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine[0], loss_refine_att=loss_refine[1])
        else:
            if self.pass_obj_recls_loss:
                output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine)
            else:
                output_losses = dict(loss_rel=loss_relation)

        output_losses.update(add_losses)

        return roi_features, proposals, output_losses


def build_roi_relation_head(cfg, in_channels):
    """
    Constructs a new relation head.
    By default, uses ROIRelationHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIRelationHead(cfg, in_channels)