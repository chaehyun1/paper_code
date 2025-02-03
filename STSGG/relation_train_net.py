# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""
import os
import sys
#sys.path.append(os.getcwd())
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import time
import datetime

import torch
from torch.nn.utils import clip_grad_norm_

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.checkpoint import clip_grad_norm
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger, debug_print
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
import random
import numpy as np

try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

SEED = 666
torch.cuda.manual_seed(SEED)          # GPU 난수 고정
torch.cuda.manual_seed_all(SEED)      # 여러 GPU 난수 고정
torch.manual_seed(SEED)               # CPU 난수 고정
random.seed(SEED)                     # Python 기본 난수 고정
np.random.seed(SEED)                  # NumPy 난수 고정

torch.backends.cudnn.enabled = True  # GPU를 사용하는 경우 CuDNN을 활용해 딥러닝 연산의 성능을 최적화
torch.backends.cudnn.benchmark = False # False!!, True는 입력 크기에 따라 알고리즘 선택이 달라질 수 있어 결과가 비결정적임
torch.backends.cudnn.deterministic = True # CuDNN의 연산 결과가 일정하게 나옴
torch.set_num_threads(4) # CPU 연산에 사용할 thread 수

torch.autograd.set_detect_anomaly(False) ##################################
# PyTorch의 autograd 과정에서 예외나 비정상 동작(예: NaN 발생, inf 값 등)을 탐지하는 기능을 설정
# False로 설정하면 예외나 비정상 동작을 탐지하지 않음

def train(cfg, local_rank, distributed, logger):
    debug_print(logger, 'prepare training') # logger에 'prepare training' 출력
    model = build_detection_model(cfg)  # GeneralizedRCNN(cfg) 객체 생성
    debug_print(logger, 'end model construction') # logger에 'end model construction' 출력

    # modules that should be always set in eval mode
    # their eval() method should be called after model.train() is called
    eval_modules = (model.rpn, model.backbone, model.roi_heads.box,) 
    fix_eval_modules(eval_modules) # eval_modules에 있는 모듈들을 eval 모드로 설정

    # NOTE, we slow down the LR of the layers start with the names in slow_heads
    # IMPPredictor라면, 특정 레이어(slow_heads)의 학습률(LR)을 낮추도록 지정
    if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "IMPPredictor": 
        slow_heads = ["roi_heads.relation.box_feature_extractor",
                      "roi_heads.relation.union_feature_extractor.feature_extractor",] 
    else:
        slow_heads = []

    # load pretrain layers to new layers
    # 기존에 학습된 레이어(key)의 가중치를 새로운 모델의 레이어(value)로 복사
    load_mapping = {"roi_heads.relation.box_feature_extractor" : "roi_heads.box.feature_extractor",
                    "roi_heads.relation.union_feature_extractor.feature_extractor" : "roi_heads.box.feature_extractor"}
    

    device = torch.device(cfg.MODEL.DEVICE) # device 설정
    model.to(device)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    num_batch = cfg.SOLVER.IMS_PER_BATCH # 배치 사이즈
    optimizer = make_optimizer(cfg, model, logger, slow_heads=slow_heads, slow_ratio=10.0, rl_factor=float(num_batch)) # optimizer 생성
    scheduler = make_lr_scheduler(cfg, optimizer, logger) # scheduler 생성 (학습률을 동적으로 조정)
    debug_print(logger, 'end optimizer and shcedule') # logger에 'end optimizer and shcedule' 출력
    
    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16" # mixed precision 사용 여부 설정
    # Mixed Precision: 모델의 일부 연산을 float16 데이터 타입으로 수행하여 GPU 메모리 사용량을 줄이고, 학습 속도를 높이는 기법
    amp_opt_level = 'O1' if use_mixed_precision else 'O0' 
    # NVIDIA의 Apex 라이브러리에서 제공하는 AMP(Automatic Mixed Precision) 최적화 레벨을 설정
    # 'O0': 혼합 정밀도 비활성화 (기본 float32 사용).
    # 'O1': 자동 혼합 정밀도 사용 (성능과 안정성의 균형).
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level) # AMP는 모델과 옵티마이저를 wrapping하여 혼합 정밀도 학습이 가능하도록 설정

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel( # 여러 GPU에서 학습을 병렬로 수행
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    debug_print(logger, 'end distributed')


    output_dir = cfg.OUTPUT_DIR # 학습 결과를 저장할 디렉토리 경로 설정

    save_to_disk = get_rank() == 0  # 현재 실행 중인 프로세스가 마스터 프로세스인지 확인
    # 분산 학습에서는 여러 GPU/노드에서 동시에 모델을 학습시키기 때문에, 체크포인트를 여러 번 저장하지 않도록 마스터 프로세스만 저장 수행
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
    ) # DetectronCheckpointer 객체 생성
    
    # if there is certain checkpoint in output_dir, load it, else load pretrained detector
    checkpoint = None 
    if cfg.MODEL.WEIGHT == "": # config 파일에 설정된 모델 가중치가 없는 경우
        checkpointer.load(
            cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping
        )  # config 파일에서 Pretrained Detector의 체크포인트를 로드
    else: # config 파일에 설정된 모델 가중치가 있는 경우
        checkpoint = checkpointer.load(
            cfg.MODEL.WEIGHT, 
            with_optim=False,
        ) # config 파일에서 설정된 모델 가중치를 로드
    if checkpoint: # 위의 if-else 중 후자의 경우 (특정 체크포인트에서 학습을 재개할 가능성이 있음)
        checkpoint['iteration'] = 0 # 새로운 학습을 시작하기 위해 iteration 리셋
    arguments = {} # 학습 중에 사용할 인자들을 저장할 딕셔너리
    arguments["iteration"] = 0 # iteration 초기화
    # arguments["iteration"] = checkpoint['iteration']
    debug_print(logger, 'end load checkpointer')
    
    train_data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    ) # 학습 데이터 로더 생성
    val_data_loaders = make_data_loader(
        cfg,
        mode='val',
        is_distributed=distributed,
    ) # 검증 데이터 로더 생성
    debug_print(logger, 'end dataloader')
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD # 체크포인트 저장 주기 설정

    if cfg.SOLVER.PRE_VAL: # 학습 전에 검증을 수행할 경우
        logger.info("Validate before training")
        run_val(cfg, model, val_data_loaders, distributed, logger)

    logger.info("Start training")
    meters = MetricLogger(delimiter="  ") 
    max_iter = len(train_data_loader) # 전체 학습 데이터의 반복 횟수
    start_iter = arguments["iteration"] # 학습이 시작되는 iteration 번호
    start_training_time = time.time()
    end = time.time()
    pre_clser_pretrain_on = False 
    if (
        cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.PRETRAIN_RELNESS_MODULE
        and cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON
    ):
        if distributed:
            m2opt = model.module
        else:
            m2opt = model
        m2opt.roi_heads.relation.predictor.start_preclser_relpn_pretrain() # Graph 사용 x
        logger.info("Start preclser_relpn_pretrain")
        pre_clser_pretrain_on = True 
        STOP_ITER = (
            cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.PRETRAIN_ITER_RELNESS_MODULE
        ) # 특정 반복 이후에는 사전 학습을 중단
    print_first_grad = True 
        
    for iteration, (images, targets, _) in enumerate(train_data_loader, start_iter):
        # train_data_loader: (images, targets, _)
        # targets: 이미지에 대한 GT 정보, 배치 내 각 이미지마다 1개의 target 딕셔너리
        # images: (batch_size, C, H, W)
        # start_iter부터 시작 
        # targets: 현재 배치의 레이블(타겟)

        if any(len(target) < 1 for target in targets): # 각 타겟(target)이 비어 있는지 확인 -> targets 중 하나라도 비어 있으면 에러 로그를 기록 
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        model.train() # 모델을 학습 모드로 설정
        fix_eval_modules(eval_modules) # model.rpn, model.backbone, model.roi_heads.box를 eval 모드로 설정

        if pre_clser_pretrain_on: # 특정 사전 학습 단계가 진행 중인 경우
            if iteration == STOP_ITER: # 특정 반복 이후에는 사전 학습을 중단
                logger.info("pre clser pretraining ended.")
                m2opt.roi_heads.relation.predictor.end_preclser_relpn_pretrain() # 사전 학습 종료
                pre_clser_pretrain_on = False # 사전 학습 종료

        images = images.to(device) # 한 배치 내 이미지들 
        targets = [target.to(device) for target in targets] # 

        loss_dict = model(images, targets) # GeneralizedRCNN의 forward 실행, loss dict 반환

        losses = sum(loss for loss in loss_dict.values()) # 모든 loss의 합

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict) # 모든 GPU에서 계산된 손실값을 평균하여 하나의 통합 손실값을 만든다. 
        losses_reduced = sum(loss for loss in loss_dict_reduced.values()) # 모든 손실 값을 합산한 총 손실
        meters.update(loss=losses_reduced, **loss_dict_reduced) 

        optimizer.zero_grad() # optimizer의 gradient 초기화
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses: 
            scaled_losses.backward() 

        # add clip_grad_norm from MOTIFS, tracking gradient, used for debug
        # Gradient Clipping
        verbose = (iteration % cfg.SOLVER.PRINT_GRAD_FREQ) == 0 or print_first_grad # print grad or not
        print_first_grad = False
        clip_grad_norm([(n, p) for n, p in model.named_parameters() if p.requires_grad], max_norm=cfg.SOLVER.GRAD_NORM_CLIP, logger=logger, verbose=verbose, clip=True)

        optimizer.step() # 모델 파라미터 업데이트

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 200 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join( 
                    [
                        "\ninstance name: {instance_name}\n",
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                        "gpu: {cur_gpu}\n",
                    ]
                ).format(
                    instance_name=cfg.OUTPUT_DIR[len("checkpoints/") :],
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[-1]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    cur_gpu=os.environ['CUDA_VISIBLE_DEVICES']
                )
            ) # 학습 중간 결과 출력

        # if iteration % checkpoint_period == 0:
        if iteration % checkpoint_period == 0:
        # if iteration % checkpoint_period == 0 and iteration > 500:
            checkpointer.save("model_{:07d}".format(iteration), **arguments) # 중간 체크포인트 저장
            
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

        val_result = None # used for scheduler updating
        # if cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD == 0:
        if cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD == 0:
        # The above code is checking if the configuration variable `TO_VAL` is True and if the current
        # iteration number is a multiple of `VAL_PERIOD` or if the current iteration number is equal
        # to 400. If either of these conditions is true, some code block will be executed.
        # if cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD == 0 or iteration > 500:
            logger.info(f"Start validating - {iteration}")
            val_result = run_val(cfg, model, val_data_loaders, distributed, logger)
            logger.info("Validation Result: %.4f" % val_result)
            # if get_rank() == 0:
            #     for each_ds_eval in val_result[0]:
            #         for each_evalator_res in each_ds_eval[1]:
            #             logger.log(4, (each_evalator_res, iteration))
 
        # scheduler should be called after optimizer.step() in pytorch>=1.1.0
        # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        if cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau": 
            scheduler.step(val_result, epoch=iteration)
            if scheduler.stage_count >= cfg.SOLVER.SCHEDULE.MAX_DECAY_STEP: # 학습률이 일정 횟수 이상 감소되었음에도 성능 개선이 없으면 학습을 종료.
                logger.info("Trigger MAX_DECAY_STEP at iteration {}.".format(iteration))
                break
        else:
            scheduler.step() 

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False,
                                              update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
    return model

def fix_eval_modules(eval_modules):
    for module in eval_modules:
        for _, param in module.named_parameters(): # module의 파라미터를 가져옴
            param.requires_grad = False # 파라미터의 requires_grad를 False로 설정
        # DO NOT use module.eval(), otherwise the module will be in the test mode, i.e., all self.training condition is set to False

def run_val(cfg, model, val_data_loaders, distributed, logger):
    if distributed:
        model = model.module # 분산 학습 환경에서 래핑된 모델의 원래 구조에 접근하기 위한 설정
    torch.cuda.empty_cache() # GPU 메모리 해제
    # iou_types = ("bbox",)
    iou_types = () 
    if cfg.MODEL.MASK_ON: 
        iou_types = iou_types + ("segm",) 
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )

    dataset_names = cfg.DATASETS.VAL # 검증 데이터셋 이름 설정
    val_result = []
    for dataset_name, val_data_loader in zip(dataset_names, val_data_loaders):
        dataset_result = inference(
                            cfg,
                            model,
                            val_data_loader,
                            dataset_name=dataset_name,
                            iou_types=iou_types,
                            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                            device=cfg.MODEL.DEVICE,
                            expected_results=cfg.TEST.EXPECTED_RESULTS,
                            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                            output_folder=None,
                            logger=logger,
                        ) # 검증 데이터셋에 대한 inference 수행
        synchronize() # 분산 학습 중 모든 프로세스가 같은 상태에서 시작하도록 보장
        val_result.append(dataset_result)
    # support for multi gpu distributed testing
    gathered_result = all_gather(torch.tensor(dataset_result).cpu()) # 모든 프로세스에서 수행한 결과를 수집
    gathered_result = [t.view(-1) for t in gathered_result] # 수집한 결과를 하나의 리스트로 변환
    gathered_result = torch.cat(gathered_result, dim=-1).view(-1) # 수집한 결과를 하나의 텐서로 변환
    valid_result = gathered_result[gathered_result>=0] # 수집한 결과 중 유효한 결과만 추출
    val_result = float(valid_result.mean()) # 유효한 결과의 평균값 계산
    del gathered_result, valid_result # 메모리 해제
    torch.cuda.empty_cache()
    return val_result 

def run_test(cfg, model, distributed, logger):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()
    iou_types = ()
    # iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )
    output_folders = [None] * len(cfg.DATASETS.TEST) 
    dataset_names = cfg.DATASETS.TEST
    
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, mode='test', is_distributed=distributed)
    
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            logger=logger,
        ) # 테스트 데이터셋에 대한 inference 수행
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    
    # Scene Graph Generation을 위한 딥러닝 모델 학습 및 평가 설정
    # weakly supervised
    parser.add_argument(
        "--config-file",
        default="configs/wsup-50.yaml", # wsup-50_check, sup-1000 sup-50_motif_vctree
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0) # 분산 학습을 위한 로컬 랭크 설정
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    
    # 동일한 구성 파일을 유지하면서도 다양한 설정을 명령줄에서 쉽게 변경
    # --config-file을 기반으로, 명령줄에서 전달되는 추가 옵션(opts)으로 특정 설정을 동적으로 수정할 수 있음
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER, # 고정된 개수가 아닌, 나머지 모든 명령줄 인자를 받음
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1 # 분산 학습에 사용될 GPU의 개수를 지정
    args.distributed = num_gpus > 1 # 분산 학습 여부를 확인, num_gpus가 1보다 크면 분산 학습으로 설정

    if args.distributed:
        torch.cuda.set_device(args.local_rank) # 각 프로세스가 사용할 GPU를 설정
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        ) # PyTorch의 분산 학습 기능을 초기화
        synchronize() # 분산 학습 중 모든 프로세스가 같은 상태에서 시작하도록 보장

    cfg.set_new_allowed(True) # cfg 객체에 새로운 속성을 추가할 수 있도록 설정
    cfg.merge_from_file(args.config_file) # config 파일을 cfg 객체에 병합
    cfg.merge_from_list(args.opts) # config 파일에 있는 설정을 command line에서 변경된 설정으로 업데이트
    cfg.set_new_allowed(True) # cfg 객체에 새로운 속성을 추가할 수 있도록 설정

    cfg.freeze() # cfg 객체를 변경할 수 없도록 설정

    output_dir = cfg.OUTPUT_DIR # 학습 결과를 저장할 디렉토리 경로를 설정
    if output_dir:
        mkdir(output_dir) # 디렉토리가 존재하지 않으면 생성

    logger = setup_logger("ST-SGG", output_dir, get_rank()) # logger 설정
    logger.info("Using {} GPUs".format(num_gpus)) # 사용할 GPU의 개수를 출력
    # if cfg.DEBUG:
    #     logger.info("Collecting env info (might take some time)")
    #     logger.info("\n" + collect_env_info())

    # logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf: 
        config_str = "\n" + cf.read() # config 파일의 내용을 읽어서 config_str에 저장
        # logger.info(config_str)

    output_config_path = os.path.join(cfg.OUTPUT_DIR, "config.yml") # config 파일을 저장할 경로 설정
    logger.info("Saving config into: {}".format(output_config_path)) # config 파일을 저장할 경로 출력
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path) # config 파일을 저장

    model = train(cfg, args.local_rank, args.distributed, logger) # 학습을 진행하고 학습된 모델을 반환

    if not args.skip_test: # skip_test가 False인 경우
        run_test(cfg, model, args.distributed, logger) # 테스트를 진행


if __name__ == "__main__":
    main()
