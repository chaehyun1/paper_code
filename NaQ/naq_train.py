import os
import time
import numpy as np
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore')

import arg

from utils import *
from data import *
from task_generator import meta_batch_generator, supervised_task_generator, downstream_task_generator
from meta import UnsupMAML


def main(args, config, Data):    
    # Unsupervised Meta-learning via Neighbors as Queries
    assert args.query_generation == 'NaQ'
    Meta = UnsupMAML(args, config, Data).to(device) 

    valid_num, test_num = args.num_valid_tasks, args.num_downstream_tasks # 50, 500

    # Sample a pool of Meta-batches for training
    train_pool = [meta_batch_generator(args, Data) for _ in range(args.epoch)]
    # args.epoch만큼 meta_batch_generator를 수행하여 train_pool에 저장
    # 각 epoch마다 meta_batch_size만큼의 meta-task를 생성
    
    # Sample a pool of Valid/Test tasks
    valid_pool, test_pool = downstream_task_generator(args, Data, args.n_way, args.k_shot_test)
    
    # 위 코드까지: Unsupervised Episode Generation 수행 완료 
    # 아래 코드: 학습 및 평가 수행

    # Define 'Empty Cans' to store train/valid/test accs
    meta_train_loss, meta_train_acc = [], []
    valid_acc = []
    test_acc, test_stdev = [], []

    del Data.Diffusion_matrix # 메모리 절약을 위해 Diffusion matrix 삭제

    train_start = time.time()
    for epoch, (id_spt, y_spt, id_query, y_query) in enumerate(train_pool):
        epoch_start = time.time()

        #------------------------------<Training Phase>------------------------------#
        train_loss, train_acc = Meta(id_spt, y_spt.to(device), id_query, y_query.to(device)) # UnsupMAML의 forward 함수

        # Record Training Loss & Acc and Report training accuracy
        meta_train_loss.append(train_loss)
        meta_train_acc.append(train_acc)
        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch}- Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} / time_elapsed: {epoch_time:.4f} seconds')

        # -------------------------<Validation & Test Phase>-------------------------#
        if (epoch + 1) % 10 == 0: # 10 epochs마다 validation과 test 수행
            val_temp, test_temp = [], []
            # Validation for 50 validation tasks
            for id_spt, y_spt, id_query, y_query in valid_pool: # 50개의 validation task에 대해 수행
                _, val_acc = Meta.fine_tuning(id_spt, y_spt.to(device), id_query, y_query.to(device))
                val_temp.append(val_acc)
            val_temp = np.array(val_temp).mean(axis=0) # 전체 validation task에 대한 평균 accuracy
            valid_acc.append(val_temp) # Record validation accuracy for selection

            # Testing for 500 downstream tasks
            for id_spt, y_spt, id_query, y_query in test_pool: # 500개의 test task에 대해 수행
                _, tst_acc = Meta.fine_tuning(id_spt, y_spt.to(device), id_query, y_query.to(device))
                test_temp.append(tst_acc)
            test_temp = np.array(test_temp)
            test_temp, test_std = test_temp.mean(axis=0), test_temp.std(axis=0)/np.sqrt(len(test_pool))
            # Record testing accuracy and stdev
            test_acc.append(test_temp)
            test_stdev.append(test_std)

            # Reporting Valid/Test acc by 10 epochs
            print(f'[Valid/Test] epoch {epoch}- Valid Acc: {val_temp:.4f} / Test Acc: {test_temp:.4f}')
        
    train_time = time.time() - train_start
    print(f'Training ended with total elapsed time {train_time:.4f} seconds')

    # Reporting the best performance based on validation
    meta_train_acc, valid_acc, test_acc, test_stdev = np.array(meta_train_acc), np.array(valid_acc), np.array(test_acc), np.array(test_stdev)
    Best_ix = np.argmax(valid_acc) # Best validation accuracy의 index -> 최적의 파라미터일 때 test 성능을 알기 위해서 
    Best_epoch = Best_ix * 10 - 1
    print('<Best Performance (with 95%% confidence interval)>')
    print(f'Valid Acc: {valid_acc[Best_ix]:.4f} / Test Acc: {test_acc[Best_ix]:.4f}±{1.96*test_stdev[Best_ix]:.4f} at epoch {Best_epoch}')

    return valid_acc[Best_ix], test_acc[Best_ix], test_stdev[Best_ix]





if __name__ == '__main__':
    torch.set_num_threads(4) # CPU 연산에 사용할 스레드의 최대 개수 설정
    seed = 1212
    seed_everything(seed) # Set random seed

    setting = 'unsup' # unsupervised setting

    # Parse arguments
    args = arg.parse_args(setting)

    #------------------------------<Loading & Preprocessing Data>------------------------------#
    load_started = time.time()
    print(args) # Checking experimental settings

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    # torch.cuda.set_device(device)

    # Loading Data
    Data = load_data(args) # GraphData 객체 반환
    if device != 'cpu':
        Data.set_device(device) # Set device for data
    
    # Applying graph Diffusion
    Data.add_diffusion(args) # node similarity matrix or diffusion matrix 생성

    num_node_feat = Data.features.shape[1] 

    # Model configuration, Here, args.latent value is set to be 64.
    config = [
        ('GCN', (num_node_feat, 2*args.latent), True),
        ('GCN', (2*args.latent, args.latent), True),
        ('linear', (args.latent, args.n_way), True)
    ]

    print(config) # Checking model configuration

    print(f'Data loading and preprocessing have been ended in {time.time()-load_started:4f} seconds')

    # Result recording
    os.makedirs(f'./results/{args.dataset}/', exist_ok=True)
    results = open(f'./results/{args.dataset}/{args.setting}_{args.query_generation}.txt', 'a')
    results.write('\n====================================================================================================\n')

    # Record hyperparameter settings
    for arg in vars(args).keys(): # vars(args)는 args의 속성을 dictionary로 반환
        results.write(f'{arg}={vars(args)[arg]}, ')
    results.write('\n')

    #------------------------------<Run Experiments>------------------------------#
    Exp_started = time.time()

    Valid_summaries, Test_summaries = [], []
    
    val_acc, test_acc, test_std = main(args, config, Data) # Run the main function

    Total_elapsed = time.time() - Exp_started
    print(f'Total time elapsed in {Total_elapsed:.4f} seconds')

    # Record the summary of the experimental results with 95% C.I.
    results.write(f'Valid Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}±{1.96*test_std:.4f}, Total elapsed: {Total_elapsed:.4f} seconds')

    results.write('\n====================================================================================================\n')
    results.close()
