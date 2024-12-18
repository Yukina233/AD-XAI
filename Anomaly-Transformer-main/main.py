import gc
import os
import argparse
from glob import glob

import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.backends import cudnn
from utils.utils import *

from solver import Solver_modified


def str2bool(v):
    return v.lower() in ('true')


def run(args):
    solver_args = args.copy()
    seed = solver_args['seed']
    train_files = glob(os.path.join(solver_args['data_path'], '*train*'))
    test_files = glob(os.path.join(solver_args['data_path'], '*test*'))

    os.makedirs(os.path.join(solver_args['path_results'], f'{seed}/scores'), exist_ok=True)
    for id, file in enumerate(train_files):
        print(f"Train file {id}: {file}")
        solver_args['data_path'] = file

        solver = Solver_modified(solver_args)

        solver.train()

    auc_roc_list = []
    auc_pr_list = []
    for id, file in enumerate(test_files):
        print(f"Test file {id}: {file}")
        solver_args['data_path'] = file

        solver = Solver_modified(solver_args)

        scores, labels = solver.predict()
        scores.tofile(os.path.join(solver_args['path_results'], f"{seed}/scores/anomaly_scores-{id}.csv"), sep="\n")

        # 计算AUC-ROC和AUC-PR
        auc_roc = roc_auc_score(labels, scores)
        auc_pr = average_precision_score(labels, scores)

        auc_roc_list.append(auc_roc)
        auc_pr_list.append(auc_pr)

    # 创建包含AUC-ROC和AUC-PR值的DataFrame
    result = pd.DataFrame({'AUC-ROC': auc_roc_list, 'AUC-PR': auc_pr_list})

    # 计算均值
    mean_auc_roc = result['AUC-ROC'].mean()
    mean_auc_pr = result['AUC-PR'].mean()

    # 将均值添加为DataFrame的新行
    mean_row = pd.DataFrame({'AUC-ROC': [mean_auc_roc], 'AUC-PR': [mean_auc_pr]}, index=['Mean'])
    result = pd.concat([result, mean_row])

    # 保存结果到CSV文件
    results_csv_path = os.path.join(solver_args['path_results'], f'{seed}/results.csv')
    result.to_csv(results_csv_path, index=True)  # index=True保留索引，这样'Mean'也会被写入文件

    print('Results saved.')


def run_1_by_1(args):
    solver_args = args.copy()
    seed = solver_args['seed']
    dataset_files = glob(os.path.join(solver_args['data_path'], '*'))

    os.makedirs(os.path.join(solver_args['path_results'], f'{seed}/models'), exist_ok=True)
    os.makedirs(os.path.join(solver_args['path_results'], f'{seed}/scores'), exist_ok=True)

    auc_roc_list = []
    auc_pr_list = []

    for id, file in enumerate(dataset_files):
        print(f"Test file {id}: {file}")

        solver_args['data_path'] = file

        solver = Solver_modified(solver_args)

        solver.train()

        scores, labels = solver.predict()
        scores.tofile(os.path.join(solver_args['path_results'], f"{seed}/scores/anomaly_scores-{id}.csv"), sep="\n")

        # 计算AUC-ROC和AUC-PR
        auc_roc = roc_auc_score(labels, scores)
        auc_pr = average_precision_score(labels, scores)

        auc_roc_list.append(auc_roc)
        auc_pr_list.append(auc_pr)

        del solver
        gc.collect()

    # 创建包含AUC-ROC和AUC-PR值的DataFrame
    result = pd.DataFrame({'AUC-ROC': auc_roc_list, 'AUC-PR': auc_pr_list})

    # 计算均值
    mean_auc_roc = result['AUC-ROC'].mean()
    mean_auc_pr = result['AUC-PR'].mean()

    # 将均值添加为DataFrame的新行
    mean_row = pd.DataFrame({'AUC-ROC': [mean_auc_roc], 'AUC-PR': [mean_auc_pr]}, index=['Mean'])
    result = pd.concat([result, mean_row])

    # 保存结果到CSV文件
    results_csv_path = os.path.join(solver_args['path_results'], f'{seed}/results.csv')
    result.to_csv(results_csv_path, index=True)  # index=True保留索引，这样'Mean'也会被写入文件

    print('Results saved.')


path_project = '/home/yukina/Missile_Fault_Detection/project_data'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=1)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--input_c', type=int, default=180)
    parser.add_argument('--output_c', type=int, default=180)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='TLM-RATE')
    # parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])

    parser.add_argument('--anormly_ratio', type=float, default=4.00)

    config = parser.parse_args()

    config.data_path = os.path.join(path_project,
                                    f'data/{config.dataset}/yukina_data/DeepSAD_data, window=10, step=2')
    config.path_results = os.path.join(path_project, f'Anomaly-Transformer-main/results/{config.dataset}/num_epochs={config.num_epochs}, win_size={config.win_size}, step={config.step}, lr={config.lr}')
    config.model_save_path = os.path.join(path_project, f'Anomaly-Transformer-main/checkpoints/{config.dataset}/num_epochs={config.num_epochs}, win_size={config.win_size}, step={config.step}, lr={config.lr}')
    if config.dataset == 'SMD':
        config.input_c = 180
        config.output_c = 180
    elif config.dataset == 'Metro':
        config.input_c = 5
        config.output_c = 5
    elif config.dataset == 'GHL':
        config.input_c = 80
        config.output_c = 80
    elif config.dataset == 'SWAT':
        config.input_c = 255
        config.output_c = 255
    elif config.dataset == 'TLM-RATE':
        config.input_c = 48
        config.output_c = 48

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)

    all_scores = []
    AUCROC_seed = []
    AUCPR_seed = []
    for seed in range(3):
        args['seed'] = seed
        args['model_save_path'] = os.path.join(config.model_save_path, f'seed={seed}')
        if args['dataset'] == 'SMD':
            run_1_by_1(args)
        else:
            run(args)
        result = pd.read_csv(os.path.join(args['path_results'], f'{seed}/results.csv'), index_col=0)
        AUCROC_seed.append(result['AUC-ROC'].values)
        AUCPR_seed.append(result['AUC-PR'].values)

        scores = pd.read_csv(os.path.join(args['path_results'], f'{seed}/scores/anomaly_scores-0.csv'), header=None)
        scores = scores.to_numpy().mean(axis=1)
        all_scores.append(scores)
    all_scores = np.array(all_scores)
    mean_scores = all_scores.mean(axis=0)

    # 计算AUC-ROC和AUC-PR
    mean_aucroc = np.mean(AUCROC_seed, axis=0)[:-1] * 100
    std_aucroc = np.std(AUCROC_seed, axis=0)[:-1] * 100
    mean_aucpr = np.mean(AUCPR_seed, axis=0)[:-1] * 100
    std_aucpr = np.std(AUCPR_seed, axis=0)[:-1] * 100

    # 输出为csv文件
    np.save(os.path.join(args['path_results'], 'scores.npy'), mean_scores)

    result = pd.DataFrame(
        {'mean_AUCROC': mean_aucroc, 'std_AUCROC': std_aucroc, 'mean_AUCPR': mean_aucpr, 'std_AUCPR': std_aucpr})

    # 将均值添加为DataFrame的新行
    mean_row = pd.DataFrame({'mean_AUCROC': [np.mean(mean_aucroc)], 'std_AUCROC': [np.mean(std_aucroc)],
                             'mean_AUCPR': [np.mean(mean_aucpr)], 'std_AUCPR': [np.mean(std_aucpr)]}, index=['Mean'])
    result = pd.concat([result, mean_row])

    result.to_csv(os.path.join(args['path_results'], 'results.csv'), index=True)
