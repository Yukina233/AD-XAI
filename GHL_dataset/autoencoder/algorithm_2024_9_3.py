import argparse
import json
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from model import AutoEn
from dataclasses import dataclass, asdict


@dataclass
class CustomParameters:
    latent_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.005
    early_stopping_delta: float = 1e-2
    early_stopping_patience: int = 10
    split: float = 0.8
    random_state: int = 42


class AlgorithmArgs(argparse.Namespace):
    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random, tensorflow
    random.seed(seed)
    np.random.seed(seed)
    tensorflow.random.set_seed(seed)
    # tensorflow.set_random_seed(seed)


def load_data(args):
    df = pd.read_csv(args.dataInput)
    data = df.iloc[:, 1:-1].values
    labels = df.iloc[:, -1].values
    return data, labels

def train(args):
    xtr, ytr = load_data(args)
    ii = (ytr == 0)
    not_anomaly_data = xtr[ii]
    params = asdict(args.customParameters)
    del params["random_state"]
    model = AutoEn(**params)
    model.fit(not_anomaly_data, args.modelOutput)
    model.save(args.modelOutput)


def pred(args):
    xte, _ = load_data(args)
    model = AutoEn.load(args.modelInput)
    pred = model.predict(xte)
    pred = np.mean(np.abs(pred - xte), axis=1)
    np.savetxt(args.dataOutput, pred, delimiter= ",")

path_project = '/home/yukina/Missile_Fault_Detection/project'


def run(seed=0, suffix='window=100, step=10', dataset_name='GHL', epoch=20, path_results=''):

    input_path = os.path.join(path_project, f"data/{dataset_name}/csv/{suffix}")

    train_files = glob(os.path.join(input_path, '*train.csv'))
    test_files = glob(os.path.join(input_path, '*test.csv'))

    os.makedirs(os.path.join(path_results, f'{seed}/models'), exist_ok=True)
    os.makedirs(os.path.join(path_results, f'{seed}/scores'), exist_ok=True)
    for id, file in enumerate(train_files):
        print(f"Train file {id}: {file}")
        # 模拟传递参数
        mock_args = {
            "executionType": "train",  # 或 "execute"
            "dataInput": os.path.join(path_project, file),
            "modelOutput": os.path.join(path_results, f"{seed}/models/model.h5"),
            "modelInput": os.path.join(path_results, f"{seed}/models/model.h5"),  # 仅在 execute 时需要
            "dataOutput": os.path.join(path_results, f"{seed}/scores/anomaly_scores-{id}.csv"),  # 仅在 execute 时需要
            "customParameters": {'random_state': seed, 'epochs': epoch}
        }
        sys.argv = [sys.argv[0], json.dumps(mock_args)]

        args = AlgorithmArgs.from_sys_args()
        set_random_state(args)

        if args.executionType == "train":
            train(args)
        elif args.executionType == "execute":
            pred(args)
        else:
            raise ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")

    auc_roc_list = []
    auc_pr_list = []
    for id, file in enumerate(test_files):
        print(f"Test file {id}: {file}")
        mock_args = {
            "executionType": "execute",  # 或 "execute"
            "dataInput": os.path.join(path_project, file),
            "modelOutput": os.path.join(path_results, f"{seed}/models/model.h5"),
            "modelInput": os.path.join(path_results, f"{seed}/models/model.h5"),  # 仅在 execute 时需要
            "dataOutput": os.path.join(path_results, f"{seed}/scores/anomaly_scores-{id}.csv"),
            # 仅在 execute 时需要
            "customParameters": {}
        }
        sys.argv = [sys.argv[0], json.dumps(mock_args)]

        args = AlgorithmArgs.from_sys_args()
        set_random_state(args)

        if args.executionType == "train":
            train(args)
        elif args.executionType == "execute":
            pred(args)
        else:
            raise ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")

        _, labels = load_data(args)

        scores = np.loadtxt(args.dataOutput, delimiter=",")

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
    results_csv_path = os.path.join(path_results, f'{seed}/results.csv')
    result.to_csv(results_csv_path, index=True)  # index=True保留索引，这样'Mean'也会被写入文件

    print('Results saved.')

def run_1_by_1(seed=0, suffix='window=100, step=10', dataset_name='SMD', epoch=20, path_results=''):
    input_path = os.path.join(path_project, f"data/{dataset_name}/csv/{suffix}")

    train_files = glob(os.path.join(input_path, '*train.csv'))
    test_files = glob(os.path.join(input_path, '*test.csv'))

    os.makedirs(os.path.join(path_results, f'{seed}/models'), exist_ok=True)
    os.makedirs(os.path.join(path_results, f'{seed}/scores'), exist_ok=True)

    auc_roc_list = []
    auc_pr_list = []

    for id, file in enumerate(train_files):
        print(f"Train file {id}: {file}")
        mock_args = {
            "executionType": "train",  # 或 "execute"
            "dataInput": os.path.join(path_project, file),
            "modelOutput": os.path.join(path_results, f"{seed}/models/model.h5"),
            "modelInput": os.path.join(path_results, f"{seed}/models/model.h5"),  # 仅在 execute 时需要
            "dataOutput": os.path.join(path_results, f"{seed}/scores/anomaly_scores-{id}.csv"),  # 仅在 execute 时需要
            "customParameters": {'random_state': seed, 'epochs': epoch}
        }
        sys.argv = [sys.argv[0], json.dumps(mock_args)]

        args = AlgorithmArgs.from_sys_args()
        set_random_state(args)

        if args.executionType == "train":
            train(args)
        elif args.executionType == "execute":
            pred(args)
        else:
            raise ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")

        file = test_files[id]
        print(f"Test file {id}: {file}")
        mock_args = {
            "executionType": "execute",  # 或 "execute"
            "dataInput": os.path.join(path_project, file),
            "modelOutput": os.path.join(path_results, f"{seed}/models/model.h5"),
            "modelInput": os.path.join(path_results, f"{seed}/models/model.h5"),  # 仅在 execute 时需要
            "dataOutput": os.path.join(path_results, f"{seed}/scores/anomaly_scores-{id}.csv"),
            # 仅在 execute 时需要
            "customParameters": {}
        }
        sys.argv = [sys.argv[0], json.dumps(mock_args)]

        args = AlgorithmArgs.from_sys_args()
        set_random_state(args)

        if args.executionType == "train":
            train(args)
        elif args.executionType == "execute":
            pred(args)
        else:
            raise ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")

        _, labels = load_data(args)

        scores = np.loadtxt(args.dataOutput, delimiter=",")

        # 计算AUC-ROC和AUC-PR
        auc_roc = roc_auc_score(labels, scores)
        auc_pr = average_precision_score(labels, scores)

        auc_roc_list.append(auc_roc)
        auc_pr_list.append(auc_pr)
        import gc
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
    results_csv_path = os.path.join(path_results, f'{seed}/results.csv')
    result.to_csv(results_csv_path, index=True)  # index=True保留索引，这样'Mean'也会被写入文件

    print('Results saved.')

if __name__ == '__main__':
    suffix = 'window=100, step=10'
    dataset_name = 'SMD'
    epoch = 10
    path_results = os.path.join(path_project, f'{dataset_name}_dataset/autoencoder/results/{suffix}, epoch={epoch}')

    all_scores = []
    AUCROC_seed = []
    AUCPR_seed = []
    for seed in range(3):
        if dataset_name == 'SMD':
            run_1_by_1(seed, suffix, dataset_name, epoch, path_results)
        else:
            run(seed, suffix, dataset_name, epoch, path_results)
        result = pd.read_csv(os.path.join(path_results, f'{seed}/results.csv'), index_col=0)
        AUCROC_seed.append(result['AUC-ROC'].values)
        AUCPR_seed.append(result['AUC-PR'].values)

        scores = pd.read_csv(os.path.join(path_results, f'{seed}/scores/anomaly_scores-0.csv'), header=None)
        scores = scores.to_numpy().mean(axis=1)
        all_scores.append(scores)
    all_scores = np.array(all_scores)
    mean_scores = all_scores.mean(axis=0)

    # 计算AUC-ROC和AUC-PR
    mean_aucroc = np.mean(AUCROC_seed, axis=0)[:-1]
    std_aucroc = np.std(AUCROC_seed, axis=0)[:-1]
    mean_aucpr = np.mean(AUCPR_seed, axis=0)[:-1]
    std_aucpr = np.std(AUCPR_seed, axis=0)[:-1]

    # 输出为csv文件
    np.save(os.path.join(path_results, 'scores.npy'), mean_scores)

    result = pd.DataFrame({'mean_AUCROC': mean_aucroc, 'std_AUCROC':std_aucroc, 'mean_AUCPR': mean_aucpr, 'std_AUCPR':std_aucpr})

    # 将均值添加为DataFrame的新行
    mean_row = pd.DataFrame({'mean_AUCROC': [np.mean(mean_aucroc)], 'std_AUCROC': [np.mean(std_aucroc)], 'mean_AUCPR':[np.mean(mean_aucpr)], 'std_AUCPR':[np.mean(std_aucpr)]}, index=['Mean'])
    result = pd.concat([result, mean_row])

    result.to_csv(os.path.join(path_results, 'results.csv'), index=True)


