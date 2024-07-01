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

def predict_threshold(args):
    # 计算阈值
    xtr, ytr = load_data(args)
    model = AutoEn.load(args.modelInput)
    pred = model.predict(xtr)
    score_train = np.mean(np.abs(pred - xtr), axis=1)
    thresholds = np.percentile(score_train, 95)
    return thresholds


path_project = '/home/yukina/Missile_Fault_Detection/project'

if __name__=="__main__":



    suffix = 'window=100, step=10'
    input_path = os.path.join(path_project, "data/GHL/csv/window=100, step=10")


    train_files = glob(os.path.join(input_path, '*train.csv'))
    test_files = glob(os.path.join(input_path, '*test.csv'))

    path_results = os.path.join(path_project, f'GHL_dataset/autoencoder/results/{suffix}')
    os.makedirs(os.path.join(path_results, 'models'), exist_ok=True)
    os.makedirs(os.path.join(path_results, 'scores'), exist_ok=True)
    for id, file in enumerate(train_files):
        print(f"Train file {id}: {file}")
        # 模拟传递参数
        mock_args = {
            "executionType": "train",  # 或 "execute"
            "dataInput": os.path.join(path_project, file),
            "modelOutput": os.path.join(path_results, f"models/model.h5"),
            "modelInput": os.path.join(path_results, f"models/model.h5"),  # 仅在 execute 时需要
            "dataOutput": os.path.join(path_results, f"scores/anomaly_scores-{id}.csv"),  # 仅在 execute 时需要
            "customParameters": {}
        }
        sys.argv = [sys.argv[0], json.dumps(mock_args)]

        args = AlgorithmArgs.from_sys_args()
        set_random_state(args)

        if args.executionType == "train":
            train(args)
            thresholds = predict_threshold(args)
        elif args.executionType == "execute":
            pred(args)
        else:
            raise ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")



    auc_roc_list = []
    auc_pr_list = []
    precision_list = []
    recall_list = []
    for id, file in enumerate(test_files):
        print(f"Test file {id}: {file}")
        mock_args = {
            "executionType": "execute",  # 或 "execute"
            "dataInput": os.path.join(path_project, file),
            "modelOutput": os.path.join(path_results, f"models/model.h5"),
            "modelInput": os.path.join(path_results, f"models/model.h5"),  # 仅在 execute 时需要
            "dataOutput": os.path.join(path_results, f"scores/anomaly_scores-{id}.csv"),
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

        id_anomaly_pred = np.where(scores > thresholds)[0]
        id_normal_pred = np.where(scores <= thresholds)[0]

        tp = np.size(np.where(labels[id_anomaly_pred] == 1)[0], 0)
        fp = np.size(np.where(labels[id_anomaly_pred] == 0)[0], 0)
        fn = np.size(np.where(labels[id_normal_pred] == 1)[0], 0)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        precision_list.append(precision)
        recall_list.append(recall)

        # 计算AUC-ROC和AUC-PR
        auc_roc = roc_auc_score(labels, scores)
        auc_pr = average_precision_score(labels, scores)

        auc_roc_list.append(auc_roc)
        auc_pr_list.append(auc_pr)


    # 创建包含AUC-ROC和AUC-PR值的DataFrame
    result = pd.DataFrame({'AUC-ROC': auc_roc_list, 'AUC-PR': auc_pr_list, 'Precision': precision_list, 'Recall': recall_list})

    # 计算均值
    mean_auc_roc = result['AUC-ROC'].mean()
    mean_auc_pr = result['AUC-PR'].mean()

    mean_precision = result['Precision'].mean()
    mean_recall = result['Recall'].mean()

    # 将均值添加为DataFrame的新行
    mean_row = pd.DataFrame({'AUC-ROC': [mean_auc_roc], 'AUC-PR': [mean_auc_pr], 'Precision': [mean_precision], 'Recall': [mean_recall]}, index=['Mean'])
    result = pd.concat([result, mean_row])

    # 保存结果到CSV文件
    results_csv_path = os.path.join(path_project, f'GHL_dataset/autoencoder/results/{suffix}/results.csv')
    result.to_csv(results_csv_path, index=True)  # index=True保留索引，这样'Mean'也会被写入文件

    print('Results saved.')


