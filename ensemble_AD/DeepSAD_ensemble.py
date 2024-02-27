import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from adbench_modified.baseline.DeepSAD.src.datasets import load_dataset
from adbench_modified.baseline.DeepSAD.src.deepsad import deepsad

path_project = '/home/yukina/Missile_Fault_Detection/project'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_jobs_dataloader = 0


# 加载所有模型
models_path = os.path.join(path_project, 'ensemble_AD/log/ensemble_id', f'DeepSAD_origin_seed=1')
output_path = os.path.join(models_path, 'ensemble_results_mean-std')
os.makedirs(output_path, exist_ok=True)
task_dirs = glob.glob(models_path + '/iot*')

model_list = []
for task_dir in task_dirs:
    task_path = os.path.join(models_path, task_dir)
    ae_net = torch.load(os.path.join(task_path, 'model_ae_net.pth'))
    net = torch.load(os.path.join(task_path, 'model_net.pth'))
    c = np.load(os.path.join(task_path, 'c.npy')).tolist()

    model = deepsad(1.0, None)
    model.net = net
    model.ae_net = ae_net
    model.c = c

    model_list.append(model)

# 加载所有数据集
data_path = path_project + f'/data/iot_data_with_id'
dataset_files = os.listdir(data_path)


dataset_list = []
for dataset_path in dataset_files:
    data = np.load(os.path.join(data_path, dataset_path), allow_pickle=True)
    dataset = load_dataset(data={'X_test': data['X_test'], 'y_test': data['y_test']}, train=False)
    dataset_list.append(dataset)

# 结果收集
columns = ['Customized']
df_AUCROC = pd.DataFrame(data=None, index=[f'iot_{i}' for i in range(len(task_dirs))], columns=columns)
df_AUCPR = pd.DataFrame(data=None, index=[i for i in range(len(task_dirs))], columns=columns)

for id in tqdm(range(len(dataset_list)), desc='Testing'):
    test_data = dataset_list[id]
    models_use = [model_list[i] for i in range(len(model_list)) if i != id]

    scores_list = []
    for model in models_use:
        scores_list.append(model.test(test_data, device=device, n_jobs_dataloader=n_jobs_dataloader))
    scores = np.array(scores_list)
    # metric = np.mean(scores, axis=0)
    metric = np.mean(scores, axis=0) + np.std(scores, axis=0)

    aucroc = roc_auc_score(y_true=test_data.test_set.targets, y_score=metric)
    aucpr = average_precision_score(y_true=test_data.test_set.targets, y_score=metric, pos_label=1)

    df_AUCROC[columns[0]].iloc[id] = aucroc
    df_AUCPR[columns[0]].iloc[id] = aucpr

    df_AUCROC.to_csv(os.path.join(output_path, 'AUCROC.csv'), index=True)
    df_AUCPR.to_csv(os.path.join(output_path, 'AUCPR.csv'), index=True)

    results = {'metric': metric, 'label': test_data.test_set.targets}

    df_results = pd.DataFrame(data=results)
    df_scores = pd.DataFrame(data=scores)
    os.makedirs(os.path.join(output_path, f'iot_{id}'))
    df_results.to_csv(os.path.join(output_path, f'iot_{id}', f'results.csv'), index=False)
    df_scores.to_csv(os.path.join(output_path, f'iot_{id}', f'scores.csv'), index=False)

    print(f'{dataset_files[id]}: aucroc={aucroc}, aucpr={aucpr}')

print("All down!")