import logging

from tqdm import tqdm

from adbench_modified.run import RunPipeline
from adbench_modified.baseline.DeepSAD.src.run import DeepSAD

import os
import numpy as np
import time
import glob

from scripts.group_results_SMD_dataset import group_results

# logging.basicConfig(level=logging.INFO)

# 设置项目路径
path_project = '/home/yukina/Missile_Fault_Detection/project'

seed = 3
n_samples_threshold = 0

data_path = os.path.join(path_project, 'data/SMD/yukina_data/DeepSAD_data, window=100, step=10')
timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')

# 遍历所有数据集文件
for dataset_path in tqdm(os.listdir(data_path), desc='Total progress'):
    base_name = os.path.basename(dataset_path).replace('.npz', '')
    # 创建结果文件夹路径

    data = np.load(os.path.join(data_path, dataset_path))
    dataset = {'X': data['X'], 'y': data['y'], 'X_train': data['X_train'], 'y_train': data['y_train'],
               'X_test': data['X_test'], 'y_test': data['y_test']}

    path_save = os.path.join(path_project, f'SMD_dataset/log/SMD/DeepSAD', 'DeepSAD_simple_dense, n_epoch=10, lr=0.005, window=100, step=10',
                             base_name)
    os.makedirs(path_save, exist_ok=True)  # 创建结果文件夹

    # 实例化并运行pipeline
    pipeline = RunPipeline(suffix='DeepSAD', parallel='unsupervise', n_samples_threshold=n_samples_threshold, seed=seed,
                           realistic_synthetic_mode=None,
                           noise_type=None, path_result=path_save)
    results = pipeline.run(clf=DeepSAD, dataset=dataset)

print("All down!")

group_results(base_dir=os.path.join(path_project, 'SMD_dataset/log/SMD/DeepSAD', 'DeepSAD_simple_dense, n_epoch=10, lr=0.005, window=100, step=10'))