import logging

from tqdm import tqdm

from adbench_modified.run import RunPipeline
from adbench_modified.baseline.DeepSAD.src.run import DeepSAD

import os
import numpy as np
import time
import glob

# logging.basicConfig(level=logging.INFO)

# 设置项目路径
path_project = '/home/yukina/Missile_Fault_Detection/project'

seed = 1
n_samples_threshold = 0


data_path = path_project + f'/data/iot_data_baseline'
dataset_files = glob.glob(data_path + '/iot_*.npz')
timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')

# 遍历所有数据集文件
for dataset_path in tqdm(dataset_files):
    base_name = os.path.basename(dataset_path).replace('.npz', '')
    # 创建结果文件夹路径

    path_save = os.path.join(path_project, f'ensemble_AD/log/baseline', f'DeepSAD_origin_seed=3', base_name)
    os.makedirs(path_save, exist_ok=True)  # 创建结果文件夹

    # 加载数据集
    data = np.load(dataset_path, allow_pickle=True)
    dataset = {'X': data['X'], 'y': data['y'], 'X_train': data['X_train'], 'y_train': data['y_train'],
               'X_test': data['X_test'], 'y_test': data['y_test']}

    # 实例化并运行pipeline
    pipeline = RunPipeline(suffix='DeepSAD', parallel='unsupervise', n_samples_threshold=n_samples_threshold, seed=seed,
                           realistic_synthetic_mode=None,
                           noise_type=None, path_result=path_save)
    results = pipeline.run(clf=DeepSAD, dataset=dataset)


print("All down!")