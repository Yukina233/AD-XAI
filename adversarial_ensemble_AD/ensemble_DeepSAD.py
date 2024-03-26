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

seed = 3
n_samples_threshold = 0

data_path = os.path.join(path_project, 'data/banwuli_data/yukina_data/DeepSAD_data')
timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')

# 遍历所有数据集文件
for fault in tqdm(os.listdir(data_path), desc='Total progress'):
    base_name = os.path.basename(fault).replace('.npz', '')
    # 创建结果文件夹路径


    # 加载数据集
    FDRs = []
    FARs = []
    path_fault = os.path.join(data_path, fault)
    files = os.listdir(path_fault)

    for i in range(1, int(len(files) + 1)):
        data = np.load(os.path.join(path_fault, f"dataset_{i}.npz"))
        dataset = {'X': data['X'], 'y': data['y'], 'X_train': data['X_train'], 'y_train': data['y_train'],
                'X_test': data['X_test'], 'y_test': data['y_test']}

        path_save = os.path.join(path_project, f'adversarial_ensemble_AD/log/ensemble', f'DeepSAD',
                                 base_name, f'{i}')
        os.makedirs(path_save, exist_ok=True)  # 创建结果文件夹

        # 实例化并运行pipeline
        pipeline = RunPipeline(suffix='DeepSAD', parallel='unsupervise', n_samples_threshold=n_samples_threshold, seed=seed,
                           realistic_synthetic_mode=None,
                           noise_type=None, path_result=path_save)
        results = pipeline.run(clf=DeepSAD, dataset=dataset)

print("All down!")
