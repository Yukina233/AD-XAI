from tqdm import tqdm

from adbench_modified.run import RunPipeline
from adbench_modified.baseline.DeepSAD.src.run import DeepSAD

import os
import numpy as np
import time
import glob

# 设置项目路径
path_project = '/home/yukina/Missile_Fault_Detection/project'

# 获取所有以'MVTec_AD'开头的数据集文件路径
dataset_files = glob.glob(os.path.join(path_project, 'data/mvtec_ad/MVTec-AD*.npz'))
timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')

seed = 5
aug_type = 'cutmix'
lamda = 0.8
aux_size = 1
use_preprocess = False
noise_type = None
n_samples_threshold = 0

# 遍历所有数据集文件
for dataset_path in tqdm(dataset_files):
    # 获取数据集的基本名称，例如：MVTec_AD_transistor
    base_name = os.path.basename(dataset_path).replace('.npz', '')
    # 创建结果文件夹路径

    path_save = os.path.join(path_project, f'auxiliary_data_AD/log/n_samples_threshold = {n_samples_threshold}',
                             f'DeepSAD_{aug_type},lamda={lamda},aux_size={aux_size}',
                             base_name)
    os.makedirs(path_save, exist_ok=True)  # 创建结果文件夹

    # 实例化并运行pipeline
    pipeline = RunPipeline(suffix='DeepSAD', parallel='unsupervise', n_samples_threshold=n_samples_threshold, seed=seed,
                           realistic_synthetic_mode=None,
                           noise_type=noise_type, path_result=path_save)
    results = pipeline.run_universum(clf=DeepSAD, target_dataset_name=base_name,
                                     path_datasets=path_project + '/data/mvtec_ad',
                                     use_preprocess=use_preprocess,
                                     universum_params={'aug_type': aug_type, 'lamda': lamda, 'aux_size': aux_size})

print("All down!")
