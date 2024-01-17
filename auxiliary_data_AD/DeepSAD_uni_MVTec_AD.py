from tqdm import tqdm

from adbench_modified.run import RunPipeline
from adbench_modified.baseline.DeepSAD.src.run import DeepSAD

import os
import numpy as np
import time
import glob

# 设置项目路径
path_project = '/home/yukina/Missile_Fault_Detection/project'



seed = 2
aug_type = 'mixup'
lamda = 0
aux_size = 1
use_preprocess = True
noise_type = None
n_samples_threshold = 0
imagesize = 224
encoder_name = 'resnet50'
layers = ['layer3']
interpolate = False
# 获取所有以'MVTec_AD'开头的数据集文件路径
if layers == ['avgpool']:
    layers_name = ''
else:
    layers_name = layers.__str__().replace('[', '').replace(']', '').replace(' ', '').replace('\'', '')
# 获取所有以'MVTec_AD'开头的数据集文件路径
data_path = path_project + f'/data/mvtec_ad_imgsize={imagesize}'
dataset_files = glob.glob(data_path + '/MVTec-AD*.npz')
timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
# 遍历所有数据集文件
for dataset_path in tqdm(dataset_files, desc='Total progress'):
    # 获取数据集的基本名称，例如：MVTec_AD_transistor
    base_name = os.path.basename(dataset_path).replace('.npz', '')
    # 创建结果文件夹路径

    path_save = os.path.join(path_project, f'auxiliary_data_AD/log/n_samples_threshold={n_samples_threshold},imgsize={imagesize}',
                             f'DeepSAD_{aug_type},lamda={lamda},aux_size={aux_size}_{encoder_name}_{layers_name}',
                             base_name)
    os.makedirs(path_save, exist_ok=True)  # 创建结果文件夹

    # 实例化并运行pipeline
    pipeline = RunPipeline(suffix='DeepSAD', parallel='unsupervise', n_samples_threshold=n_samples_threshold, seed=seed,
                           realistic_synthetic_mode=None,
                           noise_type=noise_type, path_result=path_save)
    results = pipeline.run_universum(clf=DeepSAD, target_dataset_name=base_name,
                                     path_datasets=data_path,
                                     use_preprocess=use_preprocess,
                                     universum_params={'aug_type': aug_type, 'lamda': lamda, 'aux_size': aux_size},
                                     preprocess_params={'encoder_name': encoder_name, 'layers': layers, 'interpolate': interpolate})

print("All down!")
