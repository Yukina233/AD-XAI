from tqdm import tqdm

from adbench_modified.run import RunPipeline
from adbench_modified.baseline.DeepSAD.src.run import DeepSAD

import os
import numpy as np
import time
import glob


# 设置项目路径
path_project = '/home/yukina/Missile_Fault_Detection/project'

seed = 5
n_samples_threshold = 0
imagesize = 224
encoder_name = 'resnet50'
layers = ['layer1', 'layer2', 'layer3']
# 获取所有以'MVTec_AD'开头的数据集文件路径
if layers == ['avgpool']:
    layers_name = ''
else:
    layers_name = layers.__str__().replace('[', '').replace(']', '').replace(' ', '').replace('\'', '')
data_path = path_project + f'/data/mvtec_ad_preprocessed_{encoder_name}_{layers_name}_imgsize={imagesize}'
dataset_files = glob.glob(data_path + '/MVTec-AD*.npz')
timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')

# 遍历所有数据集文件
for dataset_path in tqdm(dataset_files):
    # 获取数据集的基本名称，例如：MVTec_AD_transistor
    base_name = os.path.basename(dataset_path).replace('.npz', '')
    # 创建结果文件夹路径

    path_save = os.path.join(path_project, f'auxiliary_data_AD/log/n_samples_threshold={n_samples_threshold},imgsize={imagesize}', f'DeepSAD_origin_{encoder_name}_{layers_name}', base_name)
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