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

# 遍历所有数据集文件
for dataset_path in dataset_files:
    # 获取数据集的基本名称，例如：MVTec_AD_transistor
    base_name = os.path.basename(dataset_path).replace('.npz', '')
    # 创建结果文件夹路径

    path_save = os.path.join(path_project, 'auxiliary_data_AD/log', timestamp, base_name)
    os.makedirs(path_save, exist_ok=True)  # 创建结果文件夹

    # 加载数据集
    data = np.load(dataset_path, allow_pickle=True)
    dataset = {'X': data['X'], 'y': data['y']}

    # 实例化并运行pipeline
    pipeline = RunPipeline(suffix='DeepSAD', parallel='unsupervise', n_samples_threshold=200,
                           realistic_synthetic_mode=None,
                           noise_type=None, path_result=path_save)
    results = pipeline.run(clf=DeepSAD, dataset=dataset)


print("All down!")