import os
import math
import cmath
import numpy as np
import pandas as pd

path_project = '/home/yukina/Missile_Fault_Detection/project'

def combine_epoch_results(base_dir, prefix):
    # 初始化一个空的列表来存储所有结果
    all_results = []

    # 遍历所有实验文件夹
    for seed_folder in os.listdir(base_dir):
        if not seed_folder.startswith(prefix):
            continue
        seed_path = os.path.join(base_dir, seed_folder)
        if os.path.isdir(seed_path):
            results_file = os.path.join(seed_path, 'all_results.csv')
            if os.path.exists(results_file):
                # 读取当前实验的结果
                df = pd.read_csv(results_file)
                # 移除非数值列
                df_numeric = df.select_dtypes(include=[np.number])
                # 将当前实验结果添加到汇总列表中
                all_results.append(df_numeric)

    # 将所有结果表格堆叠到一个三维数组中
    results_array = np.array([df.values for df in all_results])

    # 计算每个元素的均值和方差
    mean_results = np.mean(results_array, axis=0)
    std_results = np.std(results_array, axis=0)

    # 将结果转换为DataFrame
    mean_results_df = pd.DataFrame(mean_results, columns=all_results[0].columns)
    std_results_df = pd.DataFrame(std_results, columns=all_results[0].columns)

    # 保存结果到CSV文件
    save_dir = os.path.join(base_dir, 'seed_group', prefix)
    os.makedirs(save_dir, exist_ok=True)
    mean_results_df.to_csv(os.path.join(save_dir, 'mean_results.csv'), index=False)
    std_results_df.to_csv(os.path.join(save_dir, 'std_results.csv'), index=False)

    print("Mean and standard deviation results have been saved to CSV files.")

if __name__ == '__main__':
    # 定义根目录
    dataset_name = 'WQ'
    base_dir = os.path.join(path_project, f'{dataset_name}_dataset/log/{dataset_name}/ensemble/DeepSAD')
    prefix = 'GAN1_continue, window=1, step=1, no_tau2_K=7,deepsad_epoch=1,gan_epoch=1,lam1=10000,lam2=0,lam3=0,latent_dim=9,lr=0.002'
    # 调用函数
    combine_epoch_results(base_dir, prefix)
