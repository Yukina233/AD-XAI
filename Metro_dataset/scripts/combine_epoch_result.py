import os
import math
import cmath
import numpy as np
import pandas as pd

path_project = '/home/yukina/Missile_Fault_Detection/project'

def combine_epoch_results(base_dir):
    # 创建一个空的DataFrame来保存所有结果
    all_results = pd.DataFrame()

    # 遍历根目录下的所有文件夹
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            # 构建CSV文件的路径
            csv_path = os.path.join(folder_path, 'FDR_FAR_AUCROC_AUCPR_ensemble-DeepSAD_grouped.csv')
            if os.path.exists(csv_path):
                # 读取CSV文件
                df = pd.read_csv(csv_path)
                # 只选择class列为'mean'的行
                mean_row = df[df['class'] == 'mean']
                if not mean_row.empty:
                    # 添加文件夹编号信息
                    mean_row['folder'] = int(folder_name)
                    # 将结果添加到总的DataFrame中
                    all_results = pd.concat([all_results, mean_row], ignore_index=True)

    # 按照文件夹编号排序
    all_results = all_results.sort_values(by='folder')

        # 将所有结果保存到一个新的CSV文件中
    output_csv_path = os.path.join(base_dir, 'all_results.csv')
    all_results.to_csv(output_csv_path, index=False)

    print(f"所有结果已保存到 {output_csv_path}")

if __name__ == '__main__':
    # 定义根目录
    base_dir = os.path.join(path_project, 'GAN1_continue, window=1, step=1, no_tau2_K=7,deepsad_epoch=1,gan_epoch=0,lam1=0,lam2=0,lam3=0,latent_dim=5,lr=0.002')
    # 调用函数
    combine_epoch_results(base_dir)
