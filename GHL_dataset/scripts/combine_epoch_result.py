import os
import math
import cmath
import numpy as np
import pandas as pd

path_project = '/home/yukina/Missile_Fault_Detection/project_data'
# path_project = '/media/test/d/Yukina/AD-XAI'

def combine_epoch_results(base_dir):
    # 创建一个空的DataFrame来保存所有结果
    all_means = pd.DataFrame()
    all_stds = pd.DataFrame()

    # 遍历根目录下的所有文件夹
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            # 构建CSV文件的路径
            csv_path = os.path.join(folder_path, 'mean_results.csv')
            if os.path.exists(csv_path):
                # 读取CSV文件
                df = pd.read_csv(csv_path)
                # 只选择class列为'mean'的行
                mean_row = df[df['class'] == 'mean']
                if not mean_row.empty:
                    # 添加文件夹编号信息
                    mean_row['folder'] = int(folder_name)
                    # 将结果添加到总的DataFrame中
                    all_means = pd.concat([all_means, mean_row], ignore_index=True)

    # 按照文件夹编号排序
    all_means = all_means.sort_values(by='folder')

        # 将所有结果保存到一个新的CSV文件中
    output_csv_path = os.path.join(base_dir, 'all_means.csv')
    all_means.to_csv(output_csv_path, index=False)

    print(f"所有结果已保存到 {output_csv_path}")

    # 遍历根目录下的所有文件夹
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            # 构建CSV文件的路径
            csv_path = os.path.join(folder_path, 'std_results.csv')
            if os.path.exists(csv_path):
                # 读取CSV文件
                df = pd.read_csv(csv_path)
                # 只选择class列为'mean'的行
                mean_row = df[df['class'] == 'mean']
                if not mean_row.empty:
                    # 添加文件夹编号信息
                    mean_row['folder'] = int(folder_name)
                    # 将结果添加到总的DataFrame中
                    all_stds = pd.concat([all_stds, mean_row], ignore_index=True)

    # 按照文件夹编号排序
    all_stds = all_stds.sort_values(by='folder')

    # 将所有结果保存到一个新的CSV文件中
    output_csv_path = os.path.join(base_dir, 'all_stds.csv')
    all_stds.to_csv(output_csv_path, index=False)

    print(f"所有结果已保存到 {output_csv_path}")

    merge_result = pd.DataFrame(
        {'folder': all_means['folder'],
         'AUCROC_mean': all_means['AUCROC'], 'AUCROC_std': all_stds['AUCROC'],
         'AUCPR_mean': all_means['AUCPR'], 'AUCPR_std': all_stds['AUCPR']})
    output_csv_path = os.path.join(base_dir, 'merge_results.csv')
    merge_result.to_csv(os.path.join(output_csv_path), index=False)

    print(f"所有结果已保存到 {output_csv_path}")

if __name__ == '__main__':
    test_set_name = 'GHL'
    # 定义根目录
    base_dir = os.path.join(path_project, f'{test_set_name}_dataset/log/{test_set_name}/ensemble/DeepSAD/seed_group', 'GAN1_continue, euc, window=100, step=10, K=13,deepsad_epoch=1,gan_epoch=1,lam1=1,lam2=0,lam3=0,latent_dim=80,lr=0.002')
    # 调用函数
    combine_epoch_results(base_dir)
