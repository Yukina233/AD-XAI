import gc
import os
import numpy as np
import pandas as pd

from GHL_dataset.scripts.combine_epoch_result import combine_epoch_results
from SMD_dataset.scripts.auto_select_epoch import auto_select_epoch

path_project = '/home/yukina/Missile_Fault_Detection/project'


def combine_seed_results(base_dir, prefix):
    # 初始化一个空的列表来存储所有结果

    for iteration in range(0, 49):
        all_results = []
        # 遍历所有实验文件夹
        for seed_folder in os.listdir(base_dir):
            if not seed_folder.startswith(prefix):
                continue
            seed_path = os.path.join(base_dir, seed_folder, f'{iteration}')
            if os.path.isdir(seed_path):
                results_file = os.path.join(seed_path, 'FDR_FAR_AUCROC_AUCPR_ensemble-DeepSAD_grouped.csv')
                if os.path.exists(results_file):
                    # 读取当前实验的结果
                    df = pd.read_csv(results_file)
                    df.drop(df.index[-1], inplace=True)
                    # 存储非数值列
                    df_non_numeric = df.select_dtypes(exclude=[np.number])
                    # 移除非数值列
                    df_numeric = df.select_dtypes(include=[np.number])
                    # 将当前实验结果添加到汇总列表中
                    all_results.append((df_numeric, df_non_numeric))

        if not all_results:
            continue

        # 将所有结果表格堆叠到一个三维数组中
        numeric_results_array = np.array([df[0].values for df in all_results])

        # 计算每个元素的均值和方差
        mean_results = np.mean(numeric_results_array, axis=0)
        std_results = np.std(numeric_results_array, axis=0)

        # 将结果转换为DataFrame
        mean_results_df = pd.DataFrame(mean_results, columns=all_results[0][0].columns)
        std_results_df = pd.DataFrame(std_results, columns=all_results[0][0].columns)

        # 重新添加非数值列
        non_numeric_df = all_results[0][1]
        mean_results_df = pd.concat([non_numeric_df.reset_index(drop=True), mean_results_df], axis=1)
        std_results_df = pd.concat([non_numeric_df.reset_index(drop=True), std_results_df], axis=1)

        # 重新计算合并后的数据帧的均值行
        mean_row = mean_results_df.mean(numeric_only=True)
        std_row = std_results_df.mean(numeric_only=True)

        # 创建新的 DataFrame 来存储均值行
        mean_row_df = pd.DataFrame(mean_row).T
        std_row_df = pd.DataFrame(std_row).T

        # 将非数值列的值设置为 'mean'
        for col in non_numeric_df.columns:
            mean_row_df[col] = 'mean'
            std_row_df[col] = 'mean'

        # 将均值行添加到结果数据帧的最后一行
        mean_results_df = pd.concat([mean_results_df, mean_row_df], ignore_index=True)
        std_results_df = pd.concat([std_results_df, std_row_df], ignore_index=True)

        # 保存结果到CSV文件
        save_dir = os.path.join(base_dir, 'seed_group', prefix, f'{iteration}')
        os.makedirs(save_dir, exist_ok=True)
        mean_results_df.to_csv(os.path.join(save_dir, 'mean_results.csv'), index=False)
        std_results_df.to_csv(os.path.join(save_dir, 'std_results.csv'), index=False)

        print(f"Iteration {iteration}: mean and standard deviation results have been saved to CSV files.")

        del all_results
        gc.collect()


if __name__ == '__main__':
    test_set_name = 'SMD'
    # 定义根目录
    base_dir = os.path.join(path_project, f'{test_set_name}_dataset/log/{test_set_name}/ensemble/DeepSAD')
    prefix = 'WGAN-GP, euc, window=20, step=1, no_tau2_K=7,deepsad_epoch=1,gan_epoch=1,lam1=10000,lam2=100,latent_dim=180,lr=0.0002,clip_value=0.01,lambda_gp=1000,seed'
    # 调用函数
    combine_seed_results(base_dir, prefix)

    seed_dir = os.path.join(base_dir, 'seed_group', prefix)
    combine_epoch_results(seed_dir)
    auto_select_epoch(seed_dir)
