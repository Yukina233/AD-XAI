import csv
import os

import pandas as pd

path_project = '/home/yukina/Missile_Fault_Detection/project'

base_dir = os.path.join(path_project,
                            f'adversarial_ensemble_AD/log/banwuli_data/train_result/no_tau2_K=7,deepsad_epoch=20,gan_epoch=50,lam1=1,lam2=100,tau1=0.1/log_seperate_model')

for iteration in os.listdir(base_dir):
    iteration_path = os.path.join(base_dir, iteration)
    scores = []

    if not os.path.isdir(iteration_path):
        continue
    for model in os.listdir(iteration_path):
        # 从csv文件中加载字典对象
        seperate_model_path = os.path.join(iteration_path, model)
        data = pd.read_csv(os.path.join(seperate_model_path, 'FDR_FAR_AUCROC_AUCPR_ensemble-DeepSAD_grouped.csv'))
        AUCPR = data['AUCPR']
        scores.append(AUCPR.get(AUCPR.size-1))

    print(f'iteration: {iteration}, AUCPR: {scores}')

# 定义要写入的CSV文件路径
output_path = os.path.join(base_dir, 'AUCPR_seperate_model.csv')

# 创建一个空的DataFrame来存储数据
result_df = pd.DataFrame()

# 遍历数据并将其添加到DataFrame中
for iteration in os.listdir(base_dir):
    iteration_path = os.path.join(base_dir, iteration)
    iteration_scores = {}

    for model in os.listdir(iteration_path):
        separate_model_path = os.path.join(iteration_path, model)
        data = pd.read_csv(os.path.join(separate_model_path, 'FDR_FAR_AUCROC_AUCPR_ensemble-DeepSAD_grouped.csv'))
        AUCPR = data['AUCPR']
        iteration_scores[model] = AUCPR.get(AUCPR.size-1)

    result_df = result_df.append(iteration_scores, ignore_index=True)

# 将DataFrame的列排序
result_df = result_df.reindex(sorted(result_df.columns), axis=1)

# 将DataFrame写入CSV文件
result_df.to_csv(output_path, index=False)