import os
import math
import cmath
import numpy as np
import pandas as pd

path_project = '/home/yukina/Missile_Fault_Detection/project'

#  单独输出每个类的结果
# base_dir = os.path.join(path_project, 'auxiliary_data_AD/log', 'DeepSAD_origin')
# result_names = os.listdir(base_dir)
# results_paths = [os.path.join(base_dir, result_name) for result_name in result_names]
# seed_num = 3

# for result_path in results_paths:
#     csv_path = os.path.join(result_path, 'AUCROC_DeepSAD_type(None)_noise(None)_unsupervise.csv')
#     df = pd.read_csv(csv_path)
#     params_output = []
#     aucroc_output = []
#     count = 0
#     values = []
#     for index, row in df.iterrows():
#         params = row['Unnamed: 0']
#         aucroc = row['Customized']
#         if math.isnan(aucroc):
#             continue
#         dataset, ratio, seed = params.replace('(', '').replace(')', '').replace(' ', '').split(',')
#         count += 1
#         values.append(aucroc)
#         if count == seed_num:
#             params_output.append((dataset, ratio, 'avg'))
#             aucroc_output.append(np.mean(values))
#             count = 0
#             values = []
#
#     df_output = pd.DataFrame(data={'params': params_output, 'aucroc': aucroc_output})
#     df_output.to_csv(os.path.join(result_path, 'AUCROC_grouped.csv'), index=False)

# 合并输出每个类的结果

def group_results(base_dir):
    result_names = os.listdir(base_dir)

    suffix = 'DeepSAD'
    noise_type = None
    group_seed_num = 2
    classes_output = []
    aucroc_output = []
    for result_name in result_names:
        if 'MVTec-AD' not in result_name:
            continue
        result_path = os.path.join(base_dir, result_name)
        csv_path = os.path.join(result_path, f'AUCROC_{suffix}_type(None)_noise({noise_type})_unsupervise.csv')
        df = pd.read_csv(csv_path)
        count = 0
        values = []
        for index, row in df.iterrows():
            aucroc = row['Customized']
            if math.isnan(aucroc):
                continue
            count += 1
            values.append(aucroc)
            if count == group_seed_num:
                classes_output.append(result_name)
                aucroc_output.append(np.mean(values))
                count = 0
                values = []
    classes_output.append('mean')
    aucroc_output.append(np.mean(aucroc_output))

    df_output = pd.DataFrame(data={'class': classes_output, 'aucroc': aucroc_output})
    df_output.to_csv(os.path.join(base_dir, f'AUCROC_{suffix}_grouped.csv'), index=False)

    classes_output = []
    aucpr_output = []
    for result_name in result_names:
        if 'MVTec-AD' not in result_name:
            continue
        result_path = os.path.join(base_dir, result_name)
        csv_path = os.path.join(result_path, f'AUCPR_{suffix}_type(None)_noise({noise_type})_unsupervise.csv')
        df = pd.read_csv(csv_path)
        count = 0
        values = []
        for index, row in df.iterrows():
            aucpr = row['Customized']
            if math.isnan(aucpr):
                continue
            count += 1
            values.append(aucpr)
            if count == group_seed_num:
                classes_output.append(result_name)
                aucpr_output.append(np.mean(values))
                count = 0
                values = []

    classes_output.append('mean')
    aucpr_output.append(np.mean(aucpr_output))


    df_output = pd.DataFrame(data={'class': classes_output, 'aucpr': aucpr_output})
    df_output.to_csv(os.path.join(base_dir, f'AUCRPR_{suffix}_grouped.csv'), index=False)

    print("Group results finished!")

layers = ['layer3']
for layer in layers:
    base_dir = os.path.join(path_project, 'auxiliary_data_AD/log/n_samples_threshold=0,imgsize=224', f'DeepSAD_mixup,lamda=0.5,aux_size=1_resnet50_{layer}')
    group_results(base_dir)