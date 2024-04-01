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
    fault_list = os.listdir(base_dir)

    suffix = 'DeepSAD'
    noise_type = None
    group_seed_num = 3
    classes_output = []
    FDR_output = []
    FAR_output = []
    AUCROC_output = []
    AUCPR_output = []
    for fault in fault_list:
        FDRs = []
        FARs = []
        AUCROCs = []
        AUCPRs = []
        path_fault = os.path.join(base_dir, fault)
        files = os.listdir(path_fault)

        for i in range(1, int(len(files) + 1)):
            result_path = os.path.join(path_fault, f'{i}.csv')
            result = pd.read_csv(result_path)
            metrics = result['指标']
            scores = result['分数']
            df_result = {}
            for i in range(0, metrics.__len__()):
                df_result[metrics[i]] = scores[i]

            FDRs.append(df_result['FDR'])
            FARs.append(df_result['FAR'])
            AUCROCs.append(df_result['aucroc'])
            AUCPRs.append(df_result['aucpr'])

        classes_output.append(fault)
        FDR_output.append(np.mean(FDRs))
        FAR_output.append(np.mean(FARs))
        AUCROC_output.append(np.mean(AUCROCs))
        AUCPR_output.append(np.mean(AUCPRs))

    classes_output.append('mean')
    FDR_output.append(np.mean(FDR_output))
    FAR_output.append(np.mean(FAR_output))
    AUCROC_output.append(np.mean(AUCROC_output))
    AUCPR_output.append(np.mean(AUCPR_output))

    df_output = pd.DataFrame(data={'class': classes_output, 'FDR': FDR_output, 'FAR': FAR_output, 'AUCROC': AUCROC_output, 'AUCPR': AUCPR_output})
    df_output.to_csv(os.path.join(base_dir, f'FDR_FAR_AUCROC_AUCPR_{suffix}_grouped.csv'), index=False)

    print("Group results finished!")


base_dir = os.path.join(path_project, 'adversarial_ensemble_AD/log/ensemble/DeepSAD/n=2/3')
group_results(base_dir)