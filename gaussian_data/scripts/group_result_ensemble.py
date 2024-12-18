import os
import math
import cmath
import numpy as np
import pandas as pd

path_project = '/home/yukina/Missile_Fault_Detection/project_data'


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
    FDR_at_threshold_output = []
    FAR_at_threshold_output = []
    for fault in fault_list:
        if '.csv' in fault:
            continue
        # FDRs = []
        # FARs = []
        AUCROCs = []
        AUCPRs = []
        FDR_at_thresholds = []
        FAR_at_thresholds = []
        path_fault = os.path.join(base_dir, fault)

        result_path = os.path.join(path_fault, f'result.csv')
        result = pd.read_csv(result_path)
        metrics = result['指标']
        scores = result['分数']
        df_result = {}
        for i in range(0, metrics.__len__()):
            df_result[metrics[i]] = scores[i]

        # FDRs.append(df_result['FDR'])
        # FARs.append(df_result['FAR'])
        AUCROCs.append(df_result['aucroc'])
        AUCPRs.append(df_result['aucpr'])
        FDR_at_thresholds.append(df_result['FDR_at_threshold'])
        FAR_at_thresholds.append(df_result['FAR_at_threshold'])

        classes_output.append(fault)
        # FDR_output.append(np.mean(FDRs) * 100)
        # FAR_output.append(np.mean(FARs) * 100)
        AUCROC_output.append(np.mean(AUCROCs) * 100)
        AUCPR_output.append(np.mean(AUCPRs) * 100)
        FDR_at_threshold_output.append(np.mean(FDR_at_thresholds) * 100)
        FAR_at_threshold_output.append(np.mean(FAR_at_thresholds) * 100)

    classes_output.append('mean')
    # FDR_output.append(np.mean(FDR_output))
    # FAR_output.append(np.mean(FAR_output))
    AUCROC_output.append(np.mean(AUCROC_output))
    AUCPR_output.append(np.mean(AUCPR_output))
    FDR_at_threshold_output.append(np.mean(FDR_at_threshold_output))
    FAR_at_threshold_output.append(np.mean(FAR_at_threshold_output))

    df_output = pd.DataFrame(
        data={'class': classes_output, 'AUCROC': AUCROC_output,
              'AUCPR': AUCPR_output, 'FDR_at_threshold': FDR_at_threshold_output,
              'FAR_at_threshold': FAR_at_threshold_output})
    df_output.to_csv(os.path.join(base_dir, f'FDR_FAR_AUCROC_AUCPR_ensemble-{suffix}_grouped.csv'), index=False)

    print("Group results finished!")


# log_path = os.path.join(path_project, 'adversarial_ensemble_AD/log/ensemble/DeepSAD/correct')
#
# for experiment in os.listdir(log_path):
#     base_dir = os.path.join(log_path, f'{experiment}/4')
#     group_results(base_dir)
if __name__ == '__main__':

    base_dir = os.path.join(path_project,
                            f'GHL_dataset/log/GHL/ensemble/DeepSAD/std, window=100, step=10, no_tau2_K=7,deepsad_epoch=20,gan_epoch=50,lam1=1,lam2=1,tau1=1/4')
    group_results(base_dir)
