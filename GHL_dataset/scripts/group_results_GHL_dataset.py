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

    AUCROC_output_mean = []
    AUCPR_output_mean = []
    FDR_at_threshold_output_mean = []
    FAR_at_threshold_output_mean = []


    AUCROC_output_std = []
    AUCPR_output_std = []
    FDR_at_threshold_output_std = []
    FAR_at_threshold_output_std = []

    for fault in fault_list:
        if '.csv' in fault:
            continue

        AUCROCs = []
        AUCPRs = []
        FDR_at_thresholds = []
        FAR_at_thresholds = []
        path_fault = os.path.join(base_dir, fault)

        for seed in os.listdir(path_fault):
            result_path = os.path.join(path_fault, seed, f'result.csv')
            result = pd.read_csv(result_path)
            metrics = result['指标']
            scores = result['分数']
            df_result = {}
            for i in range(0, metrics.__len__()):
                df_result[metrics[i]] = scores[i]


            AUCROCs.append(df_result['aucroc'])
            AUCPRs.append(df_result['aucpr'])
            FDR_at_thresholds.append(df_result['FDR_at_threshold'])
            FAR_at_thresholds.append(df_result['FAR_at_threshold'])

        classes_output.append(fault)

        AUCROC_output_mean.append(np.mean(AUCROCs) * 100)
        AUCPR_output_mean.append(np.mean(AUCPRs) * 100)
        FDR_at_threshold_output_mean.append(np.mean(FDR_at_thresholds) * 100)
        FAR_at_threshold_output_mean.append(np.mean(FAR_at_thresholds) * 100)


        AUCROC_output_std.append(np.std(AUCROCs) * 100)
        AUCPR_output_std.append(np.std(AUCPRs) * 100)
        FDR_at_threshold_output_std.append(np.std(FDR_at_thresholds) * 100)
        FAR_at_threshold_output_std.append(np.std(FAR_at_thresholds) * 100)

    classes_output.append('mean')

    AUCROC_output_mean.append(np.mean(AUCROC_output_mean))
    AUCPR_output_mean.append(np.mean(AUCPR_output_mean))
    FDR_at_threshold_output_mean.append(np.mean(FDR_at_threshold_output_mean))
    FAR_at_threshold_output_mean.append(np.mean(FAR_at_threshold_output_mean))


    AUCROC_output_std.append(np.mean(AUCROC_output_std))
    AUCPR_output_std.append(np.mean(AUCPR_output_std))
    FDR_at_threshold_output_std.append(np.mean(FDR_at_threshold_output_std))
    FAR_at_threshold_output_std.append(np.mean(FAR_at_threshold_output_std))

    df_output_mean = pd.DataFrame(
        data={'class': classes_output,
              'AUCROC': AUCROC_output_mean,
              'AUCPR': AUCPR_output_mean,

              'FDR_at_threshold': FDR_at_threshold_output_mean,
              'FAR_at_threshold': FAR_at_threshold_output_mean})
    df_output_mean.to_csv(os.path.join(base_dir, f'all_means.csv'), index=False)

    df_output_std = pd.DataFrame(
        data={'class': classes_output,
              'AUCROC': AUCROC_output_std,
              'AUCPR': AUCPR_output_std,

              'FDR_at_threshold': FDR_at_threshold_output_std,
              'FAR_at_threshold': FAR_at_threshold_output_std})
    df_output_std.to_csv(os.path.join(base_dir, f'all_stds.csv'), index=False)

    print("Group results finished!")


# log_path = os.path.join(path_project, 'adversarial_ensemble_AD/log/ensemble/DeepSAD/correct')
#
# for experiment in os.listdir(log_path):
#     base_dir = os.path.join(log_path, f'{experiment}/4')
#     group_results(base_dir)
if __name__ == '__main__':

    base_dir = os.path.join(path_project,
                            f'GHL_dataset/log/GHL/DeepSAD/fix_pretrain, net=Dense, std, window=100, step=10, n_epochs=5, ae_n_epochs=20, lr=0.001, ae_lr=0.001')
    group_results(base_dir)



# import os
# import math
# import cmath
# import numpy as np
# import pandas as pd
#
# path_project = '/home/yukina/Missile_Fault_Detection/project'
#
#
# #  单独输出每个类的结果
# # base_dir = os.path.join(path_project, 'auxiliary_data_AD/log', 'DeepSAD_origin')
# # result_names = os.listdir(base_dir)
# # results_paths = [os.path.join(base_dir, result_name) for result_name in result_names]
# # seed_num = 3
#
# # for result_path in results_paths:
# #     csv_path = os.path.join(result_path, 'AUCROC_DeepSAD_type(None)_noise(None)_unsupervise.csv')
# #     df = pd.read_csv(csv_path)
# #     params_output = []
# #     aucroc_output = []
# #     count = 0
# #     values = []
# #     for index, row in df.iterrows():
# #         params = row['Unnamed: 0']
# #         aucroc = row['Customized']
# #         if math.isnan(aucroc):
# #             continue
# #         dataset, ratio, seed = params.replace('(', '').replace(')', '').replace(' ', '').split(',')
# #         count += 1
# #         values.append(aucroc)
# #         if count == seed_num:
# #             params_output.append((dataset, ratio, 'avg'))
# #             aucroc_output.append(np.mean(values))
# #             count = 0
# #             values = []
# #
# #     df_output = pd.DataFrame(data={'params': params_output, 'aucroc': aucroc_output})
# #     df_output.to_csv(os.path.join(result_path, 'AUCROC_grouped.csv'), index=False)
#
# # 合并输出每个类的结果
#
# def group_results(base_dir):
#     dataset_name_list = os.listdir(base_dir)
#
#     suffix = 'DeepSAD'
#     noise_type = None
#     group_seed_num = 3
#     classes_output = []
#     FDR_output = []
#     FAR_output = []
#     AUCROC_output = []
#     AUCPR_output = []
#     FDR_at_threshold_output = []
#     FAR_at_threshold_output = []
#     for dataset_name in dataset_name_list:
#         if 'csv' in dataset_name:
#             continue
#         FDRs = []
#         FARs = []
#         AUCROCs = []
#         AUCPRs = []
#         FDR_at_thresholds = []
#         FAR_at_thresholds = []
#
#
#         FDR_path = os.path.join(base_dir, dataset_name, f'FDR_{suffix}_type(None)_noise({noise_type})_unsupervise.csv')
#         FAR_path = os.path.join(base_dir, dataset_name, f'FAR_{suffix}_type(None)_noise({noise_type})_unsupervise.csv')
#         AUCROC_path = os.path.join(base_dir, dataset_name,
#                                    f'AUCROC_{suffix}_type(None)_noise({noise_type})_unsupervise.csv')
#         AUCPR_path = os.path.join(base_dir, dataset_name,
#                                   f'AUCPR_{suffix}_type(None)_noise({noise_type})_unsupervise.csv')
#         FDR_at_threshold_path = os.path.join(base_dir, dataset_name,
#                                              f'FDR_at_threshold_{suffix}_type(None)_noise({noise_type})_unsupervise.csv')
#         FAR_at_threshold_path = os.path.join(base_dir, dataset_name,
#                                              f'FAR_at_threshold_{suffix}_type(None)_noise({noise_type})_unsupervise.csv')
#         df_FDR = pd.read_csv(FDR_path)
#         df_FAR = pd.read_csv(FAR_path)
#         df_AUCROC = pd.read_csv(AUCROC_path)
#         df_AUCPR = pd.read_csv(AUCPR_path)
#         df_FDR_at_threshold = pd.read_csv(FDR_at_threshold_path)
#         df_FAR_at_threshold = pd.read_csv(FAR_at_threshold_path)
#         count = 0
#         values = []
#         for index, row in df_FDR.iterrows():
#             aucroc = row['Customized']
#             if math.isnan(aucroc):
#                 continue
#             count += 1
#             values.append(aucroc)
#             if count == group_seed_num:
#                 FDRs.append(np.mean(values))
#                 count = 0
#                 values = []
#
#         count = 0
#         values = []
#         for index, row in df_FAR.iterrows():
#             aucroc = row['Customized']
#             if math.isnan(aucroc):
#                 continue
#             count += 1
#             values.append(aucroc)
#             if count == group_seed_num:
#                 FARs.append(np.mean(values))
#                 count = 0
#                 values = []
#
#         count = 0
#         values = []
#         for index, row in df_AUCROC.iterrows():
#             aucroc = row['Customized']
#             if math.isnan(aucroc):
#                 continue
#             count += 1
#             values.append(aucroc)
#             if count == group_seed_num:
#                 AUCROCs.append(np.mean(values))
#                 count = 0
#                 values = []
#
#         count = 0
#         values = []
#         for index, row in df_AUCPR.iterrows():
#             aucroc = row['Customized']
#             if math.isnan(aucroc):
#                 continue
#             count += 1
#             values.append(aucroc)
#             if count == group_seed_num:
#                 AUCPRs.append(np.mean(values))
#                 count = 0
#                 values = []
#
#         count = 0
#         values = []
#         for index, row in df_FDR_at_threshold.iterrows():
#             aucroc = row['Customized']
#             if math.isnan(aucroc):
#                 continue
#             count += 1
#             values.append(aucroc)
#             if count == group_seed_num:
#                 FDR_at_thresholds.append(np.mean(values))
#                 count = 0
#                 values = []
#
#         count = 0
#         values = []
#         for index, row in df_FAR_at_threshold.iterrows():
#             aucroc = row['Customized']
#             if math.isnan(aucroc):
#                 continue
#             count += 1
#             values.append(aucroc)
#             if count == group_seed_num:
#                 FAR_at_thresholds.append(np.mean(values))
#                 count = 0
#                 values = []
#
#         classes_output.append(dataset_name)
#         FDR_output.append(np.mean(FDRs) * 100)
#         FAR_output.append(np.mean(FARs) * 100)
#         AUCROC_output.append(np.mean(AUCROCs) * 100)
#         AUCPR_output.append(np.mean(AUCPRs) * 100)
#         FDR_at_threshold_output.append(np.mean(FDR_at_thresholds) * 100)
#         FAR_at_threshold_output.append(np.mean(FAR_at_thresholds) * 100)
#
#
#     classes_output.append('mean')
#     FDR_output.append(np.mean(FDR_output))
#     FAR_output.append(np.mean(FAR_output))
#     AUCROC_output.append(np.mean(AUCROC_output))
#     AUCPR_output.append(np.mean(AUCPR_output))
#     FDR_at_threshold_output.append(np.mean(FDR_at_threshold_output))
#     FAR_at_threshold_output.append(np.mean(FAR_at_threshold_output))
#
#     df_output = pd.DataFrame(
#         data={'class': classes_output, 'FDR': FDR_output, 'FAR': FAR_output, 'AUCROC': AUCROC_output,
#               'AUCPR': AUCPR_output, 'FDR_at_threshold': FDR_at_threshold_output,
#               'FAR_at_threshold': FAR_at_threshold_output})
#     df_output.to_csv(os.path.join(base_dir, f'FDR_FAR_AUCROC_AUCPR{suffix}_grouped.csv'), index=False)
#
#     print("Group results finished!")
#
# if __name__ == '__main__':
#
#     base_dir = os.path.join(path_project, 'SMD/log/SMD/DeepSAD', f'DeepSAD,n_epoch=20')
#     group_results(base_dir)
