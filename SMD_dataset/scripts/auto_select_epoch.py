import os

import pandas as pd

path_project = '/home/yukina/Missile_Fault_Detection/project'


def auto_select_epoch(file_path):
    # 读取所有实验结果
    for epoch in range(0, 20):
        # 读取当前实验的结果
        mean_df = pd.read_csv(os.path.join(file_path, f'{epoch}', 'mean_results.csv'))
        std_df = pd.read_csv(os.path.join(file_path, f'{epoch}', 'std_results.csv'))
        # 去除最后一行数据
        mean_df.drop(mean_df.index[-1], inplace=True)
        std_df.drop(std_df.index[-1], inplace=True)
        if epoch == 0:
            # 按照class编号保存
            num = mean_df['class'].size
            reformed_list = []
            for i in range(num):
                i_class_df = pd.DataFrame(
                    {'class': [mean_df['class'][i]], 'AUCROC_mean': [mean_df['AUCROC'][i]],
                     'AUCROC_std': [std_df['AUCROC'][i]],
                     'AUCPR_mean': [mean_df['AUCPR'][i]],
                     'AUCPR_std': [std_df['AUCPR'][i]]})
                reformed_list.append(i_class_df)
        else:
            for i in range(num):
                reformed_list[i] = pd.concat([reformed_list[i], pd.DataFrame(
                    {'class': [mean_df['class'][i]], 'AUCROC_mean': [mean_df['AUCROC'][i]],
                     'AUCROC_std': [std_df['AUCROC'][i]],
                     'AUCPR_mean': [mean_df['AUCPR'][i]],
                     'AUCPR_std': [std_df['AUCPR'][i]]})])

    for i in range(num):
        # 找到AUCROC_mean最大的一行
        max_aucroc_index = reformed_list[i]['AUCROC_mean'].argmax()

        # 将这一行的数据提取出来
        max_aucroc_data = reformed_list[i].iloc[max_aucroc_index]
        max_aucroc_df = pd.DataFrame(
            {'class': [max_aucroc_data['class']], 'AUCROC_mean': [max_aucroc_data['AUCROC_mean']],
             'AUCROC_std': [max_aucroc_data['AUCROC_std']], 'AUCPR_mean': [max_aucroc_data['AUCPR_mean']],
             'AUCPR_std': [max_aucroc_data['AUCPR_std']], 'epoch': [max_aucroc_index]})

        # 存到新的DataFrame中
        if i == 0:
            best_epoch_df = max_aucroc_df
        else:
            best_epoch_df = pd.concat([best_epoch_df, max_aucroc_df])

    # 增加一行mean结果
    best_epoch_df = pd.concat([best_epoch_df, pd.DataFrame(
        {'class': ['mean'], 'AUCROC_mean': [best_epoch_df['AUCROC_mean'].mean()],
         'AUCROC_std': [best_epoch_df['AUCROC_std'].mean()],
         'AUCPR_mean': [best_epoch_df['AUCPR_mean'].mean()],
         'AUCPR_std': [best_epoch_df['AUCPR_std'].mean()],
         'epoch': [best_epoch_df['epoch'].mean()]})])

    # 保存到文件
    best_epoch_df.to_csv(os.path.join(file_path, 'best_epoch.csv'), index=False)
    print('Finished selecting best epoch!')

if __name__ == '__main__':
    test_set_name = 'SMD'
    # 定义根目录
    base_dir = os.path.join(path_project, f'{test_set_name}_dataset/log/{test_set_name}/ensemble/DeepSAD/seed_group')
    prefix = 'WGAN-GP, euc, window=100, step=10, no_tau2_K=7,deepsad_epoch=1,gan_epoch=1,lam1=1000000,lam2=0,latent_dim=180,lr=0.0002,clip_value=0.01,lambda_gp=1000,seed'

    auto_select_epoch(os.path.join(base_dir, prefix))
