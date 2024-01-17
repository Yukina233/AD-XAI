import glob
import os

import matplotlib.pyplot as plt

import pandas as pd
from numpy import log

path_project = '/home/yukina/Missile_Fault_Detection/project'

data_path = os.path.join(path_project, 'ensemble_AD/log/ensemble', 'DeepSAD_origin_seed=1', 'ensemble_results_var')

task_dirs = glob.glob(data_path + '/iot*')

for task_dir in task_dirs:
    df_results = pd.read_csv(os.path.join(data_path, task_dir, 'results.csv'))
    # df_scores = pd.read_csv(os.path.join(data_path, task_dir, 'scores.csv'))

    results = df_results.to_dict('records')
    # scores = df_scores.to_dict('records')


    # 分离标签为1和标签为0的数据
    var_label_1 = [log(item['var']) for item in results if item['label'] == 1]
    var_label_0 = [log(item['var']) for item in results if item['label'] == 0]

    # 绘制直方图
    plt.hist(var_label_1, alpha=0.5, label='Label 1', bins=100)  # alpha设置透明度，便于看到重叠部分
    plt.hist(var_label_0, alpha=0.5, label='Label 0', bins=100)

    # 添加图例
    plt.legend()

    # 添加标题和轴标签
    plt.title('Anomaly Value Distribution by Label')
    plt.xlabel('Anomaly Value')
    plt.ylabel('Frequency')

    # 显示图形
    plt.savefig(os.path.join(task_dir, 'hist'))

    plt.cla()