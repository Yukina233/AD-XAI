import glob
import os
import pickle
import time

import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

path_project = '/home/yukina/Missile_Fault_Detection/project/'
sub_path = 'anchor/results'
seed = 0

# 导入模型
def load_all_h5_models(directory):
    # 使用glob找到目录中所有的.h5文件
    h5_files = glob.glob(os.path.join(directory, '*.h5'))

    # 加载所有的模型并存储在一个字典中
    models = {}
    for file in h5_files:
        model_name = os.path.basename(file).split('.')[0]
        models[model_name] = load_model(file)

    return models


def load_all_pickle_files(directory):
    # 使用glob找到目录中所有的.h5文件
    pickle_files = glob.glob(os.path.join(directory, '*.pkl'))

    # 加载所有的模型并存储在一个字典中
    files = {}
    for file in pickle_files:
        file_name = os.path.basename(file).split('.')[0]
        files[file_name] = pickle.load(open(file, 'rb'))

    return files


# 使用函数加载目录下所有的.h5模型
directory = path_project + sub_path
all_models = load_all_h5_models(directory)
all_pickle_files = load_all_pickle_files(directory)

# 输出加载的模型
for model_name, model in all_models.items():
    print(f'Model {model_name} is loaded.')

# 绘制样本预测
plt.figure(figsize=(20, 10))
for file_name, explanation_dict in all_pickle_files.items():
    explanation_df = pd.DataFrame(explanation_dict)
    # 将影响力数据按照影响力大小排序
    explanation_df = explanation_df.sort_values(by='prediction', ascending=False)
    # dataframe转为dict
    explanation_dict = explanation_df.to_dict(orient='list')
    plt.plot(explanation_dict['prediction'], label=file_name)
plt.xlabel('id')
plt.ylabel('prediction')
plt.title('Prediction')
plt.legend()
plt.show()

# 绘制anchor个数
plt.figure(figsize=(20, 10))
for file_name, explanation_dict in all_pickle_files.items():
    explanation_df = pd.DataFrame(explanation_dict)
    # 将影响力数据按照影响力大小排序
    explanation_df = explanation_df.sort_values(by='prediction', ascending=False)
    # dataframe转为dict
    explanation_dict = explanation_df.to_dict(orient='list')
    achors_num = []
    for anchor_list in explanation_dict['anchor']:
        achors_num.append(len(anchor_list))
    explanation_dict['anchor_num'] = achors_num
    plt.plot(explanation_dict['anchor_num'], label=file_name)
plt.xlabel('id')
plt.ylabel('anchor_num')
plt.title('anchor_num')
plt.legend()
plt.show()

# 绘制precision
plt.figure(figsize=(20, 10))
for file_name, explanation_dict in all_pickle_files.items():
    explanation_df = pd.DataFrame(explanation_dict)
    # 将影响力数据按照影响力大小排序
    explanation_df = explanation_df.sort_values(by='prediction', ascending=False)
    # dataframe转为dict
    explanation_dict = explanation_df.to_dict(orient='list')
    plt.plot(explanation_dict['precision'], label=file_name)
plt.xlabel('id')
plt.ylabel('precision')
plt.title('precision')
plt.legend()
plt.show()

# 绘制coverage
plt.figure(figsize=(20, 10))
for file_name, explanation_dict in all_pickle_files.items():
    explanation_df = pd.DataFrame(explanation_dict)
    # 将影响力数据按照影响力大小排序
    explanation_df = explanation_df.sort_values(by='prediction', ascending=False)
    # dataframe转为dict
    explanation_dict = explanation_df.to_dict(orient='list')
    plt.plot(explanation_dict['coverage'], label=file_name)
plt.xlabel('id')
plt.ylabel('coverage')
plt.title('coverage')
plt.legend()
plt.show()