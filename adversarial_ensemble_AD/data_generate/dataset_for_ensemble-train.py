import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm

# 针对故障检测数据集，构建训练集和测试集用于DeepSAD
path_project = '/home/yukina/Missile_Fault_Detection/project'


class Dataset:
    def __init__(self, X=None, y=None, X_train=None, y_train=None, X_test=None, y_test=None):
        self.X = X
        self.y = y
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def load_data(self, file_path):
        data = np.load(file_path)
        self.X = data['X']
        self.y = data['y']
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_test = data['X_test']
        self.y_test = data['y_test']

    def save_data(self, file_path):
        np.savez(file_path, X=self.X, y=self.y, X_train=self.X_train, y_train=self.y_train, X_test=self.X_test,
                 y_test=self.y_test)

# 生成每个子模型的训练数据
def generate_train_data_seperate(path_output=None, path_normal=None, data_anomaly=None):
    if path_normal is None:
        if data_anomaly is None:
            raise Exception
        else:
            labels = np.ones(data_anomaly.shape[0])
            train_dataset = Dataset(X=[0], y=[0], X_train=data_anomaly, y_train=labels, X_test=data_anomaly[0], y_test=labels[0])
            train_dataset.save_data(os.path.join(path_output, 'init'))
    else:
        for i, file in enumerate(os.listdir(path_normal)):
            data_normal = np.load(os.path.join(path_normal, file))
            if data_anomaly is None:
                train_data = data_normal
                labels = np.zeros(data_normal.shape[0])
            else:
                train_data = np.concatenate((data_normal, data_anomaly), axis=0)
                labels = np.concatenate((np.zeros(data_normal.shape[0]), np.ones(data_anomaly.shape[0])), axis=0)
            train_dataset = Dataset(X=[0], y=[0], X_train=train_data, y_train=labels, X_test=train_data[0], y_test=labels[0])
            train_dataset.save_data(os.path.join(path_output, f'{i}'))

    print('Data generate complete.')


if __name__ == '__main__':
    path_normal = os.path.join(path_project, 'data/banwuli_data/yukina_data/cluster_normal/n=2')
    path_output = os.path.join(path_project, 'data/banwuli_data/yukina_data/train_seperate/init')
    if not os.path.isdir(path_output):
        os.mkdir(path_output)
    generate_train_data_seperate(path_output=path_output, path_normal=path_normal, data_anomaly=None)