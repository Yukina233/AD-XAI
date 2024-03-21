import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

path_project = '/home/yukina/Missile_Fault_Detection/project'

def get_dat(path):
    a = pd.read_csv(path,delim_whitespace=True)
    b = a[['dOmega_ib_b[0]', 'dOmega_ib_b[1]', 'dOmega_ib_b[2]', 'Gama', 'Theta_k', 'Psi_t', 'fb[0]', 'fb[1]',
           'fb[2]', 'Alfa', 'Beta', 'zmb', 'P']]
    b = np.array(b, dtype=float)
    return b

def timewindow(timeseries, ts_length):  # 滑动时间窗口法
    n, m = np.shape(timeseries)
    std_value = np.zeros((n - ts_length + 1, m))
    range_value = np.zeros((n - ts_length + 1, m))
    norm_value = np.zeros((n - ts_length + 1, m))
    ji_value = np.zeros((n - ts_length + 1, m))
    # min_value = np.zeros((n - ts_length + 1, m))
    for i in range(m):
        for j in range(0, n - ts_length + 1):
            ts = timeseries[j:j + ts_length - 1, i]
            std_value[j, i] = np.std(ts)
            range_value[j, i] = np.max(ts) - np.min(ts)
            norm_value[j, i] = np.linalg.norm(ts, ord=2)
    extract_value = np.concatenate((std_value, range_value, norm_value), axis=1)
    return extract_value

# 用来将原始数据处理成训练数据和测试数据，分别存储在yukina_data文件夹下的normal和anomaly文件夹下
if __name__ == '__main__':
    path_all_data = os.path.join(path_project, "data/banwuli_data")
    fault_list = ['ks', 'lqs', 'rqs', 'sf', 'T']
    normal_start = 2000
    anomaly_start = 5000
    window_size = 100
    # 处理正常数据

    scaler_data = None
    path_normal = os.path.join(path_all_data, 'normal')
    for file in tqdm(os.listdir(path_normal), desc='Read data for robust scaler'):
        path_file = os.path.join(path_normal, file)
        data = get_dat(path_file)
        data = data[normal_start:, :]
        if scaler_data is None:
            scaler_data = data
        else:
            scaler_data = np.concatenate((scaler_data, data))

    scaler = RobustScaler().fit(scaler_data)
    del scaler_data

    nor_data = None
    path_normal = os.path.join(path_all_data, 'normal')
    for file in tqdm(os.listdir(path_normal), desc='Processing normal data'):
        path_file = os.path.join(path_normal, file)
        data = get_dat(path_file)
        data = data[normal_start:, :]
        data = scaler.transform(data)
        data = timewindow(data, window_size)
        if nor_data is None:
            nor_data = data
        else:
            nor_data = np.concatenate((nor_data, data))

    nor_label = [0] * nor_data.shape[0]

    path_normal_save = os.path.join(path_all_data, "yukina_data/normal")
    if not os.path.exists(path_normal_save):
        os.makedirs(path_normal_save)
    np.save(os.path.join(path_project, path_normal_save, "features.npy"), nor_data)
    np.save(os.path.join(path_project, path_normal_save, "labels.npy"), nor_label)

    # 遍历给定路径下的所有目录
    for fault in tqdm(fault_list, desc='Processing anomaly data'):
        anomaly_data = None
        anomaly_label = None
        path_anomaly = os.path.join(path_all_data, fault)
        for location in os.listdir(path_anomaly):
            path_location = os.path.join(path_anomaly, location)
            for file in os.listdir(path_location):
                path_file = os.path.join(path_location, file)
                data = get_dat(path_file)
                data = data[normal_start:, :]
                data = scaler.transform(data)
                data = timewindow(data, window_size)
                X_0 = data[:anomaly_start - normal_start - window_size,:]
                X_1 = data[anomaly_start - normal_start - window_size:,:]
                Y_0 = [0] * X_0.shape[0]
                Y_1 = [1] * X_1.shape[0]
                if anomaly_data is None and anomaly_label is None:
                    anomaly_data = np.concatenate((X_0, X_1))
                    anomaly_label = np.concatenate((Y_0, Y_1))
                else:
                    anomaly_data = np.concatenate((anomaly_data, X_0, X_1))
                    anomaly_label = np.concatenate((anomaly_label, Y_0, Y_1))

        path_anomaly_save = os.path.join(path_all_data, "yukina_data/anomaly", fault)
        if not os.path.exists(path_anomaly_save):
            os.makedirs(path_anomaly_save)
        np.save(os.path.join(path_anomaly_save, "features.npy"), anomaly_data)
        np.save(os.path.join(path_anomaly_save, "labels.npy"), anomaly_label)
