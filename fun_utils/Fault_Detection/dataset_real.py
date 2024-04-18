import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

path_project = '/home/yukina/Missile_Fault_Detection/project'


def get_dat(path):
    a = pd.read_csv(path)
    b = a[
        ['滚转角', '俯仰角', '航向角', '滚转角速度', '俯仰角速度', '航向角速度', 'X轴加速度', 'Y轴加速度', 'Z轴加速度',
         '6号舵机', '7号舵机', '8号舵机', '9号舵机','故障类型']]
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
    path_all_data = os.path.join(path_project, "data/real_missile_data")
    fault_list = ['sf']
    window_size = 10
    # 处理正常数据

    scaler_data = None
    path_train = os.path.join(path_all_data, 'normal')
    for file in tqdm(os.listdir(path_train), desc='Read data for robust scaler'):
        path_file = os.path.join(path_train, file)
        data = get_dat(path_file)
        data = data[:, :-1]
        if scaler_data is None:
            scaler_data = data
        else:
            scaler_data = np.concatenate((scaler_data, data))

    scaler = RobustScaler().fit(scaler_data)
    del scaler_data

    X_train = None
    path_train = os.path.join(path_all_data, 'normal')
    for file in tqdm(os.listdir(path_train), desc='Processing train data'):
        path_file = os.path.join(path_train, file)
        data = get_dat(path_file)
        label = data[:, -1].astype(int)
        data = data[:, :-1]
        data = scaler.transform(data)
        data = timewindow(data, window_size)
        if X_train is None:
            X_train = data
        else:
            X_train = np.concatenate((X_train, data))

    y_train = [0] * X_train.shape[0]

    path_train_save = os.path.join(path_all_data, "yukina_data/train")
    if not os.path.exists(path_train_save):
        os.makedirs(path_train_save)
    np.save(os.path.join(path_project, path_train_save, "features.npy"), X_train)
    np.save(os.path.join(path_project, path_train_save, "labels.npy"), y_train)

    # 遍历给定路径下的所有目录
    for fault in tqdm(fault_list, desc='Processing test data'):
        count = 0
        X_test = None
        y_test = None
        path_test = os.path.join(path_all_data, fault)
        for location in os.listdir(path_test):
            path_location = os.path.join(path_test, location)
            for file in os.listdir(path_location):
                path_file = os.path.join(path_location, file)
                data = get_dat(path_file)
                label = data[:, -1].astype(int)
                label[label != 0] = 1
                data = data[:, :-1]
                data = scaler.transform(data)
                data = timewindow(data, window_size)
                anomaly_idx = np.argwhere(label == 1)
                anomaly_start = anomaly_idx[0].item()
                anomaly_end = anomaly_idx[-1].item()

                X_0_a = data[:(anomaly_start - window_size + 1), :]
                X_0_b = data[(anomaly_end + 1):, :]
                X_0 = np.concatenate((X_0_a, X_0_b))
                X_1 = data[(anomaly_start - window_size + 1) : anomaly_end + 1, :]

                Y_0 = [0] * X_0.shape[0]
                Y_1 = [1] * X_1.shape[0]

                X_test = np.concatenate((X_0, X_1))
                y_test = np.concatenate((Y_0, Y_1))

                count += 1

                path_anomaly_save = os.path.join(path_all_data, "yukina_data/test", fault)
                if not os.path.exists(path_anomaly_save):
                    os.makedirs(path_anomaly_save)
                np.save(os.path.join(path_anomaly_save, f"features_{count}.npy"), X_test)
                np.save(os.path.join(path_anomaly_save, f"labels_{count}.npy"), y_test)
