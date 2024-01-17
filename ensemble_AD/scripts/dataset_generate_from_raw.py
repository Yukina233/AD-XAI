import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm


# This file is used to genrate dataset for each AD task, with iot data.


def generate_dataset(root_dir, task):
    # read csv file
    task_path = os.path.join(root_dir, task)
    gafgyt_path = os.path.join(task_path, 'gafgyt_attacks')
    mirai_path = os.path.join(task_path, 'mirai_attacks')

    scaler = MinMaxScaler()

    df = pd.read_csv(os.path.join(task_path, 'benign_traffic.csv'))
    X_normal = scaler.fit_transform(df.values)
    y_normal = np.zeros(X_normal.shape[0])

    X_anomaly = np.array([])


    if os.path.exists(gafgyt_path):
        gafgyt_files = os.listdir(gafgyt_path)
        for file in gafgyt_files:
            df = pd.read_csv(os.path.join(gafgyt_path, file))

            X_anomaly = np.concatenate((X_anomaly, scaler.fit_transform(df.values)), axis=0) if X_anomaly.size else df.values

    if os.path.exists(mirai_path):
        mirai_files = os.listdir(mirai_path)
        for file in mirai_files:
            df = pd.read_csv(os.path.join(mirai_path, file))
            X_anomaly = np.concatenate((X_anomaly, scaler.fit_transform(df.values)), axis=0) if X_anomaly.size else df.values

    y_anomaly = np.ones(X_anomaly.shape[0])

    # split train and test
    X_train, X_test, y_train, y_test = train_test_split(X_normal, y_normal, test_size=0.2, random_state=42)

    X_test = np.concatenate((X_test, X_anomaly), axis=0)
    y_test = np.concatenate((y_test, y_anomaly), axis=0)


    # Save as NPZ
    output_path = path_project + f"/data/iot_data"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    np.savez(os.path.join(output_path, f"iot_{id}.npz"), X=[0], y=[0], X_train=X_train, y_train=y_train,
             X_test=X_test, y_test=y_test)
    # X, y 不用

path_project = '/home/yukina/Missile_Fault_Detection/project'

# Create the dataset
root_dir = path_project + '/data/iot_raw'
task_list = ['Danmini_Doorbell', 'Ecobee_Thermostat', 'Ennio_Doorbell', 'Philips_B120N10_Baby_Monitor',
                 'Provision_PT_737E_Security_Camera', 'Provision_PT_838_Security_Camera', 'Samsung_SNH_1011_N_Webcam',
                 'SimpleHome_XCS7_1002_WHT_Security_Camera', 'SimpleHome_XCS7_1003_WHT_Security_Camera']
id = 0
for task in tqdm(task_list):
    generate_dataset(root_dir, task)
    id += 1

print('Data generate complete.')
