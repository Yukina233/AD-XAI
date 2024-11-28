import os

import numpy as np

path_project = '/home/yukina/Missile_Fault_Detection/project_data'

test_set_name = 'GHL'

test_data_path = os.path.join(path_project, f'data/{test_set_name}/yukina_data/DeepSAD_data, window=100, step=10')

all_anomaly_data = []
for dataset_name in os.listdir(test_data_path):
    if not (dataset_name.startswith('test') or dataset_name.endswith('test.npz')):
        continue

    data = np.load(os.path.join(test_data_path, dataset_name))
    dataset = {'X': data['X'], 'y': data['y'], 'X_train': data['X_train'], 'y_train': data['y_train'],
               'X_test': data['X_test'], 'y_test': data['y_test']}
    anomaly_data = dataset['X_test'][np.where(dataset['y_test'] == 1)]
    all_anomaly_data.append(anomaly_data)

all_anomaly_data = np.concatenate(all_anomaly_data)
np.savez(os.path.join(test_data_path, 'all_anomally_data'), X=[], y=[], X_train=[], y_train=[], X_test=all_anomaly_data, y_test=np.ones(all_anomaly_data.shape[0]))

