import os
from itertools import combinations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm
import os
from glob import glob

# 针对故障检测数据集，构建训练集和测试集用于DeepSAD
path_project = '/home/yukina/Missile_Fault_Detection/project'


def extract_features(windows):
    """
    Extract statistical features from each window.

    Parameters:
    - windows: 3D numpy array of shape (num_windows, window_size, num_features)

    Returns:
    - features: 2D numpy array of shape (num_windows, num_extracted_features)
    """
    num_windows, window_size, num_features = windows.shape
    feature_list = []

    for window in windows:
        window_features = []
        for col in range(num_features):
            series = window[:, col]
            window_features.extend([
                np.mean(series),
                np.std(series),
                np.min(series),
                np.max(series),
                np.median(series),
                # series
                # You can add more features here if needed
            ])
        feature_list.append(window_features)

    return np.array(feature_list)


def windowed_features(data, labels, window_size, step_size):
    """
    Extract windowed features and labels from the data.

    Parameters:
    - data: The input time series data (numpy array).
    - labels: The labels corresponding to the data (numpy array).
    - window_size: The size of each window.
    - step_size: The step size to move the window.

    Returns:
    - windowed_data: A numpy array of windowed features.
    - windowed_labels: A numpy array of windowed labels.
    """
    num_windows = (len(data) - window_size) // step_size + 1
    windows = np.array([data[i * step_size:i * step_size + window_size] for i in range(num_windows)])
    windowed_labels = np.array(
        [1 if np.any(labels[i * step_size:i * step_size + window_size] == 1) else 0 for i in range(num_windows)])

    # windowed_labels = np.array([
    #     1 if np.sum(labels[i * step_size:i * step_size + window_size]) > (window_size / 2) else 0
    #     for i in range(num_windows)
    # ])

    # Extract features from each window
    windowed_data = extract_features(windows)

    return windowed_data, windowed_labels


# def windowed_features(data, labels, window_size, step_size):
#     """
#     Extract windowed features and labels from the data.
#
#     Parameters:
#     - data: The input time series data (numpy array of shape (num_samples, num_features)).
#     - labels: The labels corresponding to the data (numpy array of shape (num_samples,)).
#     - window_size: The size of each window.
#     - step_size: The step size to move the window.
#
#     Returns:
#     - windowed_data: A numpy array of windowed features with shape (num_windows, window_size, num_features).
#     - windowed_labels: A numpy array of windowed labels with shape (num_windows,).
#     """
#     num_samples, num_features = data.shape
#     num_windows = (num_samples - window_size) // step_size + 1
#
#     windows = np.array([data[i * step_size:i * step_size + window_size] for i in range(num_windows)])
#     windowed_labels = np.array([
#         1 if np.sum(labels[i * step_size:i * step_size + window_size]) > (window_size / 2) else 0
#         for i in range(num_windows)
#     ])
#
#     return windows.reshape(windows.shape[0], -1), windowed_labels

def process_datasets(input_path, output_path, plot_path, window_size, step_size):
    test_files = glob(os.path.join(input_path, '*test.csv'))

    data_list = []
    for test_file in tqdm(test_files):
        # Derive the corresponding test file name

        # Load the training and test data

        data_df = pd.read_csv(test_file)

        data = data_df.iloc[:, 1:-1].values
        labels = data_df.iloc[:, -1].values

        windows, window_labels = windowed_features(data, labels, window_size, step_size)

        data_list.append((windows, window_labels))


    # Combine all data

    all_windows = np.concatenate([data[0] for data in data_list], axis=0)
    all_labels = np.concatenate([data[1] for data in data_list], axis=0)

    all_normal_windows = all_windows[all_labels == 0]
    all_normal_labels = all_labels[all_labels == 0]

    all_anomaly_windows = all_windows[all_labels == 1]
    all_anomaly_labels = all_labels[all_labels == 1]
    # Split the data into training and test sets
    train_normal_windows, test_normal_windows, train_normal_labels, test_normal_labels = train_test_split(all_normal_windows, all_normal_labels,
                                                                                            test_size=0.2,
                                                                                            random_state=42)
    train_windows = train_normal_windows
    train_labels = train_normal_labels

    test_windows = np.concatenate((test_normal_windows, all_anomaly_windows))
    test_labels = np.concatenate((test_normal_labels, all_anomaly_labels))

    # Normalize the data
    scaler = MinMaxScaler()
    scaler.fit(train_windows)
    scaled_train_windows = scaler.transform(train_windows)
    scaled_test_windows = scaler.transform(test_windows)

    # Save to npz file

    np.savez(os.path.join(output_path, f'train.npz'), X=[0], y=[0], X_train=scaled_train_windows, y_train=train_labels, X_test=[0],
             y_test=[0])

    np.savez(os.path.join(output_path, f'test.npz'), X=[0], y=[0], X_train=[0], y_train=[0], X_test=scaled_test_windows,
             y_test=test_labels)

    print(f"Processed and saved.")

    # # 用TSNE打印训练集和测试集的数据分布
    # # 随机抽取一部分数据进行可视化
    num_samples = 5000
    np.random.seed(0)
    x_normal = train_windows[np.where(train_labels == 0)]
    x_anomaly = test_windows[np.where(test_labels == 1)]
    sampled_normal = x_normal[np.random.choice(range(0, x_normal.shape[0]), num_samples, replace=True)]
    sampled_anomaly = x_anomaly[np.random.choice(range(0, x_anomaly.shape[0]), num_samples, replace=True)]
    X = np.concatenate((sampled_normal, sampled_anomaly))
    y = np.concatenate((np.zeros(num_samples), np.ones(num_samples)))
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(X)
    plt.figure(figsize=(12, 9))
    if y is not None:
        for i in range(2):
            plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], label=f'Label {i}')
        plt.title(f'Train data distribution')
        plt.legend()
        # plt.show()
        plt.savefig(os.path.join(plot_path, f'train.png'))

    x_normal = test_windows[np.where(test_labels == 0)]
    x_anomaly = test_windows[np.where(test_labels == 1)]
    sampled_normal = x_normal[np.random.choice(range(0, x_normal.shape[0]), num_samples, replace=True)]
    sampled_anomaly = x_anomaly[np.random.choice(range(0, x_anomaly.shape[0]), num_samples, replace=True)]
    X = np.concatenate((sampled_normal, sampled_anomaly))
    y = np.concatenate((np.zeros(num_samples), np.ones(num_samples)))
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(X)
    plt.figure(figsize=(12, 9))
    if y is not None:
        for i in range(2):
            plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], label=f'Label {i}')
        plt.title(f'Test data distribution')
        plt.legend()
        # plt.show()
        plt.savefig(os.path.join(plot_path, f'test.png'))


# Example usage
window_size = 100  # Define your window size
step_size = 10  # Define your step size
input_path = os.path.join(path_project, 'data/Daphnet')
output_path = os.path.join(path_project,
                           f'data/Daphnet/yukina_data/DeepSAD_data, window={window_size}, step={step_size}')
plot_path = os.path.join(path_project, f'data/Daphnet/plot/DeepSAD_data, window={window_size}, step={step_size}')
os.makedirs(output_path, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)

process_datasets(input_path, output_path, plot_path, window_size, step_size)

print('Data generate complete.')
# # Create the dataset
# root_dir = os.path.join(path_project, 'data/SMD')
# # Save as NPY
#
# X_train = np.load(os.path.join(root_dir, 'train/features.npy'))
# y_train = np.load(os.path.join(root_dir, 'train/labels.npy'))
#
# for fault in tqdm(os.listdir(os.path.join(root_dir, 'test'))):
#     path_fault = os.path.join(root_dir, 'test', fault)
#     files = os.listdir(path_fault)
#
#     for i in range(1, int(len(files) / 2 + 1)):
#         X_test = np.load(os.path.join(root_dir, 'test', fault, f"features_{i}.npy"))
#         y_test = np.load(os.path.join(root_dir, 'test', fault, f"labels_{i}.npy"))
#
#         output_path = os.path.join(path_project, root_dir, 'DeepSAD_data', fault)
#         os.makedirs(output_path, exist_ok=True)
#
#         np.savez(os.path.join(output_path, f"dataset_{i}.npz"), X=[0], y=[0], X_train=X_train, y_train=y_train,
#              X_test=X_test, y_test=y_test)
#
# print('Data generate complete.')
