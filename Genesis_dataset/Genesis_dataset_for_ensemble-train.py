import os
from itertools import combinations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm
import os
from glob import glob

# 针对故障检测数据集，构建训练集和测试集用于DeepSAD
path_project = '/home/yukina/Missile_Fault_Detection/project'

# Create the dataset
root_dir = os.path.join(path_project, 'data/Genesis/yukina_data')


# Save as NPY


# 定义Kmeans聚类函数
def kmeans_clustering(X_train, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X_train)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    return centroids, labels


num_clusters = 7
output_path = os.path.join(root_dir, f'ensemble_data, window=1, step=1/init/K={num_clusters}')
os.makedirs(output_path, exist_ok=True)

input_path = os.path.join(root_dir, 'DeepSAD_data, window=1, step=1')

data = np.load(os.path.join(input_path, 'train.npz'))

X_train = data['X_train']
y_train = data['y_train']


# 调用Kmeans聚类函数
centroids, cluster_labels = kmeans_clustering(X_train, num_clusters)

# 按照聚类结果划分训练数据
groups = [X_train[np.where(cluster_labels == i)] for i in range(num_clusters)]

X = X_train
y = cluster_labels
tsne = TSNE(n_components=2, random_state=42)  # n_components表示目标维度

X_2d = tsne.fit_transform(X)  # 对数据进行降维处理
plt.figure(figsize=(12, 9))
if y is not None:
    # 如果有目标数组，根据不同的类别用不同的颜色绘制
    for i in np.unique(y):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], label=i, alpha=0.5)
    plt.legend()
else:
    # 如果没有目标数组，直接绘制
    plt.scatter(X_2d[:, 0], X_2d[:, 1])
plt.title('t-SNE Visualization of normal data after clustering')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

# X_3d = tsne.fit_transform(X)
# # 创建3D坐标系
# fig = plt.figure(figsize=(16, 12))
# ax = fig.add_subplot(111, projection='3d')
#
# # 绘制散点图
# scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y, s=8, alpha=0.5)
#
# # 添加标题和坐标轴标签
# ax.set_title('t-SNE 3D Scatter Plot')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#
# # 添加颜色条
# cbar = plt.colorbar(scatter)

plt.show()


for item in enumerate(groups):
    np.savez(os.path.join(output_path, f'{item[0]}'), X=[0], y=[0], X_train=item[1],
             y_train=np.zeros(item[1].shape[0]), X_test=[0],
             y_test=[0])

print('Data cluster complete.')
