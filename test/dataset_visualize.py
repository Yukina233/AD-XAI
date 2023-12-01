import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import os

path_project = '/home/yukina/Missile_Fault_Detection/project'
# 下载训练集并应用变换
# trainset = torchvision.datasets.CIFAR10(root=path_project + '/Deep-SAD-OriginalPaper/data', train=False,
#                                         download=True)
# X = trainset.data.reshape((len(trainset.data), -1))
# y = np.array(trainset.targets)
data = np.load(path_project + '/adbench_modified/datasets/CV_by_ResNet18/MVTec-AD_transistor.npz', allow_pickle=True)
X, y = data['X'], data['y']


tsne = TSNE(n_components=2, random_state=0)  # n_components表示目标维度
X_2d = tsne.fit_transform(X)  # 对数据进行降维处理
plt.figure(figsize=(8, 6))
if y is not None:
    # 如果有目标数组，根据不同的类别用不同的颜色绘制
    for i in np.unique(y):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], label=i)
    plt.legend()
else:
    # 如果没有目标数组，直接绘制
    plt.scatter(X_2d[:, 0], X_2d[:, 1])
plt.title('t-SNE Visualization of latent space')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig(path_project + '/test/result/t-SNE Visualization')
print("t-SNE Visualization finished!")
