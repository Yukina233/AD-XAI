import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# 生成示例数据
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 使用DBSCAN进行聚类
# eps是邻域的大小，min_samples是形成一个核心点所需要的邻域样本数
db = DBSCAN(eps=0.5, min_samples=10).fit(X)

# 标签是-1的点是噪声点
labels = db.labels_

# 获取核心样本的索引
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

# 绘制结果
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # 黑色用于噪声点
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    # 绘制核心点
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    # 绘制非核心点
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % len(unique_labels))
plt.show()