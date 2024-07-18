import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

path_project = '/home/yukina/Missile_Fault_Detection/project'

output_path = os.path.join(path_project, 'data/gaussian/yukina_data/DeepSAD')

seed = 42
# 设置随机种子以确保可重复性
np.random.seed(seed)

# 生成正常数据（高斯分布）

cov = [[1, 0], [0, 1]]  # 协方差矩阵
normal_data1 = np.random.multivariate_normal([-4, -4], cov, 1500)
normal_data2 = np.random.multivariate_normal([4, 4], cov, 1500)

normal_data = np.vstack([normal_data1, normal_data2])


# 生成异常数据（确保远离高斯分布的中心）
def generate_anomaly_data(num_points, radius_min, radius_max):
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    radii = np.random.uniform(radius_min, radius_max, num_points)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.vstack((x, y)).T


# 设置异常数据的半径范围（确保远离高斯分布中心）
anomaly_data = generate_anomaly_data(50, 8, 12)

train_normal, test_normal, train_normal_labels, test_normal_labels = train_test_split(normal_data,
                                                                                      np.zeros(len(normal_data)),
                                                                                      test_size=0.3,
                                                                                      random_state=seed)
train_data = train_normal
train_labels = train_normal_labels

test_data = np.vstack([test_normal, anomaly_data])
test_labels = np.hstack([test_normal_labels, np.ones(len(anomaly_data))])

scaler = MinMaxScaler()
scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)

np.savez(os.path.join(output_path, f'train.npz'), X=[0], y=[0], X_train=scaled_train_data, y_train=train_labels,
         X_test=[0],
         y_test=[0])

np.savez(os.path.join(output_path, f'test.npz'), X=[0], y=[0], X_train=[0], y_train=[0], X_test=scaled_test_data,
         y_test=test_labels)

print(f"Processed and saved.")



# 可视化数据
plt.figure(figsize=(8, 8))
plt.scatter(train_data[:, 0], train_data[:, 1], c='blue', label='Normal Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('2D Anomaly Detection train Dataset')
plt.axis('equal')  # 设置x轴和y轴的尺度相同
plt.show()

plt.cla()

# 可视化数据
plt.figure(figsize=(8, 8))
plt.scatter(test_normal[:, 0], test_normal[:, 1], c='blue', label='Normal Data')
plt.scatter(anomaly_data[:, 0], anomaly_data[:, 1], c='red', label='Anomaly Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('2D Anomaly Detection Test Dataset')
plt.axis('equal')  # 设置x轴和y轴的尺度相同
plt.show()

