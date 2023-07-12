path_project = '/home/yukina/Missile_Fault_Detection/project/'
from sklearn.cluster import KMeans
# from spot import dSPOT
import scipy.io
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import LocallyLinearEmbedding


# path = path_project + "anomaly_detection/6x27000.mat"
# mat = scipy.io.loadmat(path)
# a = np.squeeze(mat["r_dot"])
# data = np.transpose(np.vstack(
#     (np.squeeze(mat["p_dot"]),
#      np.squeeze((-1) * mat["r_dot"]),
#      np.squeeze(mat["q_dot"]),
#      np.squeeze(mat['phi']),
#      np.squeeze(mat['theta']),
#      np.squeeze(mat['psai']),
#      mat['acc'][:, 0],
#      (-1) * mat['acc'][:, 2],
#      mat['acc'][:, 1],
#      np.squeeze(mat['alpha']),
#      np.squeeze(mat['beta']),
#      np.squeeze(mat['y']),
#      np.squeeze(mat['T_command'])
#      )))
# print(a)

def create_window_dataset(dataset, k):
    # X是所有长度为k的窗口数据，Y是每个窗口对应的下一个数据点。
    dataX, dataY = [], []
    for i in range(len(dataset) - k):
        a = dataset[i:(i + k), :]
        dataX.append(a)
        dataY.append(dataset[i + k, :])
    return np.array(dataX), np.array(dataY)


data = np.zeros(1000).reshape(-1, 5)
for i in range(200):
    data[i, :] = np.array([i, i, i, i, i])

# 创建窗口大小为k的时序数据
k = 10  # 你可以根据实际情况调整这个值
X, Y = create_window_dataset(data, k)
print(X.shape)
print(Y.shape)
