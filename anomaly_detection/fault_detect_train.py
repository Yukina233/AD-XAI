path_project = '/home/yukina/Missile_Fault_Detection/project/'
from sklearn.cluster import KMeans
#from spot import dSPOT
import scipy.io
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.manifold import LocallyLinearEmbedding
import matplotlib.pyplot as plt

class Fault_Detect_Train():
    def __init__(self):
        self.Train_Path = path_project + 'anomaly_detection/6x27000.mat'
        self.Test_Path = path_project + 'anomaly_detection/6x27000.mat'
        self.param = ['p_dot','phi','theta','psai','acc','alpha','beta','y','T_command']
        # 获得数据
        self.Train_Data = self.read_mat(self.Train_Path)[3000:, :]
        self.Test_Data = self.read_mat(self.Test_Path)[3000:, :]

        # 归一化
        self.ts_length = 100  # 滑动窗口长度
        self.Train_scaler = RobustScaler().fit(self.Train_Data[0:self.ts_length,:])
        self.Test_scaler = RobustScaler().fit(self.Test_Data[0:self.ts_length,:])
        self.Xtrain = self.Train_scaler.transform(self.Train_Data)
        self.Xtest = self.Test_scaler.transform(self.Test_Data)

        self.Train_extract = self.time_window(self.Xtrain)
        self.Test_extract = self.time_window(self.Xtest)

        # 降维
        self.lle_train_data = LocallyLinearEmbedding(n_neighbors=5, n_components=9).fit_transform(self.Train_extract)
        self.A = self.pro_matirx(self.lle_train_data, self.Train_extract)

        self.kmeans = KMeans(n_clusters=10)
        self.kmeans.fit_transform(self.lle_train_data)


    def read_mat(self,path):
        mat = scipy.io.loadmat(path)
        a = np.squeeze(mat["r_dot"])
        data = np.transpose(np.vstack(
                         (np.squeeze(mat["p_dot"]),
                          np.squeeze((-1)*mat["r_dot"]),
                          np.squeeze(mat["q_dot"]),
                          np.squeeze(mat['phi']),
                          np.squeeze(mat['theta']),
                          np.squeeze(mat['psai']),
                          mat['acc'][:, 0],
                          (-1)*mat['acc'][:, 2],
                          mat['acc'][:, 1],
                          np.squeeze(mat['alpha']),
                          np.squeeze(mat['beta']),
                          np.squeeze(mat['y']),
                          np.squeeze(mat['T_command'])
                          )))
        return data

    def time_window(self,data):
        n, m = np.shape(data)
        std_value = np.zeros((n - self.ts_length + 1, m))
        range_value = np.zeros((n - self.ts_length + 1, m))
        norm_value = np.zeros((n - self.ts_length + 1, m))
        for i in range(m):
            for j in range(0, n - self.ts_length + 1):
                ts = data[j:j + self.ts_length - 1, i]
                std_value[j, i] = np.std(ts)
                range_value[j, i] = np.max(ts) - np.min(ts)
                norm_value[j, i] = np.linalg.norm(ts, ord=2)
        extract_value = np.concatenate((std_value, range_value, norm_value), axis=1)
        return extract_value

    def pro_matirx(self, low_mat, high_mat):
        mat1 = np.dot(np.transpose(low_mat), high_mat)
        mat2 = np.dot(np.transpose(high_mat), high_mat)
        mat = np.dot(mat1, np.linalg.pinv(mat2))
        return mat

    def km_cluster(self,test_new, x_train_new):
        # 降维数据
        center = self.kmeans.cluster_centers_
        test_distance = np.zeros(test_new.shape[0])
        train_distance = np.zeros(self.lle_train_data.shape[0])
        for i in range(test_new.shape[0]):
            test_distance[i] = np.sqrt(np.sum(np.square(test_new[i, :] - center[0])))
        for i in range(x_train_new.shape[0]):
            train_distance[i] = np.sqrt(np.sum(np.square(x_train_new[i, :] - center[0])))
        differ = np.divide(test_distance[:10], train_distance[-10:])
        c = np.mean(differ)
        return train_distance * c, test_distance, center

    def Run(self):
        np.save(path_project + 'anomaly_detection/model/projection.npy', self.A)
        test_new = np.transpose(np.dot(self.A, np.transpose(self.Test_extract)))
        x_train_new = np.transpose(np.dot(self.A, np.transpose(self.Train_extract)))
        train_distance, test_distance, center = self.km_cluster(test_new, x_train_new)
        np.save(path_project + 'anomaly_detection/model/center.npy', center)
        np.save(path_project + 'anomaly_detection/model/train_distance.npy', train_distance)



if __name__ == "__main__":
    test = Fault_Detect_Train()
    test.Run()
    print("finish")



