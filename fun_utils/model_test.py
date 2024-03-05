from keras.models import load_model
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
import time
class train_test():
    def __init__(self):
        self.filePath = r"..\data\test"  # 数据所在的文件夹
        self.model = load_model(r"..\model\Fault_Identif_Train.h5")

        self.SF_Fault_Path = os.path.join(self.filePath, "sf")  # 松浮故障数据路径
        self.RQS_Fault_Path = os.path.join(self.filePath, "r")  # 舵面缺损故障数据路径
        self.KS_Fault_Path = os.path.join(self.filePath, "ks")  # 舵面卡死故障数据路径
        self.LQS_Fault_Path = os.path.join(self.filePath, "l")  # 升力面缺损故障数据
        self.T_Fault_Path = os.path.join(self.filePath, "T")  # 推力下降数据数据路径
        self.gyro_abrupt_Path = os.path.join(self.filePath, "gy_a")
        self.gyro_slow_Path = os.path.join(self.filePath, "gy_s")
        self.altimeter_abrupt_Path = os.path.join(self.filePath, "h_a")
        self.altimeter_slow_Path = os.path.join(self.filePath, "h_s")

        self.class_num = 9
        self.param = ['dOmega_ib_b[0]', 'dOmega_ib_b[1]', 'dOmega_ib_b[2]', 'Gama', 'Theta_k', 'Psi_t', 'fb[0]',
                      'fb[1]', 'fb[2]', 'Alfa', 'Beta', 'zmb', 'P']  # 需要提取的列
        # self.param = ['Gama', 'Theta_k', 'Psi_t', 'fb[2]', 'Alfa', 'Beta','Omega_ib_b[0]', 'Omega_ib_b[2]', 'dOmega_ib_b[0]', 'dOmega_ib_b[2]']
        self.scaler = RobustScaler().fit(self.read_dat(r"..\data\normal\zc1.dat")[3000:10000, :])
        self.lookback = 5
        self.LAET = load_model(r"..\model\Lstm_Auto_Encoder.h5")  # 加载LSTM自编码器模型用于残差生成

        self.SF_Data = self.get_Train_data(self.SF_Fault_Path, "sf")
        self.KS_Data = self.get_Train_data(self.KS_Fault_Path, "ks")
        self.RQS_Data = self.get_Train_data(self.RQS_Fault_Path, "rqs")
        self.LQS_Data = self.get_Train_data(self.LQS_Fault_Path, "lqs")
        self.T_Data = self.get_Train_data(self.T_Fault_Path, "T")
        self.gyro_abrupt_Data = self.get_Train_data(self.gyro_abrupt_Path, "gyro_abrupt")
        self.gyro_slow_Data = self.get_Train_data(self.gyro_slow_Path, "gyro_slow")
        self.altimeter_abrupt_Data = self.get_Train_data(self.altimeter_abrupt_Path,"altimeter_abrupt")
        self.altimeter_slow_Data = self.get_Train_data(self.altimeter_slow_Path,"altimeter_slow")

        self.train_data_pre = np.concatenate((self.SF_Data, self.KS_Data, self.RQS_Data, self.LQS_Data, self.T_Data,
                                              self.gyro_abrupt_Data, self.gyro_slow_Data, self.altimeter_abrupt_Data,
                                              self.altimeter_slow_Data), axis=0)
        self.train_data_real = self.residual()
        # self.train_data_real = np.load(r"..\data\test\hhs_data\residual1.npy")
        # self.label = [0]*8*12+[2]*8*12+[1]*8*12+[4]*8*12+[3]*8*12+[5]*8*12+[6]*8*12+[7]*8*12+[8]*8*12
        self.label = [0] * 8  + [2] * 8  + [1] * 8  + [4] * 8  + [3] * 8 + [5] * 8  + [6] * 8  + [7] * 8  + [8] * 8
        print(0)

    def read_dat(self, path):
        a = pd.read_csv(path, delim_whitespace=True)
        return np.array(a[self.param], dtype=float)

    # 滑动窗口
    def temporalize(self, X):
        output_X = []
        for i in range(len(X) - self.lookback - 1):
            t = []
            for j in range(1, self.lookback + 1):
                t.append(X[[i + j + 1], :])
            output_X.append(t)
        return np.reshape(np.array(output_X), (np.array(output_X).shape[0], self.lookback, np.array(output_X).shape[3]))

    # 根据输入路径提取出数据训练集和标签
    def get_Train_data(self,Path,flag):
        # train_data = np.array([self.temporalize(self.scaler.transform(self.read_dat(os.path.join(os.path.join(os.path.join(self.filePath,Path),dic),data_id))[2000:5200,:])) for dic in os.listdir(Path) for data_id in os.listdir(os.path.join(os.path.join(self.filePath, Path), dic))])
        train_data = [self.temporalize(self.scaler.transform(self.read_dat(os.path.join(Path,dic))[2000:5200,:])) for dic in os.listdir(Path)]
        np.save(r"..\data\test\hhs_data\{}.npy".format(flag),train_data)  # 数据保存
        return np.array(train_data)

        # 将3维LSTM数据展平为2维
    def data_flatten(self, X):
        flattened_X = np.array([X[i, (X.shape[1] - 1), :] for i in range(X.shape[0])])
        return flattened_X

    def residual(self):
        XX = [self.data_flatten(real_data)-self.data_flatten(self.LAET.predict(real_data)) for real_data in self.train_data_pre]
        # X = np.array([residual[3007+j:3034+j,:].T for residual in XX for j in range(12)])
        # X = np.array([residual[3007+j:3034+j, :].T for residual in XX for j in range(12)])
        X = np.array([residual[3007:3034 :].T for residual in XX])
        np.save(r"..\data\test\hhs_data\residual1.npy", X)  # 保存训练集残差
        return X

    def cof(self,model):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        t = time.perf_counter()
        train_pred = model.predict(self.train_data_real)
        print("用时：{}".format((time.perf_counter()-t)/len(self.train_data_real)))   #0.0216236388888889
        train_pred = np.argmax(train_pred, axis=1)
        con_mat = confusion_matrix(self.label, train_pred)
        con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]  # 归一化
        con_mat_norm = np.around(con_mat_norm, decimals=2)
        plt.figure(figsize=(10, 10))
        print(con_mat)
        LABELS = ['松浮故障', '缺损故障', '卡死故障','升力面缺损',"推力损失","陀螺仪突变故障","陀螺仪缓变故障","气压计突变故障","气压计缓变故障"]
        sns.heatmap(con_mat_norm, xticklabels=LABELS, yticklabels=LABELS, annot=True)

        plt.show()
        print('a', accuracy_score(self.label, train_pred))
        print('f', f1_score(self.label, train_pred, average='macro'))
        print('r', recall_score(self.label, train_pred, average='macro'))
        # 0.9583333333333334 a
        # 0.9591757170704539 f
        # 0.9583333333333334 r

if __name__ =="__main__":
    a = train_test()
    a.cof(a.model)
