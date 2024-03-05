import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import os
from keras.layers import Dense,LSTM,RepeatVector,TimeDistributed
from keras.models import Sequential

#  LSTM自编码器的训练程序程序
class LSTM_Auto_Encoder_Train():
    def __init__(self):
        self.filePath = r"..\..\data\normal"     # 数据所在的文件夹
        self.data_name = os.listdir(self.filePath)
        self.data_normalize_path = os.path.join(self.filePath,self.data_name[0])      # 将文件夹中的第一个数据用于归一化

        # 根据路径提取需要的行
        self.param = ['dOmega_ib_b[0]', 'dOmega_ib_b[1]', 'dOmega_ib_b[2]', 'Gama', 'Theta_k', 'Psi_t', 'fb[0]', 'fb[1]',
           'fb[2]', 'Alfa', 'Beta', 'zmb', 'P']    # 需要提取的列
        # self.param = ['Gama', 'Theta_k', 'Psi_t', 'fb[2]', 'Alfa', 'Beta', 'Omega_ib_b[0]', 'Omega_ib_b[2]', 'dOmega_ib_b[0]', 'dOmega_ib_b[2]']
        self.data_normalize = self.read_dat(self.data_normalize_path)

        # 获得训练集数据
        self.scaler = RobustScaler().fit(self.data_normalize)  # 注意这里3000，10000可调（根据数据具体长度）
        self.lookback = 5  # 设置数据的滑动窗口为5
        self.train_data = self.get_train_data()   # 自动获得训练数据

        # 获得验证集数据
        self.valid_data = self.temporalize(RobustScaler().fit_transform(self.data_normalize))

        # 训练模型
        self.epoch = 100
        self.batch = 64
        self.lr = 0.0001
        self.time_steps = self.train_data.shape[1]
        self.n_features = self.train_data.shape[2]
        self.model = self.get_model()        # 获得模型

    # 输入数据的路径，读取需要的数据
    def read_dat(self, path):
        a = pd.read_csv(path, delim_whitespace=True)
        return np.array(a[self.param], dtype=float)[3000:10000, :]   # 3000,10000可修改，根据导弹平飞段和总时间决定

    # 滑动窗口
    def temporalize(self, X):
        output_X = []
        for i in range(len(X) - self.lookback - 1):
            t = []
            for j in range(1, self.lookback + 1):
                t.append(X[[i + j + 1], :])
            output_X.append(t)
        return np.reshape(np.array(output_X),(np.array(output_X).shape[0], self.lookback, np.array(output_X).shape[3]))

    # 将数据集中第一个数据除外的所有数据汇总，归一化并滑动后输出
    def get_train_data(self):
        return np.vstack([self.temporalize(self.scaler.transform(self.read_dat(os.path.join(self.filePath,self.data_name[i])))) for i in range(1,len(self.data_name))])

    def get_model(self):
        model = Sequential()
        model.add(LSTM(32, activation='relu', input_shape=(self.time_steps, self.n_features), return_sequences=True))
        model.add(LSTM(16, activation='relu', return_sequences=False))
        model.add(RepeatVector(self.time_steps))  # 将输入重复n次，n为滑动窗口步长
        model.add(LSTM(16, activation='relu', return_sequences=True))
        model.add(LSTM(32, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(self.n_features)))
        model.summary()
        return model

    def model_train(self):
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        history = self.model.fit(self.train_data, self.train_data, epochs=self.epoch, batch_size=self.batch,
                            validation_data=(self.valid_data, self.valid_data))
        self.model.save(r'..\..\model\Lstm_Auto_Encoder.h5')
        return history

LAET = LSTM_Auto_Encoder_Train()
history = LAET.model_train()


