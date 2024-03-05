import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import RobustScaler
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, LSTM, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout,multiply
from keras.models import Model

class Fault_Identif_Train():
    def __init__(self):
        self.filePath = r"..\..\data\all_data"  # 数据所在的文件夹
        self.data_class = np.array(os.listdir(self.filePath))   # 获得所有故障的名称

        # 故障数据路径
        self.SF_Fault_Path = os.path.join(self.filePath,"sf")   # 松浮故障数据路径
        self.RQS_Fault_Path = os.path.join(self.filePath, "rqs")  # 舵面缺损故障数据路径
        self.KS_Fault_Path = os.path.join(self.filePath, "ks")  # 舵面卡死故障数据路径
        self.LQS_Fault_Path = os.path.join(self.filePath, "lqs") # 升力面缺损故障数据
        self.T_Fault_Path = os.path.join(self.filePath, "T")  # 推力下降数据数据路径

        # 提取数据
        self.class_num = 9
        self.param = ['dOmega_ib_b[0]', 'dOmega_ib_b[1]', 'dOmega_ib_b[2]', 'Gama', 'Theta_k', 'Psi_t', 'fb[0]',
                      'fb[1]','fb[2]', 'Alfa', 'Beta', 'zmb', 'P']     # 需要提取的列
        self.scaler = RobustScaler().fit(self.read_dat(r"..\..\data\normal\zc1.dat")[3000:10000, :])
        self.lookback = 5
        self.label = {"sf":0,"rqs":1,"ks":2,"T":3,"lqs":4}
        # 获取数据集和标签
        # self.SF_Data, self.SF_Label = self.get_Train_data(self.SF_Fault_Path, "sf")
        # self.KS_Data, self.KS_Label = self.get_Train_data(self.KS_Fault_Path, "ks")
        # self.RQS_Data, self.RQS_Label = self.get_Train_data(self.RQS_Fault_Path, "rqs")
        # self.LQS_Data, self.LQS_Label= self.get_Train_data(self.LQS_Fault_Path, "lqs")
        # self.T_Data, self.T_Label = self.get_Train_data(self.T_Fault_Path, "T")          # 第一次需要执行，后面可以直接读取npy文件，节约时间

        self.SF_Data = np.load(r"..\..\data\all_data\hhsdata\sf.npy")
        self.KS_Data = np.load(r"..\..\data\all_data\hhsdata\ks.npy")
        self.RQS_Data = np.load(r"..\..\data\all_data\hhsdata\rqs.npy")
        self.LQS_Data = np.load(r"..\..\data\all_data\hhsdata\lqs.npy")
        self.T_Data = np.load(r"..\..\data\all_data\hhsdata\T.npy")

        self.SF_Label = [0]*len(self.SF_Data)
        self.KS_Label = [2]*len(self.KS_Data)
        self.RQS_Label = [1]*len(self.RQS_Data)
        self.LQS_Label = [4]*len(self.LQS_Data)
        self.T_Label = [3]*len(self.T_Data)

        self.train_data_pre = np.concatenate((self.SF_Data, self.KS_Data, self.RQS_Data, self.LQS_Data, self.T_Data), axis=0)
        self.LAET = load_model(r"..\..\model\Lstm_Auto_Encoder.h5")    # 加载LSTM自编码器模型用于残差生成
        # self.train_data_real = self.residual()                         # 训练集数据
        self.train_data_real = np.load(r"..\..\data\all_data\hhsdata\residual.npy")
        self.train_label = to_categorical(
            self.SF_Label * 12 + self.KS_Label * 12 + self.RQS_Label * 12 + self.LQS_Label * 12 + self.T_Label * 12,
            num_classes=9)                                             # 训练集标签
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.train_data_real, self.train_label, test_size=0.30, random_state=42)
        self.model = self.get_mode()   # 获得模型

    # 根据路径提取数据
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
        train_data = np.array([self.temporalize(self.scaler.transform(self.read_dat(os.path.join(os.path.join(os.path.join(self.filePath,Path),dic),data_id))[2000:5200,:])) for dic in os.listdir(Path) for data_id in os.listdir(os.path.join(os.path.join(self.filePath, Path), dic))])
        np.save(r"..\..\data\all_data\hhsdata\{}.npy".format(flag),train_data)  # 数据保存
        return train_data,[self.label[flag]]*(len(train_data))

    # 将3维LSTM数据展平为2维
    def data_flatten(self,X):
        flattened_X = np.array([X[i, (X.shape[1] - 1), :] for i in range(X.shape[0])])
        return flattened_X

    # 利用LSTM自编码器生成残差
    def residual(self):
        XX = [self.data_flatten(real_data)-self.data_flatten(self.LAET.predict(real_data)) for real_data in self.train_data_pre]
        X = np.array([residual[3007+j:3034+j,:].T for residual in XX for j in range(12)])
        np.save(r"..\..\data\all_data\hhsdata\residual.npy", X)  # 保存训练集残差
        return X

    def squeeze_excite_block(self,input):
        ''' Create a squeeze-excite block
        Args:
            input: input tensor
            filters: number of output filters
            k: width factor

        Returns: a keras tensor
        '''
        filters = input.shape[-1]  # channel_axis = -1 for TF
        se = GlobalAveragePooling1D()(input)
        se = Reshape((1, filters))(se)
        se = Dense(filters // 16, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
        se = multiply([input, se])
        return se
    # 获得模型
    def get_mode(self):
        ip = Input(shape=(self.train_data_real.shape[1], self.train_data_real.shape[2]))
        x = Masking()(ip)
        x = LSTM(64)(x)
        x = Dropout(0.8)(x)
        y = Permute((2, 1))(ip)
        y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = self.squeeze_excite_block(y)
        y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = self.squeeze_excite_block(y)
        y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = GlobalAveragePooling1D()(y)
        x = concatenate([x, y])
        out = Dense(self.class_num, activation='softmax')(x)
        model = Model(ip, out)
        model.summary()
        return model

    def model_train(self):
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = self.model.fit(self.X_train, self.Y_train, validation_data=(self.X_test, self.Y_test), batch_size=64,
                            epochs=100)
        self.model.save(r'..\..\model\Fault_Identif_Train.h5')
        return history

a = Fault_Identif_Train()
a.model_train()

