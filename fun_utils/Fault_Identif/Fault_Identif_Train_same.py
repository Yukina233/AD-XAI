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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
import seaborn as sns
import pickle

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

        self.gyro_abrupt_Path = os.path.join(self.filePath, "gyro_abrupt")
        self.gyro_slow_Path = os.path.join(self.filePath, "gyro_slow")
        self.altimeter_abrupt_Path = os.path.join(self.filePath, "altimeter_abrupt")
        self.altimeter_slow_Path = os.path.join(self.filePath, "altimeter_slow")

        # 提取数据
        self.class_num = 9
        self.param = ['dOmega_ib_b[0]', 'dOmega_ib_b[1]', 'dOmega_ib_b[2]', 'Gama', 'Theta_k', 'Psi_t', 'fb[0]',
                      'fb[1]','fb[2]', 'Alfa', 'Beta', 'zmb', 'P']     # 需要提取的列
        # self.param = ['Gama', 'Theta_k', 'Psi_t', 'fb[2]', 'Alfa', 'Beta', 'Omega_ib_b[0]', 'Omega_ib_b[2]', 'dOmega_ib_b[0]', 'dOmega_ib_b[2]']
        self.scaler = RobustScaler().fit(self.read_dat(r"..\..\data\normal\zc1.dat")[3000:10000, :])
        self.lookback = 5
        self.label = {"sf":0,"rqs":1,"ks":2,"T":3,"lqs":4,"gyro_abrupt":5,"gyro_slow":6,"altimeter_abrupt":7,"altimeter_slow":8}

        # 获取数据集和标签
        self.SF_Data, self.SF_Label = self.get_Train_data(self.SF_Fault_Path, "sf")
        self.KS_Data, self.KS_Label = self.get_Train_data(self.KS_Fault_Path, "ks")
        self.RQS_Data, self.RQS_Label = self.get_Train_data(self.RQS_Fault_Path, "rqs")
        self.LQS_Data, self.LQS_Label= self.get_Train_data(self.LQS_Fault_Path, "lqs")
        self.T_Data, self.T_Label = self.get_Train_data(self.T_Fault_Path, "T")
        self.gyro_abrupt_Data, self.gyro_abrupt_Label = self.get_Train_data(self.gyro_abrupt_Path, "gyro_abrupt")
        self.gyro_slow_Data, self.gyro_slow_Label = self.get_Train_data(self.gyro_slow_Path, "gyro_slow")
        self.altimeter_abrupt_Data, self.altimeter_abrupt_Label = self.get_Train_data(self.altimeter_abrupt_Path, "altimeter_abrupt")
        self.altimeter_slow_Data, self.altimeter_slow_Label = self.get_Train_data(self.altimeter_slow_Path, "altimeter_slow")  # 第一次需要执行，后面可以直接读取npy文件，节约时间

        # self.SF_Data = np.load(r"..\..\data\all_data\hhsdata\sf.npy").repeat(10,axis=0)
        # self.KS_Data = np.load(r"..\..\data\all_data\hhsdata\ks.npy").repeat(3,axis=0)
        # self.RQS_Data = np.load(r"..\..\data\all_data\hhsdata\rqs.npy")
        # self.LQS_Data = np.load(r"..\..\data\all_data\hhsdata\lqs.npy").repeat(2,axis=0)
        # self.T_Data = np.load(r"..\..\data\all_data\hhsdata\T.npy").repeat(8,axis=0)
        # self.gyro_abrupt_Data = np.load(r"..\..\data\all_data\hhsdata\gyro_abrupt.npy").repeat(20,axis=0)
        # self.gyro_slow_Data = np.load(r"..\..\data\all_data\hhsdata\gyro_slow.npy").repeat(20,axis=0)
        # self.altimeter_abrupt_Data = np.load(r"..\..\data\all_data\hhsdata\altimeter_abrupt.npy").repeat(20,axis=0)
        # self.altimeter_slow_Data = np.load(r"..\..\data\all_data\hhsdata\altimeter_slow.npy").repeat(20,axis=0)

        self.SF_Label = [0]*len(self.SF_Data)
        self.KS_Label = [2]*len(self.KS_Data)
        self.RQS_Label = [1]*len(self.RQS_Data)
        self.LQS_Label = [4]*len(self.LQS_Data)
        self.T_Label = [3]*len(self.T_Data)
        self.gyro_abrupt_Label = [5]*len(self.gyro_abrupt_Data)
        self.gyro_slow_Label = [6]*len(self.gyro_slow_Data)
        self.altimeter_abrupt_Label = [7]*len(self.altimeter_abrupt_Data)
        self.altimeter_slow_Label = [8]*len(self.altimeter_slow_Data)

        self.train_data_pre = np.concatenate((self.SF_Data, self.KS_Data, self.RQS_Data, self.LQS_Data, self.T_Data, self.gyro_abrupt_Data, self.gyro_slow_Data, self.altimeter_abrupt_Data, self.altimeter_slow_Data), axis=0)
        self.LAET = load_model(r"..\..\model\Lstm_Auto_Encoder.h5")    # 加载LSTM自编码器模型用于残差生成
        self.train_data_real = self.residual()                         # 训练集数据
        # self.train_data_real = np.load(r"..\..\data\all_data\hhsdata\residual.npy")

        self.label = self.SF_Label * 12 + self.KS_Label * 12 + self.RQS_Label * 12 + self.LQS_Label * 12 + self.T_Label * 12+self.gyro_abrupt_Label*12+self.gyro_slow_Label*12+self.altimeter_abrupt_Label*12+self.altimeter_slow_Label*12
        self.train_label = to_categorical(
            self.SF_Label * 12 + self.KS_Label * 12 + self.RQS_Label * 12 + self.LQS_Label * 12 + self.T_Label * 12+self.gyro_abrupt_Label*12+self.gyro_slow_Label*12+self.altimeter_abrupt_Label*12+self.altimeter_slow_Label*12,num_classes=9)                                             # 训练集标签
        print([np.argmax(j) for j in self.train_label])
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

    def cof(self,model):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        train_pred = model.predict(self.X_test)
        train_pred = np.argmax(train_pred, axis=1)
        con_mat = confusion_matrix([np.argmax(j) for j in self.Y_test], train_pred)
        con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]  # 归一化
        con_mat_norm = np.around(con_mat_norm, decimals=2)
        plt.figure(figsize=(10, 10))
        print(con_mat)
        LABELS = ['松浮故障', '缺损故障', '卡死故障','升力面缺损',"推力损失","陀螺仪突变故障","陀螺仪缓变故障","气压计突变故障","气压计缓变故障"]
        sns.heatmap(con_mat_norm, xticklabels=LABELS, yticklabels=LABELS, annot=True)

        plt.show()
        print('a', accuracy_score([np.argmax(j) for j in self.Y_test], train_pred))
        print('f', f1_score([np.argmax(j) for j in self.Y_test], train_pred, average='macro'))
        print('r', recall_score([np.argmax(j) for j in self.Y_test], train_pred, average='macro'))
        # 0.9987843792736666  a
        # 0.9988423897986536  f
        # 0.998755057578587   r

a = Fault_Identif_Train()
#a.model_train()
a.cof(load_model(r"..\..\model\Fault_Identif_Train.h5"))
# history = a.model_train()


# with open('trainHistoryDict.txt', 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# with open('trainHistoryDict.txt','rb') as file_pi:
#     history=pickle.load(file_pi)
# print(history)
# Loss = history['loss']
# val_loss = history['val_loss']
#
# print(1)
# plt.grid(which='major', ls='--', alpha=.8, lw=.8)
# plt.plot(history['loss'], label='训练集Loss')
# plt.plot(history['val_loss'], label='验证集Loss')
#
# plt.xlabel("训练轮数")
# plt.ylabel("Loss")
# plt.legend(prop={'family':'SimHei','size':15})
#
# plt.show()


