import numpy as np
import os
from sklearn.preprocessing import RobustScaler
from keras.models import load_model
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.layers import Input, Dense, LSTM, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout,multiply
from keras.models import Model

class Fault_Location_Train():
    def __init__(self):
        # 数据获取
        self.filePath = r"..\..\data\all_data"  # 数据所在的文件夹
        self.SF_Data = np.load(r"..\..\data\all_data\hhsdata\sf.npy")
        self.KS_Data = np.load(r"..\..\data\all_data\hhsdata\ks.npy")
        self.RQS_Data = np.load(r"..\..\data\all_data\hhsdata\rqs.npy")
        self.LQS_Data = np.load(r"..\..\data\all_data\hhsdata\lqs.npy")    # 必须先运行Fault_Identif_Train程序才行，该数据已经归一化且滑动完成

        self.param = ['dOmega_ib_b[0]', 'dOmega_ib_b[1]', 'dOmega_ib_b[2]', 'Gama', 'Theta_k', 'Psi_t', 'fb[0]',
                      'fb[1]', 'fb[2]', 'Alfa', 'Beta', 'zmb', 'P']          # 需要提取的列
        self.scaler = RobustScaler().fit(self.read_dat(r"..\..\data\normal\zc1.dat")[3000:10000, :])
        self.lookback = 5
        self.LAET = load_model(r"..\..\model\Lstm_Auto_Encoder.h5")  # 加载LSTM自编码器模型用于残差生成
        self.due = 10
        self.SF_Label = to_categorical(self.get_label("sf").repeat(self.due), num_classes=4)
        self.KS_Label = to_categorical(self.get_label("ks").repeat(self.due), num_classes=4)
        self.RQS_Label = to_categorical(self.get_label("rqs").repeat(self.due), num_classes=4)
        self.LQS_Label = to_categorical(self.get_label("lqs").repeat(self.due), num_classes=2)

        # self.SF_Train = self.residual(self.SF_Data,"sf")
        # self.KS_Train = self.residual(self.KS_Data, "ks")
        # self.RQS_Train = self.residual(self.RQS_Data, "rqs")
        # self.LQS_Train = self.residual(self.LQS_Data, "lqs")
        self.SF_Train = np.load(r"..\..\data\all_data\hhsdata\residual_location_sf.npy")
        self.KS_Train = np.load(r"..\..\data\all_data\hhsdata\residual_location_ks.npy")
        self.RQS_Train = np.load(r"..\..\data\all_data\hhsdata\residual_location_rqs.npy")
        self.LQS_Train = np.load(r"..\..\data\all_data\hhsdata\residual_location_lqs.npy")

        self.SF_Model = self.get_model(self.SF_Train,4)
        self.KS_Model = self.get_model(self.KS_Train,4)
        self.RQS_Model = self.get_model(self.RQS_Train,4)
        self.LQS_Model = self.get_model(self.LQS_Train,2)   # 获得模型

    # 根据路径提取数据
    def read_dat(self, path):
        a = pd.read_csv(path, delim_whitespace=True)
        return np.array(a[self.param], dtype=float)

    def get_label(self, flag):  # 根据flag获取标签
        dic = os.listdir(r"..\..\data\all_data\{}".format(flag))
        label = [loc for loc in range(len(dic)) for j in os.listdir(r"..\..\data\all_data\{}\{}".format(flag,dic[loc]))]
        return np.array(label)

    # 将3维LSTM数据展平为2维
    def data_flatten(self,X):
        flattened_X = np.array([X[i, (X.shape[1] - 1), :] for i in range(X.shape[0])])
        return flattened_X

    # 利用LSTM自编码器生成残差
    def residual(self,X_data,Flag):
        XX = [self.data_flatten(real_data)-self.data_flatten(self.LAET.predict(real_data)) for real_data in X_data]
        X = np.array([residual[3007+j:3034+j,:].T for residual in XX for j in range(self.due)])
        np.save(r"..\..\data\all_data\hhsdata\residual_location_{}.npy".format(Flag), X)  # 保存训练集残差
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

    # 输入为训练的数据和类别数量
    def get_model(self,train_data,class_num):
        ip = Input(shape=(train_data.shape[1], train_data.shape[2]))
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
        out = Dense(class_num, activation='softmax')(x)
        model = Model(ip, out)
        model.summary()
        return model

    def model_train(self,model,train_data,train_flag,flag):
        X_train, X_test, Y_train, Y_test = train_test_split(train_data, train_flag, test_size=0.30, random_state=42)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=50)
        model.save(r'..\..\model\Fault_Location_{}.h5'.format(flag))  # 保存模型
        return history

    def run(self):
        self.model_train(self.SF_Model,self.SF_Train,self.SF_Label,"sf")
        self.model_train(self.KS_Model, self.KS_Train, self.KS_Label, "ks")
        self.model_train(self.RQS_Model, self.RQS_Train, self.RQS_Label, "rqs")
        self.model_train(self.LQS_Model, self.LQS_Train, self.LQS_Label, "lqs")

if __name__ =="__main__":
    a = Fault_Location_Train()
    a.run()