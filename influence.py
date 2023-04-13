import os

import tensorflow as tf
import scipy.io as scio
import numpy as np
from matplotlib import colors, colorbar, cm

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape, RepeatVector, \
    TimeDistributed, Lambda
from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential, load_model

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, precision_recall_curve
import seaborn as sns
import pandas as pd
import time
import matplotlib.ticker as mtick

path_project = '/home/yukina/Missle_Fault_Detection/project/'


def data_preprocess(path1):
    rty = scio.loadmat(path1)
    rty = rty['RTY_XYZ']
    data = rty[:, :11]
    data = np.concatenate((data, rty[:, 15:16]), axis=1)
    data = np.array(data, dtype='float16')
    return data


def getnormaldata(path):
    rty = scio.loadmat(path)
    rty = rty['RTY_XYZ']
    data = rty[:, :11]
    data = np.concatenate((data, rty[:, 15:16]), axis=1)
    data = np.array(data, dtype='float16')
    return data


def get_train_data():
    id = ['1stuck-5', '1stuck-3', '1stuck1', '1stuck-1', '2stuck-13', '2stuck-7', '2stuck-3', '2stuck-1']
    train_data = []
    scaler = RobustScaler().fit(normal_data[3000:8000])
    for i in range(len(id)):
        path = 'F:\微信文件\研一上大作业\DD\demoV9-7.5_20220706check\data\{}.mat'.format(id[i])
        data = data_preprocess(path)
        fault_start = 8000
        fault_data = data[fault_start - 5:fault_start + 20]
        xtrain = scaler.transform(fault_data)
        train_data.append(xtrain)
    label = [0, 0, 0, 0, 1, 1, 1, 1]
    mode = to_categorical(label, num_classes=2)
    return train_data, mode


def get_test_data():
    path = 'F:\微信文件\研一上大作业\DD\demoV9-7.5_20220706check\data\\2stuck-5.mat'
    data = data_preprocess(path)
    fault_start = 8000
    fault_data = data[fault_start - 5:fault_start + 20]
    scaler = RobustScaler().fit(normal_data[3000:8000, :])
    xtest = scaler.transform(fault_data)
    test_data = xtest
    return test_data


def data_flatten(X):
    '''
    Flatten a 3D array.
    Input
    X            A 3D array for lstm, where the array is sample x timesteps x features.
    Output
    flattened_X  A 2D array, sample x features.
    '''
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1] - 1), :]
    return (flattened_X)


def temporalize(X, lookback):
    output_X = []
    for i in range(len(X) - lookback - 1):
        t = []
        for j in range(1, lookback + 1):
            # Gather past records upto the lookback period
            t.append(X[[(i + j + 1)], :])
        output_X.append(t)
    return output_X


def get_dat(path):
    a = pd.read_csv(path, delim_whitespace=True)
    b = a[['dOmega_ib_b[0]', 'dOmega_ib_b[1]', 'dOmega_ib_b[2]', 'Gama', 'Theta_k', 'Psi_t', 'fb[0]', 'fb[1]',
           'fb[2]', 'Alfa', 'Beta', 'zmb', 'P']]
    b = np.array(b, dtype=float)
    return b


def get_ka_data():
    id_index1 = np.arange(-19, 0, 1)
    id_index2 = np.arange(1, 20, 1)
    id1, id2, id3, id4 = [], [], [], []
    for i in range(len(id_index1)):
        name1 = 'ks1_' + str(id_index1[i])
        id1.append(name1)
        name2 = 'ks2_' + str(id_index2[i])
        id2.append(name2)
        name3 = 'ks3_' + str(id_index2[i])
        id3.append(name3)
        name4 = 'ks4_' + str(id_index1[i])
        id4.append(name4)
    path3 = path_project + 'raw/normal/zc1.dat'
    normal_data = get_dat(path3)
    alltest = []
    scaler = RobustScaler().fit(normal_data[3000:10000, :])
    print(id1)
    train_end = 2000
    path_ks = path_project + 'raw/ks/'
    for i in range(len(id1)):
        path = path_ks + '1/{}.dat'.format(id1[i])
        data = get_dat(path)
        testdata = data[train_end:5200, :]
        xtest = scaler.transform(testdata)
        lookback = 5
        test_extract = np.array(temporalize(xtest, lookback))
        test_extract = np.reshape(test_extract, (test_extract.shape[0], lookback, test_extract.shape[3]))
        alltest.append(test_extract)
    for i in range(len(id2)):
        path = path_ks + '2/{}.dat'.format(id2[i])
        data = get_dat(path)
        testdata = data[train_end:5200, :]
        xtest = scaler.transform(testdata)
        lookback = 5
        test_extract = np.array(temporalize(xtest, lookback))
        test_extract = np.reshape(test_extract, (test_extract.shape[0], lookback, test_extract.shape[3]))
        alltest.append(test_extract)
    for i in range(len(id3)):
        path = path_ks + '3/{}.dat'.format(id3[i])
        data = get_dat(path)
        testdata = data[train_end:5200, :]
        xtest = scaler.transform(testdata)
        lookback = 5
        test_extract = np.array(temporalize(xtest, lookback))
        test_extract = np.reshape(test_extract, (test_extract.shape[0], lookback, test_extract.shape[3]))
        alltest.append(test_extract)
    for i in range(len(id4)):
        path = path_ks + '4/{}.dat'.format(id4[i])
        data = get_dat(path)
        testdata = data[train_end:5200, :]
        xtest = scaler.transform(testdata)
        lookback = 5
        test_extract = np.array(temporalize(xtest, lookback))
        test_extract = np.reshape(test_extract, (test_extract.shape[0], lookback, test_extract.shape[3]))
        alltest.append(test_extract)
    label = [2] * (len(id1) * 4)
    alltest = np.array(alltest)
    return np.array(alltest), label


def get_que_data():
    id_index = np.arange(0.2, 10.1, 0.2)
    id_index = np.array(id_index / 10)
    id1, id2, id3, id4 = [], [], [], []
    for i in range(len(id_index)):
        name1 = 'rqs-1-%g' % id_index[i]
        id1.append(name1)
        name2 = 'rqs-2-%g' % id_index[i]
        id2.append(name2)
        name3 = 'rqs-3-%g' % id_index[i]
        id3.append(name3)
        name4 = 'rqs-4-%g' % id_index[i]
        id4.append(name4)
    print(id1)
    path3 = path_project + 'raw/normal/zc1.dat'
    normal_data = get_dat(path3)
    alltest = []
    scaler = RobustScaler().fit(normal_data[3000:10000, :])
    train_end = 2000
    path_qs = path_project + 'raw/rqs/'
    for i in range(len(id1)):
        path = path_qs + 'rqs-1/{}.dat'.format(id1[i])
        data = get_dat(path)
        testdata = data[train_end:5200, :]
        xtest = scaler.transform(testdata)
        lookback = 5
        test_extract = np.array(temporalize(xtest, lookback))
        test_extract = np.reshape(test_extract, (test_extract.shape[0], lookback, test_extract.shape[3]))
        alltest.append(test_extract)
    for i in range(len(id2)):
        path = path_qs + 'rqs-2/{}.dat'.format(id2[i])
        data = get_dat(path)
        testdata = data[train_end:5200, :]
        xtest = scaler.transform(testdata)
        lookback = 5
        test_extract = np.array(temporalize(xtest, lookback))
        test_extract = np.reshape(test_extract, (test_extract.shape[0], lookback, test_extract.shape[3]))
        alltest.append(test_extract)
    for i in range(len(id3)):
        path = path_qs + 'rqs-3/{}.dat'.format(id3[i])
        data = get_dat(path)
        testdata = data[train_end:5200, :]
        xtest = scaler.transform(testdata)
        lookback = 5
        test_extract = np.array(temporalize(xtest, lookback))
        test_extract = np.reshape(test_extract, (test_extract.shape[0], lookback, test_extract.shape[3]))
        alltest.append(test_extract)
    for i in range(len(id4)):
        path = path_qs + 'rqs-4/{}.dat'.format(id4[i])
        data = get_dat(path)
        testdata = data[train_end:5200, :]
        xtest = scaler.transform(testdata)
        lookback = 5
        test_extract = np.array(temporalize(xtest, lookback))
        test_extract = np.reshape(test_extract, (test_extract.shape[0], lookback, test_extract.shape[3]))
        alltest.append(test_extract)
    label = [1] * (len(id1) * 4)
    return np.array(alltest), label


def get_song_data():
    id_index = np.arange(1, 6, 1)
    id1, id2, id3, id4 = [], [], [], []
    for i in range(len(id_index)):
        name1 = 'sf1-' + str(id_index[i])
        id1.append(name1)
        name2 = 'sf20' + str(id_index[i])
        id2.append(name2)
        name3 = 'sf30' + str(id_index[i])
        id3.append(name3)
        name4 = 'sf4-' + str(id_index[i])
        id4.append(name4)
    path3 = path_project + 'raw/normal/zc1.dat'
    normal_data = get_dat(path3)
    alltest = []
    scaler = RobustScaler().fit(normal_data[3000:10000, :])
    path_sf = path_project + 'raw/sf2_23/'
    for i in range(len(id1)):
        path = path_sf + '1/{}.dat'.format(id1[i])
        data = get_dat(path)
        testdata = data[2000:5200, :]
        xtest = scaler.transform(testdata)
        lookback = 5
        test_extract = np.array(temporalize(xtest, lookback))
        test_extract = np.reshape(test_extract, (test_extract.shape[0], lookback, test_extract.shape[3]))
        alltest.append(test_extract)
    for i in range(len(id2)):
        path = path_sf + '2/{}.dat'.format(id2[i])
        data = get_dat(path)
        testdata = data[2000:5200, :]
        xtest = scaler.transform(testdata)
        lookback = 5
        test_extract = np.array(temporalize(xtest, lookback))
        test_extract = np.reshape(test_extract, (test_extract.shape[0], lookback, test_extract.shape[3]))
        alltest.append(test_extract)
    for i in range(len(id3)):
        path = path_sf + '3/{}.dat'.format(id3[i])
        data = get_dat(path)
        testdata = data[2000:5200, :]
        xtest = scaler.transform(testdata)
        lookback = 5
        test_extract = np.array(temporalize(xtest, lookback))
        test_extract = np.reshape(test_extract, (test_extract.shape[0], lookback, test_extract.shape[3]))
        alltest.append(test_extract)
    for i in range(len(id4)):
        path = path_sf + '4/{}.dat'.format(id4[i])
        data = get_dat(path)
        testdata = data[2000:5200, :]
        xtest = scaler.transform(testdata)
        lookback = 5
        test_extract = np.array(temporalize(xtest, lookback))
        test_extract = np.reshape(test_extract, (test_extract.shape[0], lookback, test_extract.shape[3]))
        alltest.append(test_extract)
    label = [0] * (len(id1) * 4)
    return np.array(alltest), label


def get_th_data():
    id_index = np.arange(0, 1, 0.04)
    id1 = []
    for i in range(len(id_index)):
        name1 = 'T%g' % id_index[i]
        id1.append(name1)
    print(id1)
    path3 = path_project + 'raw/normal/zc1.dat'
    normal_data = get_dat(path3)
    alltest = []
    scaler = RobustScaler().fit(normal_data[3000:10000, :])
    for i in range(len(id1)):
        path = path_project + 'raw/T/{}.dat'.format(id1[i])
        data = get_dat(path)
        testdata = data[2000:5200, :]
        xtest = scaler.transform(testdata)
        lookback = 5
        test_extract = np.array(temporalize(xtest, lookback))
        test_extract = np.reshape(test_extract, (test_extract.shape[0], lookback, test_extract.shape[3]))
        alltest.append(test_extract)
    label = [3] * len(id1)
    return np.array(alltest), label


def get_ls_data():
    id_index = np.arange(0.02, 1.02, 0.02)
    id1, id2 = [], []
    for i in range(len(id_index)):
        name1 = 'lqs-l-%g' % id_index[i]
        id1.append(name1)
        name2 = 'lqs-r-%g' % id_index[i]
        id2.append(name2)
    path3 = path_project + 'raw/normal/zc1.dat'
    normal_data = get_dat(path3)
    alltest = []
    scaler = RobustScaler().fit(normal_data[3000:10000, :])
    for i in range(len(id1)):
        path = path_project + 'raw/lqs-l/{}.dat'.format(id1[i])
        data = get_dat(path)
        testdata = data[2000:5200, :]
        xtest = scaler.transform(testdata)
        lookback = 5
        test_extract = np.array(temporalize(xtest, lookback))
        test_extract = np.reshape(test_extract, (test_extract.shape[0], lookback, test_extract.shape[3]))
        alltest.append(test_extract)
    for i in range(len(id2)):
        path = path_project + 'raw/lqs-r/{}.dat'.format(id2[i])
        data = get_dat(path)
        testdata = data[2000:5200, :]
        xtest = scaler.transform(testdata)
        lookback = 5
        test_extract = np.array(temporalize(xtest, lookback))
        test_extract = np.reshape(test_extract, (test_extract.shape[0], lookback, test_extract.shape[3]))
        alltest.append(test_extract)
    label = [4] * (len(id1) * 2)
    return np.array(alltest), label


def trainmlstmclassify(alltest):
    fea = []
    model1 = load_model(path_project + 'models/location.h5')
    for i in range(alltest.shape[0]):
        pred1 = model1.predict(alltest[i])
        test_pred = data_flatten(pred1)
        test_real = data_flatten(alltest[i])
        test_error = test_real - test_pred
        print(test_error.shape)
        # test_error = np.array(test_error)
        # cc = np.arange(2993: 3020, 10)
        for j in range(12):
            error = np.array(test_error[3007 + j: 3034 + j, :])
            fea.append(error.T)
    print(np.shape(fea))
    fea = np.array(fea)
    return fea


def generate_lstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS):
    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))  # tensor
    x = LSTM(NUM_CELLS)(ip)
    x = Dropout(0.8)(x)
    y = Permute((2, 1))(
        ip)  # permutes the first and second dimension of the input (connecting RNNs and convnets together)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)  # for temporal convolution
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = GlobalAveragePooling1D()(y)
    x = concatenate([x, y])
    out = Dense(NB_CLASS, activation='softmax')(x)
    model = Model(ip, out)
    model.summary()
    return model


def generate_model(MAX_NB_VARIABLES, NB_CLASS, MAX_TIMESTEPS):
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))
    x = Masking()(ip)
    x = LSTM(64)(x)
    x = Dropout(0.8)(x)
    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)
    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)
    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = GlobalAveragePooling1D()(y)
    # x = concatenate([x, y])
    out = Dense(NB_CLASS, activation='softmax')(y)
    model = Model(ip, out)
    model.summary()
    # add load model code here to fine-tune

    return model


def squeeze_excite_block(input):
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


def plot_acc(history, ax):
    if 'val_accuracy' in history.keys():
        ax.grid(which='major', ls='--', alpha=.8, lw=.8)
        # plt.plot(history.history['accuracy'])
        # plt.plot(history.history['val_accuracy'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        # summarize history for loss
        ax.plot(history['loss'], label='训练集')
        ax.plot(history['val_loss'], label='验证集')
        # plt.title('model loss')
        xticks = ax.get_xticks()
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, fontdict={'family': 'Times New Roman', 'size': 10})
        # ax.set_xlim(-40, 1040)
        yticks = ax.get_yticks()
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontdict={'family': 'Times New Roman', 'size': 10})
        ax.set_xlabel('训练轮数', fontdict={'family': ['SimSun'], 'size': 12})
        ax.set_ylabel('Loss', fontdict={'family': 'Times New Roman', 'size': 12})
        ax.legend(loc='best', prop={'family': ['SimSun'], 'size': 12})
        # ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=2, prop={'family': ['SimSun'], 'size': 12})
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    else:
        plt.plot(history.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()


def cof(model, fea, label, ax):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    train_pred = model.predict(fea)
    train_pred = np.argmax(train_pred, axis=1)
    con_mat = confusion_matrix(label, train_pred)
    con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]  # 归一化
    con_mat_norm = np.around(con_mat_norm, decimals=2)
    plt.figure(figsize=(6, 6))
    print(con_mat)
    LABELS = ['松浮故障', '缺损故障', '卡死故障']
    sns.heatmap(con_mat_norm, xticklabels=LABELS, yticklabels=LABELS, annot=True)

    x1_label = ax.get_xticklabels()
    [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    y1_label = ax.get_yticklabels()
    [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
    ax.tick_params(labelsize=12)  # y轴字体大小设置)

    # plt.ylabel('实际标签', fontdict={'family': ['SimSun'], 'size': 12})
    # plt.xlabel('预测分类', fontdict={'family': ['SimSun'], 'size': 12})
    plt.show()
    print('a', accuracy_score(label, train_pred))
    print('f', f1_score(label, train_pred, average='macro'))
    print('r', recall_score(label, train_pred, average='macro'))


def trainaemodel():
    path1 = path_project + 'raw/normal/zc1.dat'
    normal_data = get_dat(path1)
    alltrain = []
    for i in range(2):
        path2 = path_project + 'raw/normal/{}.dat'.format('zc' + str(i + 2))
        data = get_dat(path2)
        traindata = data[3000:10000, :]
        scaler = RobustScaler().fit(normal_data[3000:10000, :])
        xtrain = scaler.transform(traindata)
        lookback = 5
        train_extract = np.array(temporalize(xtrain, lookback))
        train_extract = np.reshape(train_extract, (train_extract.shape[0], lookback, train_extract.shape[3]))
        alltrain.append(train_extract)
    train_data = np.vstack(alltrain)
    xx = normal_data[3000:10000, :]
    xxx = RobustScaler().fit_transform(xx)
    lookback = 5
    normal_extract = np.array(temporalize(xxx, lookback))
    normal_extract = np.reshape(normal_extract, (normal_extract.shape[0], lookback, normal_extract.shape[3]))
    epochs, batch, lr = 100, 64, 0.0001
    time_steps = train_data.shape[1]
    n_features = train_data.shape[2]
    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(time_steps, n_features), return_sequences=True))
    model.add(LSTM(16, activation='relu', return_sequences=False))
    model.add(RepeatVector(time_steps))
    model.add(LSTM(16, activation='relu', return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    model.summary()
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_data, train_data, epochs=epochs, batch_size=batch,
                        validation_data=(normal_extract, normal_extract))
    model.save(path_project + 'models/location.h5')
    return history


def get_processed_data():
    fea_path = path_project + 'data/cla_fea.npy'
    label_path = path_project + 'data/cla_label.npy'
    # 判断数据是否存在
    if os.path.exists(fea_path) and os.path.exists(label_path):
        fea = np.load(fea_path)
        label = np.load(label_path)
    else:
        songtest, songlabel = get_song_data()
        quetest, quelabel = get_que_data()
        katest, kalabel = get_ka_data()
        thtest, thlabel = get_th_data()
        lstest, lslabel = get_ls_data()
        # print(np.shape(songtest),np.shape(quetest),np.shape(katest),np.shape(thtest),np.shape(lstest))
        np.save(path_project + 'data/songtest.npy', songtest)
        np.save(path_project + 'data/quetest.npy', quetest)
        np.save(path_project + 'data/katest.npy', katest)
        np.save(path_project + 'data/thtest.npy', thtest)
        np.save(path_project + 'data/lstest.npy', lstest)
        #
        # songtest = np.load('D:\share\新建文件夹\data\songtest.npy')
        # quetest = np.load('D:\share\新建文件夹\data\quetest.npy')
        # katest = np.load('D:\share\新建文件夹\data\katest.npy')
        # thtest = np.load('D:\share\新建文件夹\data\\thtest.npy')
        # lstest = np.load('D:\share\新建文件夹\data\lstest.npy')
        print(np.shape(songtest), np.shape(quetest), np.shape(katest), np.shape(thtest), np.shape(lstest))
        # print(np.shape(songlabel),np.shape(quelabel),np.shape(kalabel),np.shape(thlabel),np.shape(lslabel))
        alltest = np.concatenate((songtest, quetest, katest, thtest, lstest), axis=0)
        print(alltest.shape)
        # print(label)
        # mode = to_categorical(label,num_classes=5).0
        if not os.path.exists(path_project + 'models/location.h5'):
            ae_history = trainaemodel()
        fea = trainmlstmclassify(alltest)
        np.save(path_project + 'data/cla_fea.npy', fea)
        # np.save('D:\share\新建文件夹\data\\cla_mode.npy',label)

        # fea = np.load('D:\share\新建文件夹\data\\cla_fea.npy')
        print(np.shape(fea))
        # id1 = np.arange(-19, 2, 3)
        # kalabel = [2] * (len(id1) * 4)
        # id1 = np.arange(0.2, 10.1, 0.2)
        # quelabel = [1] * (len(id1) * 4)
        # id1 = np.arange(1, 6, 1)
        # songlabel = [0] * (len(id1) * 4)
        # id1 = np.arange(0, 1, 0.04)
        # thlabel = [3] * len(id1)
        # id1 = np.arange(0.02, 1.02, 0.02)
        # lslabel = [4] * (len(id1) * 2)
        label = songlabel * 12 + quelabel * 12 + kalabel * 12 + thlabel * 12 + lslabel * 12
        np.save(path_project + 'data/cla_label.npy', label)
        print(np.shape(label))
    return fea, label


if __name__ == "__main__":
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # # path = 'D:\share\\normal\\zc1.dat'
    # normal_data = get_dat(path)

    # label = np.load('F:\微信文件\研一上大作业\DD\python\detect\data\classify\\cla_mode.npy')

    # fea, label = get_processed_data()
    # mode = to_categorical(label, num_classes=5)
    ''''''
    # X_train = np.reshape(fea, (fea.shape[0], 1, fea.shape[1]))
    # print(fea.shape)
    # model = generate_lstmfcn(fea.shape[1],3, 8)
    # X_train = fea
    ''''''

    # X_train, X_test, Y_train, Y_test = train_test_split(fea, mode, test_size=0.30, random_state=i)
    # # Save the training set and test set
    # np.save(path_project + 'data/X_train.npy', X_train)
    # np.save(path_project + 'data/X_test.npy', X_test)
    # np.save(path_project + 'data/Y_train.npy', Y_train)
    # np.save(path_project + 'data/Y_test.npy', Y_test)

    # Load the training set and test set
    seed = 0
    X_train = np.load(f'data_seed={seed}/X_train.npy')
    X_test = np.load(f'data_seed={seed}/X_test.npy')
    Y_train = np.load(f'data_seed={seed}/Y_train.npy')
    Y_test = np.load(f'data_seed={seed}/Y_test.npy')
    ID_train = np.load(f'data_seed={seed}/ID_train.npy')
    ID_test = np.load(f'data_seed={seed}/ID_test.npy')

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).map(
        lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)))

    # load the model
    # if os.path.exists(f'data_seed={seed}/lstm-fcn-{seed}.h5'):
    #     model = load_model(f'data_seed={seed}/lstm-fcn-{seed}.h5')
    if os.path.exists(f'models/lstm-fcn-withoutCNN.h5'):
        model = load_model(f'models/lstm-fcn-withoutCNN.h5')
    else:
        assert False, 'No model found!'

    if os.path.exists(f'data_seed={seed}/Prediction_train.npy'):
        Prediction_train = np.load(f'data_seed={seed}/Prediction_train.npy')
    else:
        Prediction_train = model.predict(X_train)
        np.save(f'data_seed={seed}/Prediction_train.npy', Prediction_train)

    # 找出测试集中预测错误的样本
    test_pred = model.predict(X_test)
    test_pred = np.argmax(test_pred, axis=1)
    label_test = np.argmax(Y_test, axis=1)
    wrong_id = np.where(test_pred != label_test)[0]
    test_ds = tf.data.Dataset.from_tensor_slices((X_test[wrong_id], Y_test[wrong_id])).map(
        lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)))

    from deel_modified.influenciae.common import InfluenceModel, ExactIHVP, ConjugateGradientDescentIHVP
    from tensorflow.keras.losses import BinaryCrossentropy, Reduction

    # sequential_model = Sequential()
    # for layer in model.layers:
    #     sequential_model.add(layer)

    influence_model = InfluenceModel(
        model,
        start_layer=-1,
        last_layer=len(model.layers) - 1,
        loss_function=tf.keras.losses.BinaryCrossentropy(from_logits=False,
                                                         reduction=Reduction.NONE))
    print(f'weights list: {len(influence_model.weights)}')
    model.summary()
    from deel_modified.influenciae.influence import FirstOrderInfluenceCalculator

    # ihvp_calculator = ExactIHVP(model=influence_model, train_dataset=train_ds.batch(1))
    identity_model = Sequential([
        Input(shape=(13, 27))
    ])
    ihvp_calculator = ConjugateGradientDescentIHVP(model=influence_model, extractor_layer=0, train_dataset=train_ds.batch(1), feature_extractor=identity_model)
    influence_calculator = FirstOrderInfluenceCalculator(model=influence_model, dataset=train_ds,
                                                         ihvp_calculator=ihvp_calculator)
    tesk_point = test_ds.take(1).batch(1)
    explanations = influence_calculator.estimate_influence_values_in_batches(tesk_point, train_ds.batch(1))
    # explanations = influence_calculator.top_k(tesk_point, train_ds.take(10).batch(1), k=5)
    for (test_fea, test_label), explanation in explanations:
        fea = test_fea.numpy()
        # fea的维度为(1, 13, 27), 绘制test_fea为表格数据, 并在图上标出真实标签和预测标签
        fea = fea.reshape((fea.shape[1], fea.shape[2]))
        fea = np.transpose(fea)

        # 收集样本的影响力数据
        explanation_dict = {
            'sample_id': [],
            'train_feature': [],
            'train_label': [],
            'predicted label': [],
            'influence': []
        }
        for id, ((train_fea, train_label), influence) in enumerate(explanation):
            # 确保经过影响力计算后训练样本的id没有发生改变
            assert np.argmax(np.squeeze(train_label.numpy())) == np.argmax(Y_train[id])

            explanation_dict['sample_id'].append(int(ID_train[id]))
            explanation_dict['train_feature'].append(np.squeeze(train_fea.numpy()))
            explanation_dict['train_label'].append(np.squeeze(train_label.numpy()))
            explanation_dict['predicted label'].append(np.argmax(Prediction_train[id]))
            explanation_dict['influence'].append(np.squeeze(influence.numpy()))
        # 将影响力数据转换为DataFrame
        explanation_df = pd.DataFrame(explanation_dict)
        # 将影响力数据按照影响力大小排序
        explanation_df = explanation_df.sort_values(by='influence', ascending=False)

        # 收集所有绘图数据的最小值和最大值，以画出colorbar
        min_list = [np.min(test_fea)]
        max_list = [np.max(test_fea)]
        for i in range(0, 5):
            min_list.append(np.min(explanation_df.iloc[i]['train_feature']))
            min_list.append(np.min(explanation_df.iloc[-i - 1]['train_feature']))
            min_list.append(np.min(explanation_df.iloc[int(X_train.shape[0] / 2) + i]['train_feature']))
            max_list.append(np.max(explanation_df.iloc[i]['train_feature']))
            max_list.append(np.max(explanation_df.iloc[-i - 1]['train_feature']))
            max_list.append(np.max(explanation_df.iloc[int(X_train.shape[0] / 2) + i]['train_feature']))
        print(f'min_list: {min_list}')
        print(f'max_list: {max_list}')

        # 创建统一的colorbar
        norm = colors.Normalize(vmin=min(min_list), vmax=max(max_list))
        fig, ax = plt.subplots(4, 5, figsize=(15, 16))
        im = ax[0][0].imshow(fea, cmap='coolwarm', interpolation='nearest', norm=norm)
        fig.colorbar(im, ax=ax[0][0])

        ax[0][0].set_title('Test Sample')
        ax[0][0].set_xticks(np.arange(0, 13, 2))
        ax[0][0].set_yticks(np.arange(0, 27, 2))
        ax[0][0].set_xlabel('feature')
        ax[0][0].set_ylabel('timestep')
        ax[0][0].text(x=6, y=-4,
                      s=f'true label: {np.argmax(test_label)}, predicted label: {np.argmax(model.predict(test_fea))}',
                      ha='center', va='center')

        for i in range(1, 5):
            ax[0][i].axis('off')

        for i in range(0, 5):
            # 绘制影响力最大的前5个样本
            im = ax[1][i].imshow(np.transpose(explanation_df.iloc[i]['train_feature']), cmap='coolwarm', interpolation='nearest', norm=norm)
            fig.colorbar(im, ax=ax[1][i])
            ax[1][i].set_title(
                f'id:{explanation_df.iloc[i]["sample_id"]}, influence: {explanation_df.iloc[i]["influence"]:.3f} ')
            ax[1][i].set_xticks(np.arange(0, 13, 2))
            ax[1][i].set_yticks(np.arange(0, 27, 2))
            ax[1][i].set_xlabel('feature')
            ax[1][i].set_ylabel('timestep')
            ax[1][i].text(x=6, y=-4,
                          s=f'label: {np.argmax(explanation_df.iloc[i]["train_label"])}, predict: {explanation_df.iloc[i]["predicted label"]}, impact:{-explanation_df.iloc[i]["influence"]/X_train.shape[0]:.3f}',
                          ha='center', va='center')

            # 绘制影响力最小的前5个样本
            im = ax[2][i].imshow(np.transpose(explanation_df.iloc[-i - 1]['train_feature']), cmap='coolwarm', interpolation='nearest', norm=norm)
            fig.colorbar(im, ax=ax[2][i])
            ax[2][i].set_title(
                f'id:{explanation_df.iloc[-i - 1]["sample_id"]}, influence: {explanation_df.iloc[-i - 1]["influence"]:.3f} ')
            ax[2][i].set_xticks(np.arange(0, 13, 2))
            ax[2][i].set_yticks(np.arange(0, 27, 2))
            ax[2][i].set_xlabel('feature')
            ax[2][i].set_ylabel('timestep')
            ax[2][i].text(x=6, y=-4,
                          s=f'label: {np.argmax(explanation_df.iloc[-i - 1]["train_label"])}, predict: {explanation_df.iloc[-i - 1]["predicted label"]}, impact:{-explanation_df.iloc[-i - 1]["influence"]/X_train.shape[0]:.3f}',
                          ha='center', va='center')


            # 绘制影响力居中的5个样本
            im = ax[3][i].imshow(np.transpose(explanation_df.iloc[int(X_train.shape[0]/2) + i]['train_feature']), cmap='coolwarm', interpolation='nearest', norm=norm)
            fig.colorbar(im, ax=ax[3][i])
            ax[3][i].set_title(
                f'id:{explanation_df.iloc[int(X_train.shape[0]/2) + i]["sample_id"]}, influence: {explanation_df.iloc[int(X_train.shape[0]/2) + i]["influence"]:.3f} ')
            ax[3][i].set_xticks(np.arange(0, 13, 2))
            ax[3][i].set_yticks(np.arange(0, 27, 2))
            ax[3][i].set_xlabel('feature')
            ax[3][i].set_ylabel('timestep')
            ax[3][i].text(x=6, y=-4,
                          s=f'label: {np.argmax(explanation_df.iloc[int(X_train.shape[0]/2) + i]["train_label"])}, predict: {explanation_df.iloc[int(X_train.shape[0]/2) + i]["predicted label"]}, impact:{-explanation_df.iloc[int(X_train.shape[0]/2) + i]["influence"] / X_train.shape[0]:.3f}',
                          ha='center', va='center')



        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.4)
        plt.savefig('outputs/top 5 influence.png')
        plt.show()
