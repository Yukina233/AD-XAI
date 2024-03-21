import os

import tensorflow as tf
import scipy.io as scio
import numpy as np
from keras.layers import MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape, \
    RepeatVector, \
    TimeDistributed
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

path_project = '/home/yukina/Missile_Fault_Detection/project/'


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
    path3 = path_project + 'data/banwuli_data/normal/zc1.dat'
    normal_data = get_dat(path3)
    alltest = []
    scaler = RobustScaler().fit(normal_data[3000:10000, :])
    print(id1)
    train_end = 2000
    path_ks = path_project + 'data/banwuli_data/ks/'
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
    path3 = path_project + 'data/banwuli_data/normal/zc1.dat'
    normal_data = get_dat(path3)
    alltest = []
    scaler = RobustScaler().fit(normal_data[3000:10000, :])
    train_end = 2000
    path_qs = path_project + 'data/banwuli_data/rqs/'
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
    path3 = path_project + 'data/banwuli_data/normal/zc1.dat'
    normal_data = get_dat(path3)
    alltest = []
    scaler = RobustScaler().fit(normal_data[3000:10000, :])
    path_sf = path_project + 'data/banwuli_data/sf/'
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
    path3 = path_project + 'data/banwuli_data/normal/zc1.dat'
    normal_data = get_dat(path3)
    alltest = []
    scaler = RobustScaler().fit(normal_data[3000:10000, :])
    for i in range(len(id1)):
        path = path_project + 'data/banwuli_data/T/{}.dat'.format(id1[i])
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
    path3 = path_project + 'data/banwuli_data/normal/zc1.dat'
    normal_data = get_dat(path3)
    alltest = []
    scaler = RobustScaler().fit(normal_data[3000:10000, :])
    for i in range(len(id1)):
        path = path_project + 'data/banwuli_data/lqs/lqs-l/{}.dat'.format(id1[i])
        data = get_dat(path)
        testdata = data[2000:5200, :]
        xtest = scaler.transform(testdata)
        lookback = 5
        test_extract = np.array(temporalize(xtest, lookback))
        test_extract = np.reshape(test_extract, (test_extract.shape[0], lookback, test_extract.shape[3]))
        alltest.append(test_extract)
    for i in range(len(id2)):
        path = path_project + 'data/banwuli_data/lqs/lqs-r/{}.dat'.format(id2[i])
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
    y = concatenate([x, y])
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
    path1 = path_project + 'data/banwuli_data/normal/zc1.dat'
    normal_data = get_dat(path1)
    alltrain = []
    for i in range(2):
        path2 = path_project + 'data/banwuli_data/normal/{}.dat'.format('zc' + str(i + 2))
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
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # # path = 'D:\share\\normal\\zc1.dat'
    # normal_data = get_dat(path)

    # label = np.load('F:\微信文件\研一上大作业\DD\python\detect\data\classify\\cla_mode.npy')

    fea, label = get_processed_data()
    mode = to_categorical(label, num_classes=5)
    ''''''
    # X_train = np.reshape(fea, (fea.shape[0], 1, fea.shape[1]))
    # print(fea.shape)
    # model = generate_lstmfcn(fea.shape[1],3, 8)
    # X_train = fea
    ''''''
    a = []
    f = []
    r = []
    l = []
    seeds_num = 3
    # for j in range(0, fea.shape[1]):
    #     feature = np.concatenate((fea[:, :j, :], fea[:, j+1:, :]), axis=1)
    #     np.save(path_project + 'data/one_hot_label.npy', mode)
    for i in range(0, seeds_num):
        print(f'Training seed={i}........................')
        # add sample_id to the data, for later use of explaining
        new_col = np.array([i for i in range(mode.shape[0])]).reshape((mode.shape[0], 1))
        mode = np.concatenate((mode, new_col), axis=1)

        X_train, X_test, Y_train, Y_test = train_test_split(fea, mode, test_size=0.30, random_state=i)

        ID_train = np.squeeze(Y_train[:, -1:])
        ID_test = np.squeeze(Y_test[:, -1:])

        # remode sample_id from data
        mode = mode[:, :-1]
        Y_train = Y_train[:, :-1]
        Y_test = Y_test[:, :-1]

        # Save the training set and test set and sample_id
        np.save(path_project + 'data/X_train.npy', X_train)
        np.save(path_project + 'data/X_test.npy', X_test)
        np.save(path_project + 'data/Y_train.npy', Y_train)
        np.save(path_project + 'data/Y_test.npy', Y_test)
        np.save(path_project + 'data/ID_train.npy', ID_train)
        np.save(path_project + 'data/ID_test.npy', ID_test)

        model = generate_model(fea.shape[1], 5, fea.shape[2])
        from tensorflow.keras.optimizers import Adam

        learning_rate = 0.001
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # history = model.fit(X_train, mode, batch_size=64, epochs=100)  # 每次取32张图片，共计循环10次
        history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=64,
                            epochs=100)  # 每次取32张图片，共计循环10次
        model.save(path_project + f'anomaly_classify/models/lstm-fcn-{i}.h5')
        # with open('F:\微信文件\研一上大作业\DD\毕设\lstm-fcn\classify.txt', 'wb') as file_pi:
        #     pickle.dump(history.history, file_pi)
        # with open('F:\微信文件\研一上大作业\DD\毕设\lstm-fcn\classify.txt', 'rb') as file_pi:
        #     history = pickle.load(file_pi)

        # from deel.influenciae.common import InfluenceModel, ExactIHVP
        # influence_model = InfluenceModel(model, loss_function=tf.keras.losses.binary_crossentropy)
        fig, ax = plt.subplots(1, 2, figsize=(7, 3), dpi=200, tight_layout=True)
        ax[0].grid(which='major', ls='--', alpha=.8, lw=.8)
        ax[0].plot(history.history['loss'], label='训练集')
        ax[0].plot(history.history['val_loss'], label='验证集')
        # plt.title('model loss')
        xticks = ax[0].get_xticks()
        ax[0].set_xticks(xticks)
        ax[0].set_xticklabels(xticks, fontdict={'family': 'Times New Roman', 'size': 10})
        ax[0].set_xlim(-10, 110)
        yticks = ax[0].get_yticks()
        ax[0].set_yticks(yticks)
        ax[0].set_yticklabels(yticks, fontdict={'family': 'Times New Roman', 'size': 10})
        ax[0].set_xlabel('训练轮数', fontdict={'family': ['SimSun'], 'size': 12})
        ax[0].set_ylabel('Loss', fontdict={'family': 'Times New Roman', 'size': 12})
        ax[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
        ax[0].legend(loc='best', prop={'family': ['SimSun'], 'size': 12})
        # plot_acc(history,ax[0])
        '''plotacc'''
        # model.save('F:\微信文件\研一上大作业\DD\python\detect\data\classify\lstm-fcn.h5')
        # model = load_model('lstm-fcn.h5')
        # cof(model,fea,label,ax[1])
        '''cof'''
        train_pred = model.predict(fea)
        train_pred = np.argmax(train_pred, axis=1)
        con_mat = confusion_matrix(label, train_pred)
        con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]  # 归一化
        con_mat_norm = np.around(con_mat_norm, decimals=2)

        print(con_mat)
        LABELS = ['松浮故障', '缺损故障', '卡死故障', '推力故障', '升力面故障']
        sns.set_theme(font='Times New Roman')
        sns.heatmap(con_mat_norm, xticklabels=LABELS, yticklabels=LABELS, annot=True)
        ax[1].set_xticklabels(LABELS, fontdict={'family': ['SimSun'], 'size': 10})
        ax[1].set_yticklabels(LABELS, fontdict={'family': ['SimSun'], 'size': 10})
        plt.ylabel('实际标签', fontdict={'family': ['SimSun'], 'size': 12})
        plt.xlabel('预测分类', fontdict={'family': ['SimSun'], 'size': 12})
        plt.show()
        # plt.savefig(path_project + f'models/lstm-fcn-{i}.png')

        print('a', accuracy_score(label, train_pred))
        print('f', f1_score(label, train_pred, average='macro'))
        print('r', recall_score(label, train_pred, average='macro'))
        print('l', history.history['loss'][-1])
        a.append(accuracy_score(label, train_pred))
        f.append(f1_score(label, train_pred, average='macro'))
        r.append(recall_score(label, train_pred, average='macro'))
        l.append(history.history['loss'][-1])

    # save result to disk
    result = pd.DataFrame({
        'accuracy': a,
        'f1': f,
        'recall': r,
        'loss': l
    })
    result.to_csv(path_project + f'anomaly_classify/log/performance.csv', index=False)

'''测试'''
# print(model.predict(X_train))
# path2 = 'F:\微信文件\研一上大作业\DD\demo_x\demoV9-7.07\data\\1ka\\1ka0.mat'
# testdata = data_preprocess(path2)
# scaler = RobustScaler().fit(normal_data[3000:8000, :])
# xtest = scaler.transform(testdata)
# lookback = 5
# test_extract = np.array(temporalize(xtest, lookback))
# print(np.shape(test_extract))
# test_extract = np.reshape(test_extract, (test_extract.shape[0], lookback, test_extract.shape[3]))
# time_steps = test_extract.shape[1]
# n_features = test_extract.shape[2]
# model = load_model('location.h5')
# pred = model.predict(test_extract)
# test_pred = data_flatten(pred)
# test_real = data_flatten(test_extract)
# # test_error = np.sum(abs(test_real - test_pred), axis=1)
# test_error = test_real - test_pred
# # test_fea = np.array(test_error[7993:8020, :])
# test_fea = np.array(test_error[7993:8020,:])
# print(test_fea.shape[0])
# # x_test = np.reshape(test_fea, (1, 1, test_fea.shape[0]))
# test_fea = test_fea.T
# x_test = np.reshape(test_fea, (1, test_fea.shape[0], test_fea.shape[1]))
# print('x_test',x_test.shape)
# model_classify = load_model('lstm-fcn.h5')
# y_test = model_classify.predict(x_test)
# print('松浮-0，缺损-1，卡死-2')
# print(y_test)
# # model = generate_model_2()
# # # train_model(model, DATASET_INDEX, dataset_prefix='lp5_', epochs=1000, batch_size=128)
# # evaluate_model(model, DATASET_INDEX, dataset_prefix='lp5_', batch_size=128)
