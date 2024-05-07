import os

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from tqdm import tqdm

from spot import dSPOT
import scipy.io as scio
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import pandas as pd
import matplotlib.ticker as mtick
from scipy.io import savemat
import time
from keras.models import load_model

path_project = '/home/yukina/Missile_Fault_Detection/project'


def selfscaler(traindata, testdata):
    iq = np.percentile(traindata, (25, 75), axis=0)
    fenmu = iq[1, :] - iq[0, :]
    zhongzhi = np.median(traindata, axis=0)
    x1 = (traindata - zhongzhi) / fenmu
    x2 = (zhongzhi - traindata) / fenmu
    x1[x1 < 0] = 0
    x2[x2 < 0] = 0
    xtrain = x1 + x2
    t1 = (testdata - zhongzhi) / fenmu
    t2 = (zhongzhi - testdata) / fenmu
    t1[t1 < 0] = 0
    t2[t2 < 0] = 0
    xtest = t1 + t2
    return xtrain, xtest


def data_preprocess(path1, path2):
    t = scio.loadmat(path1)
    t = t['order_t']
    # print(np.shape(t))
    # t11 = t[1:]
    # t12 = t[:-1]
    # tt1 = t11 - t12
    # zero_line = np.zeros((1,1))
    # tt1 = np.concatenate((zero_line, tt1), axis=0)
    # print(np.shape(tt1))
    rty = scio.loadmat(path2)
    rty = rty['RTY_XYZ']
    data = rty[:, :11]
    data = np.concatenate((data, rty[:, 15:16]), axis=1)
    data = np.concatenate((data, t - 5000), axis=1)
    data = np.array(data)
    return data


def banwuli_preprocess(path1, path2):
    rty = scio.loadmat(path1)
    rty = rty['RTY_XYZ']
    xe = scio.loadmat(path2)
    xe = xe['xe']
    data = np.concatenate((rty, xe[:, 1:2]), axis=1)
    data = np.array(data)
    return data


def get_normal(path1, path2):
    t = scio.loadmat(path1)
    t = t['order_t']
    # t11 = t[1:]
    # t12 = t[:-1]
    # tt1 = t11 - t12
    # zero_line = np.zeros((1, 1))
    # tt1 = np.concatenate((zero_line, tt1), axis=0)
    rty = scio.loadmat(path2)
    rty = rty['RTY_XYZ']
    data = rty[:, :11]
    data = np.concatenate((data, rty[:, 15:16]), axis=1)
    data = np.concatenate((data, t - 5000 + 600), axis=1)
    data = np.array(data)
    return data


def pro_matirx(low_mat, high_mat):
    mat1 = np.dot(np.transpose(low_mat), high_mat)
    mat2 = np.dot(np.transpose(high_mat), high_mat)
    mat = np.dot(mat1, np.linalg.pinv(mat2))
    return mat


def timewindow(timeseries, ts_length):  # 滑动时间窗口法
    n, m = np.shape(timeseries)
    std_value = np.zeros((n - ts_length + 1, m))
    range_value = np.zeros((n - ts_length + 1, m))
    norm_value = np.zeros((n - ts_length + 1, m))
    ji_value = np.zeros((n - ts_length + 1, m))
    # min_value = np.zeros((n - ts_length + 1, m))
    for i in range(m):
        for j in range(0, n - ts_length + 1):
            ts = timeseries[j:j + ts_length - 1, i]
            std_value[j, i] = np.std(ts)
            range_value[j, i] = np.max(ts) - np.min(ts)
            norm_value[j, i] = np.linalg.norm(ts, ord=2)
    extract_value = np.concatenate((std_value, range_value, norm_value), axis=1)
    return extract_value


def detect_realtime(train_distance, test_distance):
    q = 1e-5  # risk parameter
    d = 300  # depth parameter
    s = dSPOT(q, d)  #  biDSPOT object
    s.fit(train_distance, test_distance)  #  data import
    s.initialize(verbose=True)  #  initialization step
    results = s.run()  #  run
    cl = results['thresholds']
    alarm = results['alarms']
    # print(np.shape(cl), np.shape(test_distance))
    # plt.plot(cl)
    # plt.plot(test_distance)
    # plt.show()
    return cl, alarm


def detect_accurate(train_distance, test_distance, Y_test, mode):
    # train_distance训练数据结果；test_distance测试数据结果；ts_length时间窗口长度；train_end训练结束时间；
    # fault_start故障开始时间
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    if mode == 'max':
        pass
    elif mode == 'auto':
        q = 1e-5  # risk parameter
        d = 20  # depth parameter
        s = dSPOT(q, d)  #  biDSPOT object
        s.fit(train_distance, test_distance)  #  data import
        s.initialize(verbose=True)  #  initialization step
        results = s.run()  #  run
        thresholds = results['thresholds']

        id_anomaly_pred = np.where(test_distance > thresholds)[0]
        id_normal_pred = np.where(test_distance <= thresholds)[0]

        tp = np.size(np.where(Y_test[id_anomaly_pred] == 1)[0], 0)
        tn = np.size(np.where(Y_test[id_normal_pred] == 0)[0], 0)
        fp = np.size(np.where(Y_test[id_anomaly_pred] == 0)[0], 0)
        fn = np.size(np.where(Y_test[id_normal_pred] == 1)[0], 0)

        FDR = tp * 100 / (tp + fn)
        if tp + fp == 0:
            FAR = 0
        else:
            FAR = fp * 100 / (tp + fp)

        return FDR, FAR


def km_cluster(lle_train_data, test_new, x_train_new):
    # 降维数据
    kmeans = KMeans(n_clusters=1)
    kmeans.fit_transform(lle_train_data)
    center = kmeans.cluster_centers_
    print(center[0])
    test_distance = np.zeros(test_new.shape[0])
    train_distance = np.zeros(lle_train_data.shape[0])
    for i in range(test_new.shape[0]):
        test_distance[i] = np.sqrt(np.sum(np.square(test_new[i, :] - center[0])))
    for i in range(x_train_new.shape[0]):
        train_distance[i] = np.sqrt(np.sum(np.square(x_train_new[i, :] - center[0])))
    # for i in range(lle_train_data.shape[0]):
    #     train_distance[i] = np.sqrt(np.sum(np.square(lle_train_data[i, :] - center[0])))
    differ = np.divide(test_distance[:10], train_distance[-10:])
    c = np.mean(differ)
    print('c的值', c)
    return train_distance * c, test_distance, center


def km_cluster1(lle_train_data, test_new):
    # 降维数据
    kmeans = KMeans(n_clusters=1)
    kmeans.fit_transform(lle_train_data)
    center = kmeans.cluster_centers_

    test_distance = np.zeros(test_new.shape[0])
    train_distance = np.zeros(lle_train_data.shape[0])
    for i in range(test_new.shape[0]):
        test_distance[i] = np.sqrt(np.sum(np.square(test_new[i, :] - center[0])))
    for i in range(lle_train_data.shape[0]):
        train_distance[i] = np.sqrt(np.sum(np.square(lle_train_data[i, :] - center[0])))
    differ = np.divide(test_distance[:10], train_distance[-10:])
    c = np.mean(differ)
    return train_distance * c, test_distance


def plot_1(data, title, fig_number: int):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fig_row = int(np.ceil(fig_number / 2))
    if fig_number == 2:
        plt.subplot(2, 1, 1)
        plt.plot(data[:, 0])
        plt.ylabel(title[0])
        plt.xlabel('采样点')
        plt.subplot(2, 1, 2)
        plt.plot(data[:, 1])
        plt.ylabel(title[1])
        plt.xlabel('采样点')
    else:
        for i in range(fig_number):
            i = i + 1
            plt.subplot(fig_row, 2, i)
            plt.plot(data[:, i - 1])
            plt.ylabel(title[i - 1])
            plt.xlabel('采样点')
    plt.show()


def getnormaldata():
    path1 = 'F:\微信文件\研一上大作业\DD\demo_x\demoV9-7.07\data\\thloss\\order1.mat'
    path2 = 'F:\微信文件\研一上大作业\DD\demo_x\demoV9-7.07\data\\normal\\normal1.mat'
    normal_data = get_normal(path1, path2)
    alltrain = []
    for i in range(2):
        path3 = 'F:\微信文件\研一上大作业\DD\demo_x\demoV9-7.07\data\\thloss\\order1.mat'
        path4 = 'F:\微信文件\研一上大作业\DD\demo_x\demoV9-7.07\data\\normal\\{}.mat'.format('normal' + str(i + 2))
        data = get_normal(path3, path4)
        traindata = data[3000:9000, :]
        scaler = RobustScaler().fit(normal_data[3000:8000, :])
        xtrain = scaler.transform(traindata)
        alltrain.append(xtrain)
    train_data = np.vstack(alltrain)
    return train_data


def get_dat(path):
    a = pd.read_csv(path, delim_whitespace=True)
    b = a[['dOmega_ib_b[0]', 'dOmega_ib_b[1]', 'dOmega_ib_b[2]', 'Gama', 'Theta_k', 'Psi_t', 'fb[0]', 'fb[1]',
           'fb[2]', 'Alfa', 'Beta', 'zmb', 'P']]
    b = np.array(b, dtype=float)
    return b


def get_dat1(path):
    a = pd.read_csv(path, delim_whitespace=True)
    b = a[['dOmega_ib_b[0]', 'Gama', 'Theta_k', 'Psi_t', 'fb[0]', 'fb[1]',
           'fb[2]', 'Alfa', 'Beta', 'zmb', 'P']]
    b = np.array(b, dtype=float)
    return b


def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'valid')
    return re


def temporalize(X, lookback):
    output_X = []
    for i in range(len(X) - lookback - 1):
        t = []
        for j in range(1, lookback + 1):
            # Gather past records upto the lookback period
            t.append(X[[(i + j + 1)], :])
        output_X.append(t)
    return output_X


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


def hhs_main():
    fault_list = ['ks','lqs','rqs','sf','T']
    window_length = 100
    # 读取SSLLE模型
    test_set_name = 'banwuli_data'
    A = np.load(os.path.join(path_project, f'fun_utils/origin_model/{test_set_name}/projection.npy'))
    train_distance = np.load(os.path.join(path_project, f'fun_utils/origin_model/{test_set_name}/train_distance.npy'))
    center = np.load(os.path.join(path_project, f'fun_utils/origin_model/{test_set_name}/center.npy'))

    # 读取数据
    path_dataset = os.path.join(path_project, f"data/{test_set_name}/yukina_data")

    FDR_list = []
    FAR_list = []
    AUCROC_list = []
    AUCPR_list = []
    score_list = []
    y_list = []
    recall_at_threshold_list = []
    precision_at_threshold_list = []

    for fault in tqdm(fault_list, desc='detecting'):
        FDRs = []
        FARs = []
        AUCROCs = []
        AUCPRs = []
        scores = []
        ys = []
        recall_at_thresholds = []
        precision_at_thresholds = []

        path_fault = os.path.join(path_dataset, 'test', fault)
        files = os.listdir(path_fault)

        for i in range(1, int(len(files) / 2 + 1)):
            X_test = np.load(os.path.join(path_dataset, 'test', fault, f"features_{i}.npy"))
            Y_test = np.load(os.path.join(path_dataset, 'test', fault, f"labels_{i}.npy"))
            X_test = np.transpose(np.dot(A, np.transpose(X_test)))
            test_distance = np.sqrt(np.sum(np.square(X_test - center[0]), axis=1))
            FDR, FAR = detect_accurate(train_distance, test_distance, Y_test, mode='auto')
            aucroc = roc_auc_score(y_true=Y_test, y_score=test_distance)
            aucpr = average_precision_score(y_true=Y_test, y_score=test_distance, pos_label=1)

            precision, recall, _ = precision_recall_curve(Y_test, test_distance)
            precision_threshold = 0.999
            recall_at_threshold = recall[np.where(precision >= precision_threshold)[0][0]]
            recall_threshold = 0.999
            precision_at_threshold = precision[np.where(recall >= recall_threshold)[0][-1]]

            # print('故障{}检测率为:{:.3f}%'.format(fault, FDR))
            # print('故障{}虚警率为:{:.3f}%'.format(fault, FAR))
            FDRs.append(FDR)
            FARs.append(FAR)
            AUCROCs.append(aucroc * 100)
            AUCPRs.append(aucpr * 100)
            scores.append(test_distance)
            ys.append(Y_test)
            recall_at_thresholds.append(recall_at_threshold * 100)
            precision_at_thresholds.append(precision_at_threshold * 100)

        FDR_list.append(np.mean(FDRs))
        FAR_list.append(np.mean(FARs))
        AUCROC_list.append(np.mean(AUCROCs))
        AUCPR_list.append(np.mean(AUCPRs))
        score_list.append(np.array(scores))
        y_list.append(np.array(ys))
        recall_at_threshold_list.append(np.mean(recall_at_thresholds))
        precision_at_threshold_list.append(100 - np.mean(precision_at_thresholds))
        print('平均故障检测率为:{:.3f}%'.format(np.mean(FDRs)))
        print('平均虚警率为:{:.3f}%'.format(np.mean(FARs)))
        print('平均AUCROC为:{:.3f}'.format(np.mean(AUCROCs)))
        print('平均AUCPR为:{:.3f}'.format(np.mean(AUCPRs)))

    # 结果保存到csv文件
    result = pd.DataFrame(
        {'fault': fault_list, 'FDR': FDR_list, 'FAR': FAR_list, 'AUCROC': AUCROC_list, 'AUCPR': AUCPR_list,
         'FDR_at_threshold': recall_at_threshold_list, 'FAR_at_threshold': precision_at_threshold_list})
    result.to_csv(os.path.join(path_project, f"fun_utils/Fault_Detection/log/{test_set_name}/SSLLE_result.csv"), index=False)

    # 异常分数保存到npy文件
    np.save(os.path.join(path_project, "data/scores/scores_SSLLE-AD.npy"), score_list)
    np.save(os.path.join(path_project, "data/scores/labels.npy"), y_list)


hhs_main()
