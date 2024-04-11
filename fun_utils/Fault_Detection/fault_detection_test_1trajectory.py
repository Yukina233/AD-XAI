import os

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
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


def detect_accurate(train_distance, test_distance, ts_length, train_end, fault_start, mode):
    # train_distance训练数据结果；test_distance测试数据结果；ts_length时间窗口长度；train_end训练结束时间；
    # fault_start故障开始时间
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    if mode == 'max':
        cl = np.max(train_distance)
        th_d = test_distance - cl
        id_high = np.where(th_d > 0)[0]
        id_low = np.where(th_d < 0)[0]
        divide = fault_start - train_end
        pf_rate = np.size(np.where(id_low >= divide - ts_length)[0], 0) * 100 / (
                np.size(test_distance, 0) - divide + ts_length)
        fp_rate = np.size(np.where(id_high < divide - ts_length)[0], 0) * 100 / (divide - ts_length)
        print('故障误报率为:{:.3f}%'.format(fp_rate))
        print('故障漏报率为:{:.3f}%'.format(pf_rate))

        fault_time = (id_high[np.where(id_high >= fault_start - train_end - ts_length - 1)[0]][0] + ts_length) / 100
        print('故障检测时间为:{:.3f}s'.format(fault_time))
        x_ticks = np.linspace(train_end / 100 + ts_length / 100,
                              train_end / 100 + len(test_distance) / 100 - 0.01 + ts_length / 100,
                              len(test_distance))
        colors = ["#0073C2", "#EFC000", "#868686", "#CD534C", "#7AA6DC", "#003C67"]
        fig, ax = plt.subplots(dpi=200)
        ax.grid(which='major', ls='--', alpha=.8, lw=.8)
        plt.plot(x_ticks, test_distance, color=colors[0], label='K均值距离')
        plt.plot([train_end / 100 + 3, train_end / 100 + len(test_distance) / 100 - 0.01 + 3], [cl, cl],
                 color=colors[3], label='最大值阈值')
        plt.xticks(fontproperties='Times New Roman', fontsize=10)
        plt.yticks(fontproperties='Times New Roman', fontsize=10)
        # plt.plot(x_ticks,train_distance, 'k', label='训练数据')
        # plt.title('故障检测', fontdict={'family': ['SimSun'], 'size': 14})
        # new_ticks = np.linspace(60,100,4001)
        # plt.xticks(new_ticks)
        plt.xlabel('时间/s', fontdict={'family': ['SimSun'], 'size': 12})
        plt.ylabel('故障检测指标值', fontdict={'family': ['SimSun'], 'size': 12})
        plt.legend(loc='best', prop={'family': ['SimSun'], 'size': 12})
        axins = ax.inset_axes((0.08, 0.45, 0.35, 0.25))  # 子图的位置和宽度占整个图的比例
        # axins.plot(x_ticks,test_distance,color = colors[0])
        # axins.plot([train_end / 100 + 3,train_end / 100 + len(test_distance) / 100 - 0.01 + 3],[cl,cl], color = colors[3])
        zone_left = int((fault_time - 1) * 100 - 200)
        zone_right = int((fault_time + 1) * 100 - 200)
        x_ratio = 0
        y_ratio = 0.1
        xlim0 = x_ticks[zone_left] - (x_ticks[zone_right] - x_ticks[zone_left]) * x_ratio
        xlim1 = x_ticks[zone_right] + (x_ticks[zone_right] - x_ticks[zone_left]) * x_ratio
        y = np.hstack((test_distance[zone_left:zone_right]))
        ylim0 = np.min(y) - (np.max(y) - np.min(y)) * y_ratio
        ylim1 = np.max(y) + (np.max(y) - np.min(y)) * y_ratio

        # 调整子坐标系的显示范围
        axins.set_xlim(xlim0, xlim1)
        axins.set_ylim(ylim0, ylim1)
        # plt.gcf().subplots_adjust(left=0.05, top=0.91, bottom=0.09)
        # 原图中画方框
        tx0 = xlim0
        tx1 = xlim1
        ty0 = ylim0
        ty1 = ylim1
        sx = [tx0, tx1, tx1, tx0, tx0]
        sy = [ty0, ty0, ty1, ty1, ty0]
        ax.plot(sx, sy, "black", linestyle='-.')
        # 画两条线
        # xy = (xlim0, ylim0)
        # xy2 = (xlim1, ylim0)
        # con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
        #                       axesA=axins, axesB=ax)
        # axins.add_artist(con)
        # xy = (xlim0, ylim1)
        # xy2 = (xlim1, ylim1)
        # con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
        #                       axesA=axins, axesB=ax)
        # axins.add_artist(con)
        plt.show()
    elif mode == 'auto':
        q = 1e-5  # risk parameter
        d = 20  # depth parameter
        s = dSPOT(q, d)  #  biDSPOT object
        s.fit(train_distance, test_distance)  #  data import
        s.initialize(verbose=True)  #  initialization step
        results = s.run()  #  run
        cl = results['thresholds']
        alarm = results['alarms']
        # plt.show()
        th_d = test_distance - cl
        id_high = np.where(th_d > 0)[0]
        id_low = np.where(th_d < 0)[0]
        divide = fault_start - train_end

        # b = np.where(id_high >= fault_start - train_end - ts_length + 10)[0]
        pf_rate = np.size(np.where(id_low >= divide - ts_length)[0], 0) * 100 / (
                np.size(test_distance, 0) - divide + 100)
        fp_rate = np.size(np.where(id_high < divide - ts_length)[0], 0) * 100 / (divide - 100)
        tp_rate = np.size(np.where(id_high >= divide - ts_length)[0], 0) * 100 / (divide - 100)

        tp = np.size(np.where(id_high >= divide - ts_length)[0], 0)
        tn = np.size(np.where(id_low < divide - ts_length)[0], 0)
        fp = np.size(np.where(id_high < divide - ts_length)[0], 0)
        fn = np.size(np.where(id_low >= divide - ts_length)[0], 0)

        FDR = tp * 100 / (tp + fn)
        if tp + fp == 0:
            FAR = 0
        else:
            FAR = fp * 100 / (tp + fp)

        Y_test = np.concatenate((np.zeros(divide - ts_length), np.ones(test_distance.size - divide + ts_length)))
        aucroc = roc_auc_score(y_true=Y_test, y_score=test_distance)
        aucpr = average_precision_score(y_true=Y_test, y_score=test_distance, pos_label=1)

        precision, recall, _ = precision_recall_curve(Y_test, test_distance)
        precision_threshold = 0.99
        recall_at_threshold = recall[np.where(precision >= precision_threshold)[0][0]]
        recall_threshold = 0.99
        precision_at_threshold = precision[np.where(recall >= recall_threshold)[0][-1]]

        print('故障误报率为:{:.3f}%'.format(fp_rate))
        print('故障漏报率为:{:.3f}%'.format(pf_rate))

        print('故障检测率为:{:.3f}%'.format(FDR))
        print('虚警率为:{:.3f}%'.format(FAR))
        print('AUCROC为:{:.3f}'.format(aucroc))
        print('AUCPR为:{:.3f}'.format(aucpr))
        # fault_time = (id_high[np.where(id_high >= fault_start - train_end - ts_length + 10)[0]][0] + ts_length-fault_start+train_end) / 100
        if id_high.shape[0] > 0:
            fault_time = (id_high[0] - 2901) / 100
        else:  # 如果没有检测出任何故障，检测时间为-1
            fault_time = -1
        # print(id_high)
        print('故障检测时间为:{:.3f}s'.format(fault_time))
        x_ticks = np.linspace(train_end / 100 + int(ts_length / 100),
                              train_end / 100 + len(test_distance) / 100 - 0.01 + int(ts_length / 100),
                              len(test_distance))
        colors = ["#0073C2", "#EFC000", "#868686", "#CD534C", "#7AA6DC", "#003C67"]
        fig, ax = plt.subplots(dpi=200)
        ax.grid(which='major', ls='--', alpha=.8, lw=.8)
        plt.plot(x_ticks, test_distance, color=colors[0], label='K均值距离')
        plt.plot(x_ticks, cl, color=colors[3], label='自适应阈值')
        savemat("test_distance.mat", {"test_distance": test_distance})
        savemat("cl.mat", {"cl": cl})

        plt.xticks(fontproperties='Times New Roman', fontsize=10)
        plt.yticks(fontproperties='Times New Roman', fontsize=10)
        plt.xlabel('时间/s', fontdict={'family': ['SimSun'], 'size': 12})
        plt.ylabel('故障检测指标值', fontdict={'family': ['SimSun'], 'size': 12})
        plt.title('升力面缺损故障下SSLLE自适应阈值故障检测效果', fontdict={'family': ['SimSun'], 'size': 12})
        plt.legend(loc='best', prop={'family': ['SimSun'], 'size': 12})

        axins = ax.inset_axes((0.18, 0.45, 0.35, 0.25))  # 子图
        axins.plot(x_ticks, test_distance, color=colors[0])
        axins.plot(x_ticks, cl, color=colors[3])
        zone_left = int((29 - 0.1) * 100)
        zone_right = int((29 + 0.1) * 100)  # 将故障前后0.1s的数据用于展示子图
        x_ratio = 0
        y_ratio = 0.02
        xlim0 = x_ticks[zone_left] - (x_ticks[zone_right] - x_ticks[zone_left]) * x_ratio
        xlim1 = x_ticks[zone_right] + (x_ticks[zone_right] - x_ticks[zone_left]) * x_ratio
        y = np.hstack((test_distance[zone_left:zone_right]))
        ylim0 = np.min(y) - (np.max(y) - np.min(y)) * y_ratio
        ylim1 = np.max(y) + (np.max(y) - np.min(y)) * y_ratio
        # 调整子坐标系的显示范围
        axins.set_xlim(xlim0, xlim1)
        axins.set_ylim(ylim0, ylim1)
        tx0 = xlim0
        tx1 = xlim1
        ty0 = ylim0
        ty1 = ylim1
        sx = [tx0, tx1, tx1, tx0, tx0]
        sy = [ty0, ty0, ty1, ty1, ty0]
        ax.plot(sx, sy, "black", linestyle='-.')
        # 画两条线
        xy = (xlim0, ylim0)
        xy2 = (xlim1, ylim0)
        con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                              axesA=axins, axesB=ax)
        axins.add_artist(con)
        xy = (xlim0, ylim1)
        xy2 = (xlim1, ylim1)
        con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                              axesA=axins, axesB=ax)
        axins.add_artist(con)
        plt.show()
        return FDR, FAR, aucroc, aucpr, recall_at_threshold, 1 - precision_at_threshold


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


def plot_data(data):
    fig, ax = plt.subplots(2, 2, figsize=(8, 6), dpi=200, tight_layout=True)
    x_ticks = np.linspace(0, data.shape[0] / 100, data.shape[0])
    ax[0, 0].grid(which='major', ls='--', alpha=.8, lw=.8)
    ax[0, 0].plot(x_ticks, data[:, 0], label='x轴角加速度')
    ax[0, 0].plot(x_ticks, data[:, 1], label='y轴角加速度')
    ax[0, 0].plot(x_ticks, data[:, 2], label='z轴角加速度')
    xticks = np.arange(0, data.shape[0] / 100, 10)
    ax[0, 0].set_xticks(xticks)
    ax[0, 0].set_xticklabels(xticks, fontdict={'family': 'Times New Roman', 'size': 10})
    yticks = ax[0, 0].get_yticks()
    ax[0, 0].set_yticks(yticks)
    ax[0, 0].set_yticklabels(yticks, fontdict={'family': 'Times New Roman', 'size': 10})
    ax[0, 0].set_xlabel('时间(秒)', fontdict={'family': ['SimSun'], 'size': 12})
    ax[0, 0].set_ylabel('角加速度(度/s^2)', fontdict={'family': ['SimSun'], 'size': 12})
    ax[0, 0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax[0, 0].legend(loc='lower right', prop={'family': ['SimSun'], 'size': 12})

    ax[0, 1].grid(which='major', ls='--', alpha=.8, lw=.8)
    ax[0, 1].plot(x_ticks, data[:, 3], label='滚转角')
    ax[0, 1].plot(x_ticks, data[:, 4], label='俯仰角')
    ax[0, 1].plot(x_ticks, data[:, 5], label='偏航角')
    ax[0, 1].set_xticks(xticks)
    ax[0, 1].set_xticklabels(xticks, fontdict={'family': 'Times New Roman', 'size': 10})
    yticks = ax[0, 1].get_yticks()
    ax[0, 1].set_yticks(yticks)
    ax[0, 1].set_yticklabels(yticks, fontdict={'family': 'Times New Roman', 'size': 10})
    ax[0, 1].set_xlabel('时间(秒)', fontdict={'family': ['SimSun'], 'size': 12})
    ax[0, 1].set_ylabel('姿态角(度)', fontdict={'family': ['SimSun'], 'size': 12})
    ax[0, 1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax[0, 1].legend(loc='upper left', prop={'family': ['SimSun'], 'size': 12})

    ax[1, 0].grid(which='major', ls='--', alpha=.8, lw=.8)
    ax[1, 0].plot(x_ticks, data[:, 6], label='x轴加速度')
    ax[1, 0].plot(x_ticks, data[:, 7], label='y轴加速度')
    ax[1, 0].plot(x_ticks, data[:, 8], label='z轴加速度')

    ax[1, 0].set_xticks(xticks)
    ax[1, 0].set_xticklabels(xticks, fontdict={'family': 'Times New Roman', 'size': 10})
    yticks = ax[1, 0].get_yticks()
    ax[1, 0].set_yticks(yticks)
    ax[1, 0].set_yticklabels(yticks, fontdict={'family': 'Times New Roman', 'size': 10})
    ax[1, 0].set_xlabel('时间(秒)', fontdict={'family': ['SimSun'], 'size': 12})
    ax[1, 0].set_ylabel('加速度(m/s^2)', fontdict={'family': ['SimSun'], 'size': 12})
    ax[1, 0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax[1, 0].legend(loc='best', prop={'family': ['SimSun'], 'size': 12})

    # ax[1,1].grid(which='major', ls='--', alpha=.8, lw=.8)
    # ax[1,1].plot(x_ticks,data[:, 9], label='迎角')
    # ax[1,1].plot(x_ticks,data[:, 10], label='侧滑角')
    # # ax[1,1].plot(x_ticks,data[:, 11] * 57.3, label='y轴偏移')
    # ax[1,1].set_xticks(xticks)
    # ax[1,1].set_xticklabels(xticks, fontdict={'family': 'Times New Roman', 'size': 10})
    # yticks = ax[1,1].get_yticks()
    # ax[1,1].set_yticks(yticks)
    # ax[1,1].set_yticklabels(yticks, fontdict={'family': 'Times New Roman', 'size': 10})
    # ax[1,1].set_xlabel('时间(秒)', fontdict={'family': ['SimSun'], 'size': 12})
    # ax[1,1].set_ylabel('气动角(度)', fontdict={'family': ['SimSun'], 'size': 12})
    # ax[1,1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    # ax[1,1].legend(loc='best', prop={'family': ['SimSun'], 'size': 12})
    plt.show()


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
    # path1 = os.path.join(path_project, "data/banwuli_data/normal/zc1.dat")
    # nor_data = get_dat(path1)

    # plot_data(nor_data)
    # xtrain = nor_data[2000:, :]
    # scaler = RobustScaler().fit(xtrain)

    nor_data = None
    path_all_data = os.path.join(path_project, "data/banwuli_data")
    path_normal = os.path.join(path_all_data, 'normal')
    for file in tqdm(os.listdir(path_normal), desc='Read data for robust scaler'):
        path_file = os.path.join(path_normal, file)
        data = get_dat(path_file)
        data = data[2000:, :]
        if nor_data is None:
            nor_data = data
        else:
            nor_data = np.concatenate((nor_data, data))

    scaler = RobustScaler().fit(nor_data)

    nor_t = moving_average(nor_data[:, -1], 10)
    t_mean = np.mean(nor_t)
    t_std = np.std(nor_t)
    th1 = t_mean + 3 * t_std

    A = np.load(os.path.join(path_project, 'fun_utils/origin_model/projection.npy'))
    train_distance = np.load(os.path.join(path_project, 'fun_utils/origin_model/train_distance.npy'))
    center = np.load(os.path.join(path_project, 'fun_utils/origin_model/center.npy'))

    # path2 = 'D:\share\新建文件夹\T\T0.6.dat'
    for ratio in ['0.02', '0.04', '0.06', '0.08', '0.10']:
        FDRs = []
        FARs = []
        AUCROCs = []
        AUCPRs = []
        FDR_at_thresholds = []
        FAR_at_thresholds = []

        for num in ['1', '2', '3', '4']:

            path2 = os.path.join(path_project, f'data/banwuli_data/rqs/rqs-{num}/rqs-{num}-{ratio}.dat')  #
            fault_data = get_dat(path2)
            train_end = 2000  # 2000

            fault_start = 5000
            window_length = 100
            for i in range(1):
                testdata = fault_data[2000:, :]
                xtest = scaler.transform(testdata)
                test_extract = timewindow(xtest, window_length)
                test_new = np.transpose(np.dot(A, np.transpose(test_extract)))
                test_distance = np.sqrt(np.sum(np.square(test_new - center[0]), axis=1))
                FDR, FAR, aucroc, aucpr, FDR_at_threshold, FAR_at_threshold = detect_accurate(train_distance, test_distance,
                                                                                              window_length, train_end,
                                                                                              fault_start, mode='auto')

                FDRs.append(FDR)
                FARs.append(FAR)
                AUCROCs.append(aucroc * 100)
                AUCPRs.append(aucpr * 100)
                FDR_at_thresholds.append(FDR_at_threshold * 100)
                FAR_at_thresholds.append(FAR_at_threshold * 100)
        csv_data = {'FDR': FDRs, 'FAR': FARs, 'AUCROC': AUCROCs, 'AUCPR': AUCPRs, 'FDR_at_threshold': FDR_at_thresholds,
                    'FAR_at_threshold': FAR_at_thresholds}
        df = pd.DataFrame(csv_data)
        df.to_csv(os.path.join(path_project, f'fun_utils/Fault_Detection/log/SSLLE_result_rqs_{ratio}.csv'), index=False)


hhs_main()
