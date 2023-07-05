import numpy as np
from sklearn.preprocessing import RobustScaler,MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from spot import dSPOT
from matplotlib.patches import ConnectionPatch
import pandas as pd
import matplotlib.ticker as mtick
import ctypes as c
import time
from keras.models import load_model

def pro_matirx(low_mat,high_mat):
    mat1 = np.dot(np.transpose(low_mat),high_mat)
    mat2 = np.dot(np.transpose(high_mat),high_mat)
    mat = np.dot(mat1,np.linalg.pinv(mat2))
    return mat

def timewindow(timeseries,ts_length):#滑动时间窗口法
    # n, m = np.shape(timeseries)
    # std_value = np.zeros((n - ts_length+1, m))
    # range_value = np.zeros((n - ts_length+1, m))
    # norm_value = np.zeros((n - ts_length + 1, m))
    # for i in range(m):
    #     for j in range(0, n-ts_length+1):
    #         ts = timeseries[j:j+ts_length-1, i]
    #         std_value[j, i] = np.std(ts)
    #         range_value[j, i] = np.max(ts) - np.min(ts)
    #         norm_value[j, i] = np.linalg.norm(ts, ord=2)
    # extract_value = np.concatenate((std_value, range_value, norm_value), axis=1)
    # print(extract_value.shape)
    std_value = np.std(timeseries, axis=0)
    range_value = np.max(timeseries, axis=0) - np.min(timeseries, axis=0)
    norm_value = np.linalg.norm(timeseries, ord=2, axis=0)
    extract_value = np.concatenate((std_value, range_value, norm_value))
    extract_value = np.reshape(extract_value, (1, extract_value.shape[0]))
    return extract_value

def detect_realtime(s, train_distance,test_distance):
    s.fit(train_distance, test_distance)  #  data import
    s.initialize(verbose=False)  #  initialization step
    results = s.run()  #  run
    # cl = results['thresholds']
    alarm = results['alarms']
    # print(np.shape(cl), np.shape(test_distance))
    return alarm

def detect_accurate(train_distance,test_distance,ts_length,train_end,fault_start,mode):
    #train_distance训练数据结果；test_distance测试数据结果；ts_length时间窗口长度；train_end训练结束时间；
    #fault_start故障开始时间
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    if mode == 'max':
        cl = np.max(train_distance)
        th_d = test_distance - cl
        id_high = np.where(th_d > 0)[0]
        id_low = np.where(th_d < 0)[0]
        divide = fault_start - train_end
        pf_rate=np.size(np.where(id_low>=divide-ts_length)[0],0)*100/(np.size(test_distance,0)-divide+ts_length)
        fp_rate = np.size(np.where(id_high < divide-ts_length)[0],0)*100/(divide - ts_length)
        print('故障误报率为:{:.3f}%'.format(fp_rate))
        print('故障漏报率为:{:.3f}%'.format(pf_rate))
        fault_time = (id_high[np.where(id_high >= fault_start - train_end - ts_length - 1)[0]][0] + ts_length)/100
        print('故障检测时间为:{:.3f}s'.format(fault_time))
        x_ticks = np.linspace(train_end / 100 + ts_length/100, train_end / 100 + len(test_distance) / 100 - 0.01 + ts_length/100,
                              len(test_distance))
        colors = ["#0073C2", "#EFC000", "#868686", "#CD534C", "#7AA6DC", "#003C67"]
        fig,ax = plt.subplots(dpi=200)
        ax.grid(which='major', ls='--', alpha=.8, lw=.8)
        plt.plot(x_ticks, test_distance, color = colors[0], label='K均值距离')
        plt.plot([train_end / 100 + 3,train_end / 100 + len(test_distance) / 100 - 0.01 + 3],[cl,cl], color = colors[3], label='最大值阈值')
        plt.xticks(fontproperties='Times New Roman', fontsize=10)
        plt.yticks(fontproperties='Times New Roman', fontsize=10)
        # plt.plot(x_ticks,train_distance, 'k', label='训练数据')
        # plt.title('故障检测', fontdict={'family': ['SimSun'], 'size': 14})
        # new_ticks = np.linspace(60,100,4001)
        # plt.xticks(new_ticks)
        plt.xlabel('时间/s', fontdict={'family': ['SimSun'], 'size': 12})
        plt.ylabel('故障检测指标值', fontdict={'family': ['SimSun'], 'size': 12})
        plt.legend(loc='best', prop={'family': ['SimSun'], 'size': 12})


    elif mode == 'auto':
        q = 1e-5 # risk parameter
        d = 500  # depth parameter
        s = dSPOT(q, d)  #  biDSPOT object
        s.fit(train_distance, test_distance)  #  data import
        s.initialize(verbose=True)  #  initialization step
        results = s.run()  #  run
        cl = results['thresholds']
        alarm = results['alarms']
        plt.plot(alarm)
        # plt.show()
        print(np.shape(cl),np.shape(test_distance))
        th_d = test_distance - cl
        id_high = np.where(th_d > 0)[0]
        id_low = np.where(th_d < 0)[0]
        divide = fault_start - train_end
        pf_rate = np.size(np.where(id_low >= divide - ts_length)[0], 0) * 100 / (np.size(test_distance, 0) - divide + window_length)
        fp_rate = np.size(np.where(id_high < divide - ts_length)[0], 0) * 100 / (divide - window_length)
        print('故障误报率为:{:.3f}%'.format(fp_rate))
        print('故障漏报率为:{:.3f}%'.format(pf_rate))
        # fault_time = (id_high[np.where(id_high >= fault_start - train_end - ts_length - 1)[0]][0] + ts_length) / 100
        fault_time = (id_high[np.where(id_high >= fault_start - train_end - ts_length - 1)[0]][0] + ts_length) / 100
        print('故障检测时间为:{:.3f}s'.format(fault_time))
        x_ticks = np.linspace(train_end / 100 + 3, train_end / 100 + len(test_distance) / 100 - 0.01 + 3,
                              len(test_distance))
        colors = ["#0073C2", "#EFC000", "#868686", "#CD534C", "#7AA6DC", "#003C67"]
        fig, ax = plt.subplots(dpi=200)
        ax.grid(which='major', ls='--', alpha=.8, lw=.8)
        plt.plot(x_ticks, test_distance, color=colors[0], label='K均值距离')
        plt.plot(x_ticks, cl, color=colors[3], label='自适应阈值')
        plt.xticks(fontproperties='Times New Roman', fontsize=10)
        plt.yticks(fontproperties='Times New Roman', fontsize=10)
        # plt.plot(x_ticks,train_distance, 'k', label='训练数据')
        # plt.title('故障检测', fontdict={'family': ['SimSun'], 'size': 14})
        # new_ticks = np.linspace(60,100,4001)
        # plt.xticks(new_ticks)
        plt.xlabel('时间/s', fontdict={'family': ['SimSun'], 'size': 12})
        plt.ylabel('故障检测指标值', fontdict={'family': ['SimSun'], 'size': 12})
        plt.legend(loc='best', prop={'family': ['SimSun'], 'size': 12})
        axins = ax.inset_axes((0.08, 0.45, 0.35, 0.25))
        axins.plot(x_ticks, test_distance, color=colors[0])
        axins.plot(x_ticks, cl, color=colors[3])
        zone_left = int((fault_time - 1) * 100 - 300)
        zone_right = int((fault_time + 1) * 100 - 300)
        x_ratio = 0
        y_ratio = 0.05
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
        return fp_rate,pf_rate

def km_cluster(lle_train_data, test_new, x_train_new):
    # 降维数据
    kmeans = KMeans(n_clusters=1)
    kmeans.fit_transform(lle_train_data)
    center = kmeans.cluster_centers_

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
    print('c的值',c)
    return train_distance*c, test_distance, center

def get_dat(path):
    a = pd.read_csv(path, delim_whitespace=True)
    b = a[['dOmega_ib_b[0]', 'dOmega_ib_b[1]', 'dOmega_ib_b[2]', 'Gama', 'Theta_k', 'Psi_t', 'fb[0]', 'fb[1]',
           'fb[2]', 'Alfa', 'Beta', 'zmb', 'P']]
    b = np.array(b, dtype=float)
    return b

def plot_data(data):

    fig, ax = plt.subplots(2, 2, figsize=(8, 6), dpi=200, tight_layout=True)
    x_ticks = np.linspace(0, 100, data.shape[0])
    ax[0,0].grid(which='major', ls='--', alpha=.8, lw=.8)
    ax[0,0].plot(x_ticks,data[:,0], label='x轴角加速度')
    ax[0,0].plot(x_ticks,data[:,1], label='y轴角加速度')
    ax[0,0].plot(x_ticks,data[:,2], label='z轴角加速度')
    xticks = np.arange(0, 100, 10)
    ax[0,0].set_xticks(xticks)
    ax[0,0].set_xticklabels(xticks, fontdict={'family': 'Times New Roman', 'size': 10})
    yticks = ax[0,0].get_yticks()
    ax[0,0].set_yticks(yticks)
    ax[0,0].set_yticklabels(yticks, fontdict={'family': 'Times New Roman', 'size': 10})
    ax[0,0].set_xlabel('时间(秒)', fontdict={'family': ['SimSun'], 'size': 12})
    ax[0,0].set_ylabel('角加速度(度/s^2)', fontdict={'family': ['SimSun'], 'size': 12})
    ax[0,0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax[0,0].legend(loc='lower right', prop={'family': ['SimSun'], 'size': 12})

    ax[0,1].grid(which='major', ls='--', alpha=.8, lw=.8)
    ax[0,1].plot(x_ticks,data[:, 3], label='滚转角')
    ax[0,1].plot(x_ticks,data[:, 4], label='俯仰角')
    ax[0,1].plot(x_ticks,data[:, 5], label='偏航角')
    ax[0,1].set_xticks(xticks)
    ax[0,1].set_xticklabels(xticks, fontdict={'family': 'Times New Roman', 'size': 10})
    yticks = ax[0,1].get_yticks()
    ax[0,1].set_yticks(yticks)
    ax[0,1].set_yticklabels(yticks, fontdict={'family': 'Times New Roman', 'size': 10})
    ax[0,1].set_xlabel('时间(秒)', fontdict={'family': ['SimSun'], 'size': 12})
    ax[0,1].set_ylabel('姿态角(度)', fontdict={'family': ['SimSun'], 'size': 12})
    ax[0,1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax[0,1].legend(loc='upper left', prop={'family': ['SimSun'], 'size': 12})

    ax[1,0].grid(which='major', ls='--', alpha=.8, lw=.8)
    ax[1,0].plot(x_ticks,data[:, 6], label='x轴加速度')
    ax[1,0].plot(x_ticks,data[:, 7], label='y轴加速度')
    ax[1,0].plot(x_ticks,data[:, 8], label='z轴加速度')

    ax[1,0].set_xticks(xticks)
    ax[1,0].set_xticklabels(xticks, fontdict={'family': 'Times New Roman', 'size': 10})
    yticks = ax[1,0].get_yticks()
    ax[1,0].set_yticks(yticks)
    ax[1,0].set_yticklabels(yticks, fontdict={'family': 'Times New Roman', 'size': 10})
    ax[1,0].set_xlabel('时间(秒)', fontdict={'family': ['SimSun'], 'size': 12})
    ax[1,0].set_ylabel('加速度(m/s^2)', fontdict={'family': ['SimSun'], 'size': 12})
    ax[1,0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax[1,0].legend(loc='best', prop={'family': ['SimSun'], 'size': 12})

    ax[1,1].grid(which='major', ls='--', alpha=.8, lw=.8)
    ax[1,1].plot(x_ticks,data[:, 9], label='迎角')
    ax[1,1].plot(x_ticks,data[:, 10], label='侧滑角')
    # ax[1,1].plot(x_ticks,data[:, 11] * 57.3, label='y轴偏移')
    ax[1,1].set_xticks(xticks)
    ax[1,1].set_xticklabels(xticks, fontdict={'family': 'Times New Roman', 'size': 10})
    yticks = ax[1,1].get_yticks()
    ax[1,1].set_yticks(yticks)
    ax[1,1].set_yticklabels(yticks, fontdict={'family': 'Times New Roman', 'size': 10})
    ax[1,1].set_xlabel('时间(秒)', fontdict={'family': ['SimSun'], 'size': 12})
    ax[1,1].set_ylabel('气动角(度)', fontdict={'family': ['SimSun'], 'size': 12})
    ax[1,1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax[1,1].legend(loc='best', prop={'family': ['SimSun'], 'size': 12})
    plt.show()

def temporalize(X, lookback):
    output_X = []
    for i in range(len(X) - lookback - 1):
        t = []
        for j in range(1, lookback + 1):
            # Gather past records upto the lookback period
            t.append(X[[(i + j + 1)], :])
        output_X.append(t)
    return output_X

def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'valid')
    return re

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


if __name__ == '__main__':
    model = load_model('location.h5')
    model_classify = load_model('lstm-fcn.h5')

    ka_location = load_model('ka_location.h5')
    song_location = load_model('song_location.h5')
    que_location = load_model('que_location.h5')
    ls_location = load_model('ls_location.h5')

    ls1_estimation = load_model('ls1_estimation.h5')
    ls2_estimation = load_model('ls2_estimation.h5')
    rqs1_estimation = load_model('rqs1_estimation.h5')
    rqs2_estimation = load_model('rqs2_estimation.h5')
    rqs3_estimation = load_model('rqs3_estimation.h5')
    rqs4_estimation = load_model('rqs4_estimation.h5')
    th_estimation = load_model('th_estimation.h5')
    aa = np.zeros([1, 13, 27])
    cc = model_classify(aa, training=False)
    duomian = ['一号舵面', '二号舵面', '三号舵面', '四号舵面']
    chengdu = ['一级', '二级', '三级', '四级', '五级']
    shenglimian = ['左升力面', '右升力面']
    fault_diagnosis = []
    fault_estimation = []

    path1 = 'D:\share\\normal\\zc1.dat'
    nor_data = get_dat(path1)
    xtrain = nor_data[2000:,:]

    nor_t = moving_average(nor_data[:, -1], 10)
    t_mean = np.mean(nor_t)
    t_std = np.std(nor_t)
    th1 = t_mean + 3 * t_std

    scaler = RobustScaler().fit(xtrain)
    A = np.load('projection.npy')
    train_distance = np.load('train_distance.npy')
    center = np.load('center.npy')
    # path2 = 'D:\share\\SimuData20230220_2351.dat'
    # ka_data = get_dat(path2)
    dll = c.windll.LoadLibrary('./Fiber2125API.dll')
    a = dll.FIB2125_Open(0)   #打开板卡0

    init_fault = (c.c_double * 3)(0, 0, 0)  # 数组发送，初始故障诊断结果为0
    aa = dll.FIB2125_WriteDouble(0, 0x1600, init_fault, 3)  # 单个数据
    fault_position = []
    a1 = time.time()
    initial_data = []
    l_len = 15
    add = (c.c_double * l_len)(0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0)  # 数组接收，C语言数组要先初始化
    while True:
        aa = dll.FIB2125_ReadDouble(0, 0x1008, add, l_len)  # aa得到函数返回值，表示数据接收的状态，数据接收到add中
        print(add[0])  #时间
        if 20 <= add[0] <= 21:
            break     #超过20s就停下来
    time1 = time.perf_counter()
    CNT = 0
    while True:
        if time.perf_counter() - time1 >= 0.01:  # 10ms接收一次数据
            CNT = CNT + 1
            time1 = time.perf_counter()
            aa = dll.FIB2125_ReadDouble(0, 0x1008, add, l_len)  # 数组
            print(add[0])
            # print(aa)  # 0表示正常通信
            initial_data.append(add[2:])
        if CNT == 100:
            break
    time1 = time.perf_counter()
    train_end = 2000#2000
    fault_start = 5000   #故障开始时间
    window_length = 100  #滑动窗口长度
    fault_time = 0
    q = 1e-5  # risk parameter
    d = 100  # depth parameter
    s = dSPOT(q, d)  #  biDSPOT object
    # history_data = []
    diagnosis_data = []
    fault_position_index = 0
    while True:
        if time.perf_counter() - time1 >= 0.01:
            time1 = time.perf_counter()
            aa = dll.FIB2125_ReadDouble(0, 0x1008, add, l_len)  # 数组
            print(add[:])
            testdata = initial_data
            xtest = scaler.transform(testdata)
            test_extract = timewindow(xtest, window_length)
            test_new = np.transpose(np.dot(A, np.transpose(test_extract)))
            test_distance = np.sqrt(np.sum(np.square(test_new - center[0]), axis=1))
            apo = detect_realtime(s, train_distance, test_distance)
            # history_data.append([add[2:]])
            initial_data = np.append(testdata, [add[2:]], axis=0)
            initial_data = np.delete(initial_data, 0, 0)   # 滑动窗口，末尾加入并去掉第一个元素
            fault_time = fault_time + 1
            if apo != []:  # apo为警报
                print('n')
                break
            else:
                print('y')
    a1 = time.time()
    CNT = 0
    time1 = time.perf_counter()
    while True:
        if time.perf_counter() - time1 >= 0.01:
            CNT = CNT + 1
            time1 = time.perf_counter()
            aa = dll.FIB2125_ReadDouble(0, 0x1008, add, l_len)  # 数组
            diagnosis_data.append(add[2:])
            print(add[:])
        if CNT == 33:    #拿出33个数据  330ms
            break
    # history_data = np.array(history_data)
    # history_data = np.reshape(history_data, (history_data.shape[0], history_data.shape[2]))
    # initial_data = np.reshape(initial_data, (initial_data.shape[0], initial_data.shape[2]))
    # print(initial_data)
    # diagnosis_data = np.append(initial_data[-1*counter - 7:, :], np.array(diagnosis_data), axis=0)
    diagnosis_data = scaler.transform(diagnosis_data)
    test_extract = np.array(temporalize(diagnosis_data, 5))
    test_extract = np.reshape(test_extract, (test_extract.shape[0], 5, test_extract.shape[3]))
    pred = model(test_extract, training=False)
    test_pred = data_flatten(pred)
    test_real = data_flatten(test_extract)
    test_error = np.array(test_real - test_pred)
    test_fea = test_error.T
    x_test = np.reshape(test_fea, (1, test_fea.shape[0], test_fea.shape[1]))
    fault_type = model_classify(x_test, training=False)
    fault_type_max = np.argmax(fault_type[0])

    if fault_type_max == 0:
        fault_position = song_location(x_test, training=False)
        fault_position_index = duomian[np.argmax(fault_position[0])]
        # fault_diagnosis = fault_position_index + '发生松浮故障'
        print('松浮', fault_position_index)

    elif fault_type_max == 1:
        fault_position = que_location(x_test, training=False)
        fault_position_index = np.argmax(fault_position[0])
        if fault_position_index == 0:
            fault_estimation = rqs1_estimation(x_test, training=False)
        elif fault_position_index == 1:
            fault_estimation = rqs2_estimation(x_test, training=False)
        elif fault_position_index == 2:
            fault_estimation = rqs3_estimation(x_test, training=False)
        elif fault_position_index == 3:
            fault_estimation = rqs4_estimation(x_test, training=False)
        fault_estimation_index = chengdu[np.argmax(fault_estimation[0])]
        # fault_diagnosis = duomian[fault_position_index] + '发生' + fault_estimation_index + '缺损故障'
        print('舵面缺损', duomian[fault_position_index], fault_estimation_index)

    elif fault_type_max == 2:
        fault_position = ka_location(x_test, training=False)
        fault_position_index = duomian[np.argmax(fault_position[0])]
        # fault_diagnosis = fault_position_index + '发生卡死故障'
        print('卡死',fault_position_index)

    elif fault_type_max == 3:
        fault_estimation = th_estimation(x_test, training=False)
        fault_estimation_index = chengdu[np.argmax(fault_estimation[0])]
        # fault_diagnosis = '发生' + fault_estimation_index + '推力故障'
        print('推力', fault_estimation_index)

    elif fault_type_max == 4:
        fault_position = ls_location(x_test, training=False)
        fault_position_index = np.argmax(fault_position[0])
        if fault_position_index == 0:
            fault_estimation = ls1_estimation(x_test, training=False)
        elif fault_position_index == 1:
            fault_estimation = ls2_estimation(x_test, training=False)
        fault_estimation_index = chengdu[np.argmax(fault_estimation[0])]
        # fault_diagnosis = shenglimian[fault_position_index] + '发生' + fault_estimation_index + '升力面故障'
        print(shenglimian[fault_position_index], fault_estimation_index)

    #print(type(fault_type_max))
    #print(type(fault_position_index))
    rc_fault = (c.c_double * 3)(1, fault_type_max, np.argmax(fault_position[0]))  # 数组发送
    #print(type(fault_type_max))
    #print(type(fault_position_index))
    aa = dll.FIB2125_WriteDouble(0, 0x1600, rc_fault, 3)  # 单个数据

    fault_type_max = np.argsort(fault_type[0])[-2]

    if fault_type_max == 0:
        fault_position = song_location(x_test, training=False)
        fault_position_index = duomian[np.argmax(fault_position[0])]
        # fault_diagnosis = fault_position_index + '发生松浮故障'
        print('松浮', fault_position_index)

    elif fault_type_max == 1:
        fault_position = que_location(x_test, training=False)
        fault_position_index = np.argmax(fault_position[0])
        if fault_position_index == 0:
            fault_estimation = rqs1_estimation(x_test, training=False)
        elif fault_position_index == 1:
            fault_estimation = rqs2_estimation(x_test, training=False)
        elif fault_position_index == 2:
            fault_estimation = rqs3_estimation(x_test, training=False)
        elif fault_position_index == 3:
            fault_estimation = rqs4_estimation(x_test, training=False)
        fault_estimation_index = chengdu[np.argmax(fault_estimation[0])]
        # fault_diagnosis = duomian[fault_position_index] + '发生' + fault_estimation_index + '缺损故障'
        print('舵面缺损', duomian[fault_position_index], fault_estimation_index)

    elif fault_type_max == 2:
        fault_position = ka_location(x_test, training=False)
        fault_position_index = duomian[np.argmax(fault_position[0])]
        # fault_diagnosis = fault_position_index + '发生卡死故障'
        print('卡死', fault_position_index)

    elif fault_type_max == 3:
        fault_estimation = th_estimation(x_test, training=False)
        fault_estimation_index = chengdu[np.argmax(fault_estimation[0])]
        # fault_diagnosis = '发生' + fault_estimation_index + '推力故障'
        print('推力', fault_estimation_index)

    elif fault_type_max == 4:
        fault_position = ls_location(x_test, training=False)
        fault_position_index = np.argmax(fault_position[0])
        if fault_position_index == 0:
            fault_estimation = ls1_estimation(x_test, training=False)
        elif fault_position_index == 1:
            fault_estimation = ls2_estimation(x_test, training=False)
        fault_estimation_index = chengdu[np.argmax(fault_estimation[0])]
        # fault_diagnosis = shenglimian[fault_position_index] + '发生' + fault_estimation_index + '升力面故障'
        print(shenglimian[fault_position_index], fault_estimation_index)
    a2 = time.time()
    print("故障诊断所用时间：{}ms".format((a2-a1)*1000))

    # fault_position = ka_location.predict(x_test)
    # print(fault_position)
    # a2 = time.time()
    # print(a2-a1)
    # print(np.mean(zongfp))
    # print(np.mean(zongpf))
