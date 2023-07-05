import scipy.io as scio
import numpy as np
from sklearn.preprocessing import RobustScaler,MinMaxScaler
from sklearn.manifold import LocallyLinearEmbedding,Isomap,TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.decomposition import KernelPCA,PCA
from sklearn import svm
import feature_extract
from spot import bidSPOT,biSPOT,dSPOT
from keras.models import Sequential
from keras.layers import Dense
from POT import pot, poty
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

def SD_FluxEV(train_distance, test_distance, risk, init_level):
    """
    :param X: 输入经过预处理的数据
    :param p: p period
    :param s: window sizes
    :param d: half of the window sizes to handle data drift
    :param l: period length
    :param k: window size
    :param risk: risk coefficient
    :return: 主要的是r，若r[i]为1，则该点异常；S为平滑后的数据
    """
    n = len(test_distance)
    list1 = []
    list2 = []
    thf_list = [None for _ in range(n)]
    t_list = [None for _ in range(n)]
    r = [None for _ in range(n)]
    # 先用S算初始阈值
    thf, t, y = pot.pot(train_distance, risk=risk, init_level=init_level)
    list1.append(thf)
    list2.append(t)
    y = list(y)
    nt = len(y)
    # 对每个i > k,更新阈值
    for i in range(len(test_distance)):
        r[i] = 0
        if test_distance[i] > thf:
            r[i] = 1
        elif test_distance[i] > t:
            yi = test_distance[i] - t
            y.append(yi)
            nt += 1
            thf = poty.poty(np.array(y), risk=risk, num_candidates=nt, t=t)
        else:
            pass
    # 如果是异常点，该点的特征舍去
        list1.append(thf)
    n1 = len(list1)
    n2 = len(list2)
    thf_list[n-n1: n] = list1[0: n1]
    t_list[n-n2: n] = list2[0: n2]
    return r, thf_list, t_list,list1
def selfscaler(traindata,testdata):
    iq = np.percentile(traindata,(25,75),axis=0)
    fenmu = iq[1, :] - iq[0, :]
    zhongzhi = np.median(traindata,axis=0)
    x1 = (traindata - zhongzhi)/fenmu
    x2 = (zhongzhi - traindata)/fenmu
    x1[x1<0] = 0
    x2[x2<0] = 0
    xtrain = x1 + x2
    t1 = (testdata - zhongzhi)/fenmu
    t2 = (zhongzhi - testdata)/fenmu
    t1[t1<0] = 0
    t2[t2<0] = 0
    xtest = t1 + t2
    return xtrain,xtest
def data_preprocess(path1):
    rty = scio.loadmat(path1)
    rty = rty['RTY_XYZ']
    data = rty[:, :11]
    data = np.concatenate((data, rty[:, 15:16]), axis=1)
    data = np.array(data)
    return data
def banwuli_preprocess(path1,path2):
    rty = scio.loadmat(path1)
    rty = rty['RTY_XYZ']
    xe = scio.loadmat(path2)
    xe = xe['xe']
    data = np.concatenate((rty, xe[:, 1:2]), axis=1)
    data = np.array(data)
    return data
def get_normal(path1):
    rty = scio.loadmat(path1)
    rty = rty['RTY_XYZ']
    data = rty[:, :11]
    data = np.concatenate((data, rty[:, 15:16]), axis=1)
    data = np.array(data)
    return data
def get_features(phidata,ts_length):
    n,m = np.shape(phidata)
    f_std = np.zeros((n-ts_length+1,m))
    x_var = np.zeros((n-ts_length+1,m))
    x_norm = np.zeros((n - ts_length + 1, m))
    x_range = np.zeros((n - ts_length + 1, m))
    x_en = np.zeros((n - ts_length + 1, m))
    for i in range(m):
        for j in range(n-ts_length+1):
            pro_data = phidata[j:j+ts_length-1,i]
            f,y = feature_extract.do_fft(pro_data,200)
            f_std[j,i] = feature_extract.get_fre_domain_features(f,y)
            x_var[j,i] = feature_extract.get_time_domain_features(pro_data)
            nn = feature_extract.get_norm(pro_data)#[0]范数[1]极差
            x_norm[j,i] = nn[0]
            x_range[j,i] = nn[1]
            x_en[j,i] = feature_extract.xb_energy(pro_data, 'db5', 3)[0]
    dnorm = np.diff(x_norm,axis=0)
    zero_line = np.zeros((1,m))
    dnorm_value = np.concatenate((dnorm,zero_line),axis=0)
    extract_value = np.concatenate((x_var,x_range,f_std,x_en,x_norm,dnorm_value),axis=1)
    return extract_value
def pro_matirx(low_mat,high_mat):
    mat1 = np.dot(np.transpose(low_mat),high_mat)
    mat2 = np.dot(np.transpose(high_mat),high_mat)
    mat = np.dot(mat1,np.linalg.pinv(mat2))
    return mat
def timewindow(timeseries,ts_length):#滑动时间窗口法
    n,m = np.shape(timeseries)
    std_value = np.zeros((n-ts_length+1,m))
    range_value = np.zeros((n-ts_length+1,m))
    norm_value = np.zeros((n - ts_length + 1, m))
    ji_value = np.zeros((n - ts_length + 1, m))
    # min_value = np.zeros((n - ts_length + 1, m))
    for i in range(m):
        for j in range(0,n-ts_length+1):
            ts = timeseries[j:j+ts_length-1,i]
            std_value[j,i] = np.std(ts)
            range_value[j,i] = np.max(ts) - np.min(ts)
            ji_value[j,i] = max(abs(np.max(ts)),abs(np.min(ts)))
            norm_value[j,i] = np.linalg.norm(ts,ord=2)
    dnorm = np.diff(norm_value,axis=0)
    zero_line = np.zeros((1,m))
    dnorm_value = np.concatenate((dnorm,zero_line),axis=0)
    extract_value = np.concatenate((std_value,range_value),axis=1)
    return extract_value
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
        axins = ax.inset_axes((0.08,0.45,0.35,0.25))
        axins.plot(x_ticks,test_distance,color = colors[0])
        axins.plot([train_end / 100 + 3,train_end / 100 + len(test_distance) / 100 - 0.01 + 3],[cl,cl], color = colors[3])
        zone_left = int((fault_time - 1)*100-300)
        zone_right = int((fault_time + 1)*100-300)
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
        q = 1e-3 # risk parameter
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
    return train_distance*c, test_distance
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

    return train_distance, test_distance
def plot_1(data,title,fig_number:int):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
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
            i = i+1
            plt.subplot(fig_row,2,i)
            plt.plot(data[:, i-1])
            plt.ylabel(title[i-1])
            plt.xlabel('采样点')
    plt.show()
def getnormaldata():
    path1 = 'F:\微信文件\研一上大作业\DD\demo_x\demoV9-7.07\data\\normal\\normal1.mat'
    normal_data = get_normal(path1)
    alltrain = []
    for i in range(2):
        path2 = 'F:\微信文件\研一上大作业\DD\demo_x\demoV9-7.07\data\\normal\\{}.mat'.format('normal'+str(i+2))
        data = data_preprocess(path2)
        traindata = data[3000:9000, :]
        scaler = RobustScaler().fit(normal_data[3000:8000,:])
        xtrain = scaler.transform(traindata)
        alltrain.append(xtrain)
    train_data = np.vstack(alltrain)
    return train_data

if __name__ == '__main__':
    zongfp,zongpf = [],[]
    for i in range(1):
        path2 = 'F:\微信文件\研一上大作业\DD\demo_x\demoV9-7.07\data\\lsloss\\ls{}.mat'.format(0.6)
        # path2 = 'F:\微信文件\研一上大作业\DD\毕设\数据\半物理平台数据\\2-que-30\\rty_xyz.mat'
        # path3 = 'F:\微信文件\研一上大作业\DD\毕设\数据\半物理平台数据\\2-que-30\\xe.mat'
        path4 = 'zc1.mat'
        #data = data_preprocess(path4)
        # data = banwuli_preprocess(path2,path3)
        normal_data = get_normal(path4)
        datalen, dim = np.shape(data)


        train_end = 3000#2000
        fault_start = 8000
        window_length = 300
        xtrain = getnormaldata()#1000-10000
        testdata = data[train_end:-1,:]
        scaler = RobustScaler().fit(normal_data[3000:8000,:])
        # xtrain = scaler.fit_transform(traindata)
        xtest = scaler.transform(testdata)
        # model = PCA(n_components=0.95)
        # model.fit(xtrain)
        # X_new = model.fit_transform(xtrain)
        # print(model.get_params())
        # print(np.shape(X_new))
        # xtrain,xtest = selfscaler(traindata,testdata)
        # train_extract = timewindow(xtrain, window_length)
        train_extract = np.load('train_extract.npy')
        test_extract = timewindow(xtest, window_length)
        print(np.shape(train_extract))
        print(np.shape(test_extract))
        # lle_train_data = LocallyLinearEmbedding(n_neighbors=5, n_components=9 ).fit_transform(train_extract)
        # A = pro_matirx(lle_train_data, train_extract)
        # np.save('projection.npy',A)
        # np.save('lle_train_data.npy',lle_train_data)
        A = np.load('projection.npy')
        lle_train_data = np.load('lle_train_data.npy')
        test_new = np.transpose(np.dot(A, np.transpose(test_extract)))
        x_train_new = np.transpose(np.dot(A, np.transpose(train_extract)))
        train_distance, test_distance = km_cluster(lle_train_data, test_new, x_train_new)
        # lle = LocallyLinearEmbedding(n_neighbors=8, n_components=14)
        # lle_train_data = lle.fit_transform(train_extract)
        # test_new = lle.transform(test_extract)
        # print(lle_train_data.shape)
        # train_distance, test_distance = km_cluster1(lle_train_data, test_new)
        # test_distance[7000:9000] = 0
        fp,pf = detect_accurate(train_distance, test_distance, window_length, train_end, fault_start, mode='auto')
        zongfp.append(fp)
        zongpf.append(pf)
    print(np.mean(zongfp))
    print(np.mean(zongpf))


    # A = pro_matirx(lle_train_data,train_extract1)
    # test_new = np.transpose(np.dot(A,np.transpose(test_extract1)))
    # x_train_new = np.transpose(np.dot(A,np.transpose(train_extract1)))
    # print(x_train_new.shape)
    # train_distance,test_distance = km_cluster(lle_train_data,test_new,x_train_new)
    # my_plot(train_distance,test_distance)
    # detect_accurate(train_distance,test_distance)

    # clf = svm.OneClassSVM(kernel='rbf', gamma='scale')
    # clf.fit(lle_train_data)
    # # clf.fit(kpca_train_data)
    # train_score = -clf.decision_function(x_train_new)
    # test_score = -clf.decision_function(test_new)
    # print(clf.score_samples(lle_train_data))
    # detect_accurate(train_score, test_score, window_length, train_end, fault_start, 'max')