from sklearn.cluster import KMeans
#from spot import dSPOT
from scipy import stats
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

path_project = '/home/yukina/Missile_Fault_Detection/project/'

class fault_detect():
    def __init__(self):
        """
        初始必要参数，故障类型，滑动窗口步长
        """
        self.fault_class = ["KS_Fault", "SF_Fault", "QS_Fault", "all"]  # 故障类型，卡死故障，松浮故障，缺损故障
        self.fault = "all"                                              # 当前的故障类型
        self.slide_window = 100                                                   # 滑动窗口大小
        self.start_time = 2000       # 20s后开始平飞

        # 数据的读取
        self.nor_datapath_orig = path_project + 'data/banwuli_data/normal/zc1.dat'                             # 初始正常数据路径 ..\表示上级目录
        self.fau_datapath_orig = path_project + 'data/banwuli_data/normal/zc1.dat'                             # 故障数据路径
        self.param = [['dOmega_ib_b[0]', 'dOmega_ib_b[1]', 'dOmega_ib_b[2]', 'Gama', 'Theta_k', 'Psi_t', 'fb[0]', 'fb[1]', 'fb[2]', 'Alfa', 'Beta', 'zmb'],
                      [],
                      [],
                      ['Gama', 'Alfa', 'Omega_ib_b[1]', 'Theta_k', 'Beta', 'Psi_t']
        ]                                                # 0,1,2,3元素分别为卡死故障，松浮故障，缺损故障,all(所有故障)对应的参数

        self.feature = [[],
                        [],
                        [],
                        ['ptp', 'max', 'ptp', 'kurt', 'clearance', 'kurt', 'ptp', 'skew', 'skew']
        ]                                               # 提取的特征

        self.feature_head = [[],
                             [],
                             [],
                             ['Gama', 'Alfa', 'Alfa', 'Omega_ib_b[1]', 'Omega_ib_b[1]', 'Theta_k', 'Theta_k', 'Beta', 'Psi_t']
        ]                                              # 特征对应的参量，维数必须和特征的位数对应
        self.nor_data_orig = self.read_dat(self.nor_datapath_orig)                                      # 获取正常数据
        self.fau_data_orig = self.read_dat(self.fau_datapath_orig)                                      # 获取故障数据

    """
    输入数据的路径，并根据self.fault选择提取的数据特征
    根据不同的故障，选择不同的参量进行处理
    """
    def read_dat(self, path):
        a = pd.read_csv(path, delim_whitespace=True)
        return np.array(a[self.param[self.fault_class.index(self.fault)]], dtype=float)[self.start_time:,:]

    # 输入data为(s*1)，s是滑动窗口大小，para为str，为特征的名称
    def circu_para(self,data,para):
        data = np.array(data)
        if para == 'ptp':                  # 极差
            return max(data) - min(data)
        elif para == 'max':                #峰值
            return max(abs(data))
        elif para == 'kurt':               # 峭度
            return stats.kurtosis(data,fisher=False)
        elif para == 'clearance':         # 裕度
            return (max(data) - min(data))/(np.mean(np.sqrt(abs(data)), axis=-1) ** 2)
        elif para == 'skew':              # 偏度
            return stats.skew(data)

    # 输入一个滑动窗口（s）大小，m维的数据（s*m），输出一行，n维的数据（1*n），n维特征的个数
    # idx为需要输出的特征对应的索引
    def slide_param(self, data, idx):
        data_para = np.transpose(np.array(data)[:,idx])
        return [self.circu_para(data_para[id],self.feature[self.fault_class.index(self.fault)][id]) for id in range(len(self.feature[self.fault_class.index(self.fault)]))]

    # 滑动窗口数据，获得训练集和测试集
    def slidewindow_data(self, data):
        # 根据特征找到每一个特征对应的参量的索引
        idx = [self.param[self.fault_class.index(self.fault)].index(feature) for feature in self.feature_head[self.fault_class.index(self.fault)]]
        # 按照滑动窗口的大小开始滑动
        slide_data = np.array([self.slide_param(data[flag:flag+self.slide_window-1, :],idx) for flag in range(len(data)-self.slide_window+1)])
        print(slide_data)

if __name__ == "__main__":
    test = fault_detect()
    print(test.slidewindow_data(test.nor_data_orig))



