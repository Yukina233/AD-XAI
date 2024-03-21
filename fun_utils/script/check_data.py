import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

path_project = '/home/yukina/Missile_Fault_Detection/project/'

def read_dat(path):
    a = pd.read_csv(path, delim_whitespace=True)
    b = a[['dOmega_ib_b[0]', 'dOmega_ib_b[1]', 'dOmega_ib_b[2]', 'Gama', 'Theta_k', 'Psi_t', 'fb[0]', 'fb[1]',
           'fb[2]', 'Alfa', 'Beta', 'zmb', 'P']]
    b = np.array(b, dtype=float)
    return b

def plot_data(data):

    fig, ax = plt.subplots(3, 2, figsize=(12, 10), dpi=200, tight_layout=True)
    x_ticks = np.linspace(0, data.shape[0]/100, data.shape[0])
    ax[0,0].grid(which='major', ls='--', alpha=.8, lw=.8)
    ax[0,0].plot(x_ticks,data[:,0], label='x轴角加速度', alpha=0.8)
    ax[0,0].plot(x_ticks,data[:,1], label='y轴角加速度', alpha=0.8)
    ax[0,0].plot(x_ticks,data[:,2], label='z轴角加速度', alpha=0.8)
    xticks = np.arange(0, data.shape[0]/100, 10)
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
    ax[0,1].plot(x_ticks,data[:, 3], label='滚转角', alpha=0.8)
    ax[0,1].plot(x_ticks,data[:, 4], label='俯仰角', alpha=0.8)
    ax[0,1].plot(x_ticks,data[:, 5], label='偏航角', alpha=0.8)
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
    ax[1,0].plot(x_ticks,data[:, 6], label='x轴加速度', alpha=0.8)
    ax[1,0].plot(x_ticks,data[:, 7], label='y轴加速度', alpha=0.8)
    ax[1,0].plot(x_ticks,data[:, 8], label='z轴加速度', alpha=0.8)

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
    ax[1,1].plot(x_ticks,data[:, 9], label='迎角', alpha=0.8)
    ax[1,1].plot(x_ticks,data[:, 10], label='侧滑角', alpha=0.8)
    ax[1,1].set_xticks(xticks)
    ax[1,1].set_xticklabels(xticks, fontdict={'family': 'Times New Roman', 'size': 10})
    yticks = ax[1,1].get_yticks()
    ax[1,1].set_yticks(yticks)
    ax[1,1].set_yticklabels(yticks, fontdict={'family': 'Times New Roman', 'size': 10})
    ax[1,1].set_xlabel('时间(秒)', fontdict={'family': ['SimSun'], 'size': 12})
    ax[1,1].set_ylabel('气动角(度)', fontdict={'family': ['SimSun'], 'size': 12})
    ax[1,1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax[1,1].legend(loc='best', prop={'family': ['SimSun'], 'size': 12})

    ax[2, 0].grid(which='major', ls='--', alpha=.8, lw=.8)
    ax[2, 0].plot(x_ticks, data[:, 11], label='z轴偏移', alpha=0.8)
    ax[2, 0].set_xticks(xticks)
    ax[2, 0].set_xticklabels(xticks, fontdict={'family': 'Times New Roman', 'size': 10})
    yticks = ax[2, 0].get_yticks()
    ax[2, 0].set_yticks(yticks)
    ax[2, 0].set_yticklabels(yticks, fontdict={'family': 'Times New Roman', 'size': 10})
    ax[2, 0].set_xlabel('时间(秒)', fontdict={'family': ['SimSun'], 'size': 12})
    ax[2, 0].set_ylabel('z轴偏移', fontdict={'family': ['SimSun'], 'size': 12})
    ax[2, 0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax[2, 0].legend(loc='best', prop={'family': ['SimSun'], 'size': 12})

    ax[2, 1].grid(which='major', ls='--', alpha=.8, lw=.8)
    ax[2, 1].plot(x_ticks, data[:, 12], label='推力', alpha=0.8)
    ax[2, 1].set_xticks(xticks)
    ax[2, 1].set_xticklabels(xticks, fontdict={'family': 'Times New Roman', 'size': 10})
    yticks = ax[2, 1].get_yticks()
    ax[2, 1].set_yticks(yticks)
    ax[2, 1].set_yticklabels(yticks, fontdict={'family': 'Times New Roman', 'size': 10})
    ax[2, 1].set_xlabel('时间(秒)', fontdict={'family': ['SimSun'], 'size': 12})
    ax[2, 1].set_ylabel('推力', fontdict={'family': ['SimSun'], 'size': 12})
    ax[2, 1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax[2, 1].legend(loc='best', prop={'family': ['SimSun'], 'size': 12})

    plt.show()


normal_data = read_dat(path_project + "data/banwuli_data/normal/zc1.dat")
ks_data = read_dat(path_project + "data/banwuli_data/ks/1/ks1_-1.dat")
sf_data = read_dat(path_project + "data/banwuli_data/sf/1/sf1-1.dat")
rqs_data = read_dat(path_project + "data/banwuli_data/rqs/rqs-1/rqs-1-0.10.dat")
lqs_l_data = read_dat(path_project + "data/banwuli_data/lqs/lqs-l/lqs-l-0.22.dat")
lqs_r_data = read_dat(path_project + "data/banwuli_data/lqs/lqs-r/lqs-r-0.22.dat")
t_data = read_dat(path_project + "data/banwuli_data/T/1/T0.04.dat")

plot_data(rqs_data)