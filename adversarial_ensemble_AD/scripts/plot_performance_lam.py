import os
import math
import cmath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_project = '/home/yukina/Missile_Fault_Detection/project'



data_dir = os.path.join(path_project, 'adversarial_ensemble_AD/scripts/data/lam1.CSV')

df = pd.read_csv(data_dir)

lam1 = df['lam1']
aucroc = df['AUCROC']
aucpr = df['AUCPR']
fdr = df['FDR']
far = df['FAR']

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.figure(figsize=(5, 5))
plt.title("AUCROC、AUCPR和FDR指标随lambda_1参数的变化曲线图")

x = [i for i in range(1, 6 + 1)]
# 绘制四条折线
plt.plot(x, 0.01 * aucroc, label='AUCROC')
plt.plot(x, 0.01 * aucpr, label='AUCPR')
plt.plot(x, 0.01 * fdr, label='FDR', color='green')

# 自定义x轴的刻度位置和标签
custom_ticks = [i for i in range(1, 6 + 1)]  # 自定义刻度位置
custom_labels = lam1  # 自定义标签

plt.xticks(custom_ticks, custom_labels)  # 设置x轴刻度位置和标签

plt.xlabel('lambda_1')
plt.ylabel('性能指标')
# 添加图例
plt.legend()

# 展示图形
plt.show()

plt.cla()
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.figure(figsize=(5, 5))
plt.title("FAR指标随lambda_1参数的变化曲线图")

x = [i for i in range(1, 6 + 1)]
# 绘制四条折线

plt.plot(x, 0.01 * far, label='FAR', color='red')

# 自定义x轴的刻度位置和标签
custom_ticks = [i for i in range(1, 6 + 1)]  # 自定义刻度位置
custom_labels = lam1  # 自定义标签

plt.xticks(custom_ticks, custom_labels)  # 设置x轴刻度位置和标签

plt.xlabel('lambda_1')
plt.ylabel('性能指标')
# 添加图例
plt.legend()

# 展示图形
plt.show()


plt.cla()
data_dir = os.path.join(path_project, 'adversarial_ensemble_AD/scripts/data/lam2.CSV')

df = pd.read_csv(data_dir)

lam2 = df['lam2']
aucroc = df['AUCROC']
aucpr = df['AUCPR']
fdr = df['FDR']
far = df['FAR']

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.figure(figsize=(5, 5))
plt.title("AUCROC、AUCPR和FDR指标随lambda_2参数的变化曲线图")

x = [i for i in range(1, 9 + 1)]
# 绘制四条折线
plt.plot(x, 0.01 * aucroc, label='AUCROC')
plt.plot(x, 0.01 * aucpr, label='AUCPR')
plt.plot(x, 0.01 * fdr, label='FDR', color='green')

# 自定义x轴的刻度位置和标签
custom_ticks = [i for i in range(1, 9 + 1)]  # 自定义刻度位置
custom_labels = lam2  # 自定义标签

plt.xticks(custom_ticks, custom_labels)  # 设置x轴刻度位置和标签

plt.xlabel('lambda_2')
plt.ylabel('性能指标')
# 添加图例
plt.legend()

# 展示图形
plt.show()

plt.cla()
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.figure(figsize=(5, 5))
plt.title("FAR指标随lambda_2参数的变化曲线图")

x = [i for i in range(1, 9 + 1)]
# 绘制四条折线

plt.plot(x, 0.01 * far, label='FAR', color='red')

# 自定义x轴的刻度位置和标签
custom_ticks = [i for i in range(1, 9 + 1)]  # 自定义刻度位置
custom_labels = lam2  # 自定义标签

plt.xticks(custom_ticks, custom_labels)  # 设置x轴刻度位置和标签

plt.xlabel('lambda_2')
plt.ylabel('性能指标')
# 添加图例
plt.legend()

# 展示图形
plt.show()
