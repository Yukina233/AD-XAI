import os
import math
import cmath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_project = '/home/yukina/Missile_Fault_Detection/project'



data_dir = os.path.join(path_project, 'adversarial_ensemble_AD/scripts/data/lam.CSV')

df = pd.read_csv(data_dir)

lam = df['lam']
aucroc = df['AUCROC']
aucpr = df['AUCPR']
fdr = df['FDR']
far = df['FAR']

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.figure()
plt.title("AUCROC和AUCPR指标随lambda参数的变化曲线图")

x = [1, 2, 3, 4, 5, 6, 7]
# 绘制四条折线
plt.plot(x, 0.01 * aucroc, label='AUCROC')
plt.plot(x, 0.01 * aucpr, label='AUCPR')


# 自定义x轴的刻度位置和标签
custom_ticks = [1, 2, 3, 4, 5, 6, 7]  # 自定义刻度位置
custom_labels = lam  # 自定义标签

plt.xticks(custom_ticks, custom_labels)  # 设置x轴刻度位置和标签

plt.xlabel('lambda')
plt.ylabel('性能指标')
# 添加图例
plt.legend()

# 展示图形
plt.show()

plt.cla()
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.figure()
plt.title("FDR和FAR指标随lambda参数的变化曲线图")

x = [1, 2, 3, 4, 5, 6, 7]
# 绘制四条折线
plt.plot(x, 0.01 * fdr, label='FDR', color='green')
plt.plot(x, 1- 0.01 * far, label='1-FAR', color='red')

# 自定义x轴的刻度位置和标签
custom_ticks = [1, 2, 3, 4, 5, 6, 7]  # 自定义刻度位置
custom_labels = lam  # 自定义标签

plt.xticks(custom_ticks, custom_labels)  # 设置x轴刻度位置和标签

plt.xlabel('lambda')
plt.ylabel('性能指标')
# 添加图例
plt.legend()

# 展示图形
plt.show()
