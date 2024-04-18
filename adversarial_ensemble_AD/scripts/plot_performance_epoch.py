import os
import math
import cmath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_project = '/home/yukina/Missile_Fault_Detection/project'



data_dir = os.path.join(path_project, 'adversarial_ensemble_AD/scripts/data/epoch.CSV')

df = pd.read_csv(data_dir)

aucroc = df['AUCROC']
aucpr = df['AUCPR']
fdr = df['FDR']
far = df['FAR']

x = [0, 1, 2, 3, 4]
plt.figure()
plt.title("折线图")

# 绘制四条折线
plt.plot(x, 0.01 * aucroc, label='AUCROC')
plt.plot(x, 0.01 * aucpr, label='AUCPR')
plt.plot(x, 0.01 * fdr, label='FDR')
plt.plot(x, 1- 0.01 * far, label='1-FAR')

# 添加图例
plt.legend()

# 展示图形
plt.show()
