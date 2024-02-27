import glob
import os

import numpy as np
from matplotlib import pyplot as plt

# 用来验证该数据集是否满足domain的通常假设

path_project = '/home/yukina/Missile_Fault_Detection/project'

root_dir = path_project + '/data/iot_data'

data_all = []
files = os.listdir(root_dir)
for file in files:
    data_all.append(np.load(os.path.join(root_dir, file), allow_pickle=True))

for id in range(0, 5):
    x = np.arange(0, data_all[id]['X_train'].shape[0])
    y = data_all[id]['X_train'][:,1]
    plt.plot(x, y, label=f'task{id}', alpha=0.5)

plt.ylim(0, 1)
plt.legend()
plt.show()

print('Done')