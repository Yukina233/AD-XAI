import os

import numpy as np


path_project = '/home/yukina/Missile_Fault_Detection/project'


data = np.load(os.path.join(path_project, 'data/MSL_SMAP_data/train/A-1.npy'))
print(data)