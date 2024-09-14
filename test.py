import numpy as np

data_path = "/home/yukina/Missile_Fault_Detection/project/Anomaly-Transformer-main/data/SMD"
data = np.load(data_path + "/SMD_train.npy")

print(data.shape)