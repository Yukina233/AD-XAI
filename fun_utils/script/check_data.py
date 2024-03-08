import numpy as np
import pandas as pd

path_project = '/home/yukina/Missile_Fault_Detection/project/'

def read_dat(path):
    a = pd.read_csv(path, delim_whitespace=True)
    return np.array(a, dtype=float)

SF_Data = np.load(path_project + "data/banwuli_data/hhsdata/sf.npy")
KS_Data = np.load(path_project + "data/banwuli_data/hhsdata/ks.npy")
RQS_Data = np.load(path_project + "data/banwuli_data/hhsdata/rqs.npy")
LQS_Data = np.load(path_project + "data/banwuli_data/hhsdata/lqs.npy")
T_Data = np.load(path_project + "data/banwuli_data/hhsdata/T.npy")

normal_data = read_dat(path_project + "data/banwuli_data/normal/zc1.dat")
ks_data = read_dat(path_project + "data/banwuli_data/ks/1/ks1_-1.dat")
print('hhhhh')