import glob
import os
import pickle
import time

import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model


path_project = '/home/yukina/Missile_Fault_Detection/project/'

seed = 0

def load_all_explanation_dict(directory):
    # 使用glob找到目录中所有的.h5文件
    pickle_files = glob.glob(os.path.join(directory, 'explanation_dict-*.pkl'))
    # 加载所有的模型并存储在一个字典中
    files = {}
    for file in pickle_files:
        file_name = os.path.basename(file).split('.')[0]
        files[file_name] = pickle.load(open(file, 'rb'))

    return files

all_explanation_dict = load_all_explanation_dict(path_project + f'IF/seed=0/group_explanation/')
print('Load all explanation dict successfully.')
dicts = []
for file_name, explanation_dict in all_explanation_dict.items():
    dicts.append(explanation_dict)

zipped_lists = zip(*[d['influence'] for d in dicts])
sum_influence = [sum(values)/len(dicts[0]['sample_id']) for values in zipped_lists]
group_explanation_dict = {
    'sample_id_original': all_explanation_dict['explanation_dict-test_id=0']['sample_id_original'],
    'sample_id': all_explanation_dict['explanation_dict-test_id=0']['sample_id'],
    'train_feature': all_explanation_dict['explanation_dict-test_id=0']['train_feature'],
    'train_label': all_explanation_dict['explanation_dict-test_id=0']['train_label'],
    'predicted label': all_explanation_dict['explanation_dict-test_id=0']['predicted label'],
    'influence': sum_influence
}
# 保存到本地
pickle.dump(group_explanation_dict, open(path_project + f'IF/seed={seed}/group_explanation/group_explanation_dict.pkl', 'wb'))
print('Save group explanation dict successfully.')