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
sub_path = 'IF/seed=0/2797_increase/'
seed = 0


explanation_dict = pickle.load(open(path_project + f'IF/seed=0/explanation_dict.pkl', 'rb'))
explanation_df = pd.DataFrame(explanation_dict)
df = explanation_df.sort_values(by='sample_id')

# 绘制直方图
plt.scatter(df['sample_id'], df['influence'])
plt.axhline(y=-0, color='black', linestyle='-')
# plt.axhline(y=-400, color='r', linestyle='--')
plt.xlabel('sample_id')
plt.ylabel('influence')
test_id = df.iloc[0]['test_id_original']
plt.title(f'Influence of training samples to Test sample {int(test_id)}')
plt.show()

# Load the training set and test set
X_train = np.load(path_project + f'data_seed={seed}/X_train.npy')
X_test = np.load(path_project + f'data_seed={seed}/X_test.npy')
Y_train = np.load(path_project + f'data_seed={seed}/Y_train.npy')
Y_test = np.load(path_project + f'data_seed={seed}/Y_test.npy')
ID_train = np.load(path_project + f'data_seed={seed}/ID_train.npy')
ID_test = np.load(path_project + f'data_seed={seed}/ID_test.npy')


# 导入模型
def load_all_h5_models(directory):
    # 使用glob找到目录中所有的.h5文件
    h5_files = glob.glob(os.path.join(directory, '*.h5'))

    # 加载所有的模型并存储在一个字典中
    models = {}
    for file in h5_files:
        model_name = os.path.basename(file).split('.')[0]
        models[model_name] = load_model(file)

    return models

def load_all_pickle_files(directory):
    # 使用glob找到目录中所有的.h5文件
    pickle_files = glob.glob(os.path.join(directory, '*.pkl'))

    # 加载所有的模型并存储在一个字典中
    files = {}
    for file in pickle_files:
        file_name = os.path.basename(file).split('.')[0]
        files[file_name] = pickle.load(open(file, 'rb'))

    return files

# 使用函数加载目录下所有的.h5模型
directory = path_project + sub_path
all_models = load_all_h5_models(directory)
all_pickle_files = load_all_pickle_files(directory)

# 输出加载的模型
for model_name, model in all_models.items():
    print(f'Model {model_name} is loaded.')
# print
print(f'test_id: {explanation_dict["test_id_original"]}')
print(f'label: {np.argmax(Y_test[explanation_dict["test_id"]])}')

for model_name, model in all_models.items():
    prediction = np.argmax(model.predict(np.expand_dims(X_test[explanation_dict['test_id']], axis=0), verbose=0))
    print(f'{model_name} prediction: {prediction}')


# 绘制验证集的损失函数
plt.figure(figsize=(10, 10))
for file_name, history in all_pickle_files.items():
    plt.plot(history['val_loss'], label=file_name)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.legend()
plt.show()

# 绘制验证集的准确率
plt.figure(figsize=(10, 10))
for file_name, history in all_pickle_files.items():
    plt.plot(history['val_accuracy'], label=file_name)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.show()

