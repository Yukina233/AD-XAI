import pickle

import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt

path_project = '/home/yukina/Missile_Fault_Detection/project/'
seed = 0

explanation_dict = pickle.load(open(path_project + f'data_seed={seed}/explanation_dict.pkl', 'rb'))
explanation_df = pd.DataFrame(explanation_dict)
df = explanation_df.sort_values(by='sample_id')

# 绘制直方图
plt.bar(df['sample_id'], df['influence'])
plt.axhline(y=-0.01, color='r', linestyle='--')
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
model_list = []
for i in range(0, 11):
    model = tf.keras.models.load_model(path_project + f'trac/output/model-remove-{i * 10}%.h5')
    model_list.append(model)

# print
print(f'test_id: {explanation_dict["test_id_original"]}')
print(f'label: {np.argmax(Y_test[explanation_dict["test_id"]])}')
prediction = np.argmax(model_list[0].predict(np.expand_dims(X_test[explanation_dict['test_id']], axis=0), verbose=0))
print(f'origin model prediction: {prediction}')
for i in range(1, 11):
    prediction = np.argmax(model.predict(np.expand_dims(X_test[explanation_dict['test_id']], axis=0), verbose=0))
    print(f'model-remove-{i * 10}% prediction: {prediction}')


history_list = []
for i in range(0, 11):
    history = pickle.load(open(path_project + f'trac/output/model-remove-{i * 10}%-history.pkl', 'rb'))
    history_list.append(history)

# 绘制验证集的损失函数
plt.figure(figsize=(10, 10))
for i in range(0, 11):
    plt.plot(history_list[i]['val_loss'], label=f'model-remove-{i * 10}%')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.legend()
plt.show()

# 绘制验证集的准确率
plt.figure(figsize=(10, 10))
for i in range(0, 11):
    plt.plot(history_list[i]['val_accuracy'], label=f'model-remove-{i * 10}%')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.show()

