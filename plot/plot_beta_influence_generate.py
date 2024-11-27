import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE, Isomap, MDS
from sklearn.preprocessing import MinMaxScaler

path_project = '/home/yukina/Missile_Fault_Detection/project_data'

tsne_config = {
    'perplexity': 10
}

test_set_name = 'TLM-RATE'
num_samples = 500
np.random.seed(0)

param = 'alpha'

output_dir = os.path.join(path_project, f'{test_set_name}_dataset/plot_data/{param}')

train_path = os.path.join(path_project, f'data/{test_set_name}/yukina_data/DeepSAD_data, window=10, step=2',
                          'train.npz')
test_path = os.path.join(path_project, f'data/{test_set_name}/yukina_data/DeepSAD_data, window=10, step=2',
                            'test.npz')

generated_data_dir = os.path.join(path_project, f'{test_set_name}_dataset/plot_data', param)

init_train_data = np.load(train_path)['X_train']
sampled_init_train_data = init_train_data[
    np.random.choice(range(0, init_train_data.shape[0]), num_samples*2, replace=True)]

test_data = np.load(test_path)['X_test']
test_y = np.load(test_path)['y_test']
anomaly_data = test_data[np.where(test_y == 1)]
sampled_anomaly = anomaly_data[np.random.choice(range(0, anomaly_data.shape[0]), num_samples*2, replace=True)]

beta_list = [0, 0.1, 1, 2.5]
generated_data = []
for beta in beta_list:
    data = np.load(os.path.join(generated_data_dir, f'{beta}.npz'))['X']
    generated_data.append(data[np.random.choice(range(0, data.shape[0]), num_samples, replace=True)])

all_generated_data = np.concatenate(generated_data)
X_plot = np.concatenate((sampled_init_train_data, all_generated_data))
Y_plot = np.zeros(num_samples*2)
# Y_plot = np.concatenate((Y_plot, -np.ones(num_samples*2)))
for i, beta in enumerate(beta_list):
    Y_plot = np.concatenate((Y_plot, np.ones(num_samples) * (i + 1)))

# tsne1 = TSNE(n_components=2, random_state=0, perplexity=tsne_config['perplexity'])  # n_components表示目标维度
#
# # 创建MinMaxScaler对象
# scaler1 = MinMaxScaler()
# # 对数据进行归一化
# normalized_data = scaler1.fit_transform(X_plot)
# X_2d = tsne1.fit_transform(normalized_data)  # 对数据进行降维处理
#
# plt.rcParams['font.sans-serif'] = ['SimSun']
# plt.figure(figsize=(12, 9))
#
# plt.scatter(X_2d[Y_plot == 0, 0], X_2d[Y_plot == 0, 1], label=f'normal data', alpha=0.5)
# for i, beta in enumerate(beta_list):
#     plt.scatter(X_2d[Y_plot == (i + 1), 0], X_2d[Y_plot == (i + 1), 1], label=f'generated data, beta={beta}', alpha=0.5)
#
# plt.legend()
#
# plt.xlabel('Dimension 1')
# plt.ylabel('Dimension 2')
# plt.savefig(os.path.join(output_dir, f'TSNE1 of Generated Data.jpg'), dpi=330, format='jpg')
# plt.close()

embeddings = MDS(n_components=2, random_state=0, normalized_stress='auto')  # n_components表示目标维度

# 创建MinMaxScaler对象
scaler1 = MinMaxScaler()
# 对数据进行归一化
normalized_data = scaler1.fit_transform(X_plot)
X_2d = embeddings.fit_transform(normalized_data)  # 对数据进行降维处理

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.figure(figsize=(8, 6))


for i, beta in enumerate(beta_list):
    plt.scatter(X_2d[Y_plot == 0, 0], X_2d[Y_plot == 0, 1], label=f'normal data', alpha=0.5, color='black')
    plt.scatter(X_2d[Y_plot == -1, 0], X_2d[Y_plot == -1, 1], label=f'anomaly data', alpha=0.5, color='red')
    plt.scatter(X_2d[Y_plot == (i + 1), 0], X_2d[Y_plot == (i + 1), 1], label=f'generated data, beta={beta}', alpha=0.5, color='orange')

    plt.legend()

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(os.path.join(output_dir, f'MDS of Generated Data_{param}={beta}.jpg'), dpi=330, format='jpg')
    plt.cla()

plt.scatter(X_2d[Y_plot == 0, 0], X_2d[Y_plot == 0, 1], label=f'normal data', alpha=0.5, color='black')
plt.scatter(X_2d[Y_plot == -1, 0], X_2d[Y_plot == -1, 1], label=f'anomaly data', alpha=0.5, color='red')
for i, beta in enumerate(beta_list):
    plt.scatter(X_2d[Y_plot == (i + 1), 0], X_2d[Y_plot == (i + 1), 1], label=f'generated data, beta={beta}', alpha=0.5)

plt.legend()

plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig(os.path.join(output_dir, f'MDS of Generated Data_{beta}.jpg'), dpi=330, format='jpg')
plt.cla()