import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from tqdm import tqdm

from adbench_modified.baseline.DeepSAD.src.run import DeepSAD

from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


path_project = '/home/yukina/Missile_Fault_Detection/project'

iteration = 4
test_set_name = 'banwuli_data'
model_name = 'no_tau2_K=7,deepsad_epoch=20,gan_epoch=50,lam1=1,lam2=100,tau1=0.1'
path_plot = os.path.join(path_project, f'adversarial_ensemble_AD/log/{test_set_name}/train_result', model_name, 'score')
os.makedirs(path_plot, exist_ok=True)

path_train_origin = os.path.join(path_project, f'data/banwuli_data/yukina_data/DeepSAD_data/ks/dataset_1.npz')

path_train_new = os.path.join(path_project, f'data/{test_set_name}/yukina_data/train_seperate', 'augment', model_name, f'{iteration}')

path_detector = os.path.join(path_project, f'adversarial_ensemble_AD/models/DeepSAD_seed=1.pth')
path_detector_aug = os.path.join(path_project, f'adversarial_ensemble_AD/models/banwuli_data/DeepSAD_aug/DeepSAD_seed=1.pth')

random_seed = 42

# 随机抽取的样本数
num_samples = 2000
np.random.seed(0)

train_origin = np.load(path_train_origin)

train_origin = train_origin['X_train'][np.where(train_origin['y_train'] == 0)]
sampled_train_origin = train_origin[np.random.choice(range(0, train_origin.shape[0]), num_samples, replace=True)]

datasets = []
for dataset in os.listdir(path_train_new):
    datasets.append(np.load(os.path.join(path_train_new, dataset)))

generated_data = datasets[0]['X_train'][np.where(datasets[0]['y_train'] == 1)]

test_data_path = os.path.join(path_project, f'data/{test_set_name}/yukina_data/DeepSAD_data')

# plot时是否考虑聚类标签
use_train_cluster_label = False


faults = []
fault_data = []
test_normal_data = []
for fault in tqdm(os.listdir(test_data_path)):
    base_name = os.path.basename(fault).replace('.npz', '')
    faults.append(base_name)
    # 创建结果文件夹路径

    normal_data = []
    anomaly_data = []
    path_fault = os.path.join(test_data_path, fault)
    files = os.listdir(path_fault)
    for i in range(1, int(len(files) + 1)):
        data = np.load(os.path.join(path_fault, f"dataset_{i}.npz"))
        dataset = {'X': data['X'], 'y': data['y'], 'X_train': data['X_train'], 'y_train': data['y_train'],
                   'X_test': data['X_test'], 'y_test': data['y_test']}
        normal_data.append(dataset['X_test'][np.where(dataset['y_test'] == 0)])
        anomaly_data.append(dataset['X_test'][np.where(dataset['y_test'] == 1)])

    normal_data = np.concatenate(normal_data)
    test_normal_data.append(normal_data)

    anomaly_data = np.concatenate(anomaly_data)
    fault_data.append(anomaly_data)

test_normal_data = np.concatenate(test_normal_data)
fault_data = np.concatenate(fault_data)
sampled_test_normal = test_normal_data[np.random.choice(range(0, test_normal_data.shape[0]), num_samples, replace=True)]
sampled_anomaly = fault_data[np.random.choice(range(0, fault_data.shape[0]), num_samples, replace=True)]

# normal_data = np.concatenate(normal_data)
# sampled_normal = normal_data[np.random.choice(range(0, normal_data.shape[0]), num_samples, replace=False)]

sampled_train_data = []
init_train_data = []
for dataset in datasets:
    train_data = dataset['X_train'][np.where(dataset['y_train'] == 0)]
    init_train_data.append(train_data)
    sampled_train_data.append(train_data[np.random.choice(range(0, train_data.shape[0]), num_samples, replace=True)])

init_train_data = np.concatenate(init_train_data)
sampled_init_train_data = init_train_data[np.random.choice(range(0, init_train_data.shape[0]), num_samples, replace=True)]

# X_plot = np.concatenate((sampled_normal, np.concatenate(fault_data)))
# y_plot = np.concatenate((np.zeros(num_samples), np.concatenate([np.ones(num_samples) * (id + 1) for id, fault in enumerate(fault_data)])))



if use_train_cluster_label:
    # 正常数据的标签大于0，异常数据的标签小于0，生成数据的标签为0
    X_origin = np.concatenate((np.concatenate(sampled_train_data), np.concatenate(fault_data)))
    y_origin = np.concatenate((np.concatenate([np.ones(num_samples) * (id + 1) for id, fault in enumerate(sampled_train_data)]), np.concatenate([np.ones(num_samples) * -(id + 1) for id, fault in enumerate(fault_data)])))
else:
    # 不区分训练数据的聚类标签
    # X_origin = np.concatenate((sampled_init_train_data, np.concatenate(fault_data)))
    X_origin = np.concatenate((sampled_train_origin, sampled_anomaly))
    y_origin = np.concatenate((np.ones(num_samples), np.ones(num_samples) * -1))

sampled_generated = generated_data[np.random.choice(range(0, generated_data.shape[0]), num_samples, replace=True)]

X_train = X_origin
y_train = y_origin

# X_all = np.concatenate((X_origin, sampled_generated))
# y_all = np.concatenate((y_origin, np.zeros(sampled_generated.shape[0])))


detector1 = DeepSAD(seed=1, load_model=os.path.join(path_detector))
detector1.load_model_from_file()
score1, rep_train1 = detector1.predict_score(X_train)

detector2 = DeepSAD(seed=1, load_model=os.path.join(path_detector_aug))
detector2.load_model_from_file()
score2, rep_train2 = detector2.predict_score(X_train)

# data_plot = {
#     '训练样本类型': ['使用增强数据前'] * num_samples + ['使用增强数据后'] * num_samples,
#     '状态': ['正常数据'] * num_samples + ['异常数据'] * num_samples,
#     '值': np.concatenate([score1, score2])
# }

# 将分数和标签分配给各自的场景

scores = [score1[y_train<0], score2[y_train<0]]
labels = ['使用对抗样本前', '使用对抗样本后']

sns.set(style="whitegrid")  # 设置 seaborn 的样式
plt.figure(figsize=(8, 6))  # 设置图像的大小
plt.rcParams['font.sans-serif'] = ['SimSun']
ax = sns.boxplot(data=scores, width=0.5, palette="vlag")  # 绘制箱型图

# 突出显示中位数
colors = ['red', 'green']  # 为中位数和均值设置颜色
for i, patch in enumerate(ax.artists):

    # 计算并绘制均值
    mean_value = np.mean(scores[i])
    # 添加一条表示均值的水平线
    plt.axhline(y=mean_value, color=colors[1], linestyle='--', xmin=0.125 + i*0.5, xmax=0.375 + i*0.5, label='Mean' if i == 0 else "")

plt.xticks(np.arange(len(labels)), labels)  # 设置 x 轴的标签
plt.ylabel('异常得分')  # 设置 y 轴的标签
plt.title('使用对抗样本训练前后的故障样本的异常得分的分布对比')  # 设置图表的标题
plt.show()  # 显示图表

# sns.boxplot(data=scores, width=0.5, palette="vlag")  # 绘制箱型图
# plt.xticks(np.arange(len(labels)), labels)  # 设置 x 轴的标签
# plt.ylabel('Anomaly Score')  # 设置 y 轴的标签
# plt.title('Comparison of Anomaly Scores Before and After Data Augmentation')  # 设置图表的标题
# plt.show()  # 显示图表
plt.savefig(os.path.join(path_plot, f'DeepSAD.png'))  # 保存图表

print('normal mean score1:', np.mean(score1[y_train > 0]))
print('normal mean score2:', np.mean(score2[y_train > 0]))

print('normal std score1:', np.std(score1[y_train > 0]))
print('normal std score2:', np.std(score2[y_train > 0]))

print('normal median score1:', np.median(score1[y_train > 0]))
print('normal median score2:', np.median(score2[y_train > 0]))

print('fault mean score1:', np.mean(score1[y_train < 0]))
print('fault mean score2:', np.mean(score2[y_train < 0]))

print('fault std score1:', np.std(score1[y_train < 0]))
print('fault std score2:', np.std(score2[y_train < 0]))

print('fault median score1:', np.median(score1[y_train < 0]))
print('fault median score2:', np.median(score2[y_train < 0]))

# plt.hist(score1[y_train > 0], bins=10, alpha=0.5, label='normal', density=False)
# plt.hist(score1[y_train < 0], bins=10, alpha=0.5, label='fault', density=False)
# # plt.xlim([0,5000])
# plt.legend()
# plt.title("Scores Histogram")
# plt.xlabel("Scores")
# plt.ylabel("Frequency")
# plt.show()
# plt.savefig(os.path.join(path_plot, f'DeepSAD.png'))
# plt.close()


# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
#
# # 在第一个子图上绘制正常数据的直方图
# ax1.hist(score1[y_train > 0], bins=30, color='blue', alpha=0.7)
# ax1.set_title('Normal Scores')
# ax1.set_xlabel('Score')
# ax1.set_ylabel('Frequency')
#
# # 在第二个子图上绘制异常数据的直方图
# ax2.hist(score1[y_train < 0], bins=30, color='red', alpha=0.7)
# ax2.set_title('Anomaly Scores')
# ax2.set_xlabel('Score')
#
# plt.suptitle('Histograms of Normal and Anomaly Scores with Different X Scales')
# plt.tight_layout()
# plt.show()
# plt.savefig(os.path.join(path_plot, f'DeepSAD.png'))

# fig, ax1 = plt.subplots()
#
# residual_normal = score2[y_train > 0] - score1[y_train > 0]
# residual_fault = score2[y_train < 0] - score1[y_train < 0]
# # 正常数据直方图
# score_plot1 = score1[y_train < 0][score1[y_train < 0] < 300]
# ax1.hist(score_plot1, bins=100, color='tab:blue', alpha=0.7)
# # ax1.set_xlim(0, 2.5)  # 根据正常分数的分布调整
# # ax1.set_ylim(0, 1600)
# # ax1.set_xlim(0, 800)  # 根据异常分数的分布调整
# ax1.set_xlabel('normal Scores', color='tab:blue')
# ax1.set_ylabel('Frequency')
# ax1.tick_params(axis='x', labelcolor='tab:blue')
#
# # 创建具有不同x轴尺度的第二个x轴
# ax2 = ax1.twiny()
# score_plot2 = score2[y_train < 0][score2[y_train < 0] < 300]
# ax2.hist(score_plot2, bins=100, color='tab:red', alpha=0.7)
# # ax2.set_xlim(0, 800)  # 根据异常分数的分布调整
# # ax2.set_ylim(0, 1600)
# ax2.set_xlabel('Anomaly Scores', color='tab:red')
# ax2.tick_params(axis='x', labelcolor='tab:red')
#
# plt.title('Double X-Axis Histogram')
# plt.show()
# plt.savefig(os.path.join(path_plot, f'DeepSAD.png'))

# plt.cla()
#
# plt.hist(score2[y_train > 0], bins=10, alpha=0.5, label='normal', density=False)
# plt.hist(score2[y_train < 0], bins=10, alpha=0.5, label='fault', density=False)
# # plt.xlim([0,5000])
# plt.legend()
# plt.title("Scores Histogram")
# plt.xlabel("Scores")
# plt.ylabel("Frequency")
# plt.show()
# plt.savefig(os.path.join(path_plot, f'DeepSAD_with_aug.png'))
# plt.close()

# plt.cla()
# fig, ax1 = plt.subplots()
#
# # 正常数据直方图
# ax1.hist(score2[y_train > 0], bins=30, color='tab:blue', alpha=0.7)
# ax1.set_xlim(0, 2.5)  # 根据正常分数的分布调整
# ax1.set_ylim(0, 1600)
# ax1.set_xlabel('normal Scores', color='tab:blue')
# ax1.set_ylabel('Frequency')
# ax1.tick_params(axis='x', labelcolor='tab:blue')
#
# # 创建具有不同x轴尺度的第二个x轴
# ax2 = ax1.twiny()
# ax2.hist(score2[y_train < 0], bins=30, color='tab:red', alpha=0.7)
# ax2.set_xlim(0, 3000)  # 根据异常分数的分布调整
# ax2.set_ylim(0, 1600)
# ax2.set_xlabel('Anomaly Scores', color='tab:red')
# ax2.tick_params(axis='x', labelcolor='tab:red')
#
# plt.title('Double X-Axis Histogram')
# plt.show()
# plt.savefig(os.path.join(path_plot, f'DeepSAD_with_aug.png'))