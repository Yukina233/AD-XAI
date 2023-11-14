import json

import matplotlib.colors
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from torchvision.datasets import CIFAR10, MNIST
import matplotlib.pyplot as plt

from adbench_modified.baseline.DeepSAD.src.utils import plot_images_grid

# 假设您的JSON数据保存在名为'data.json'的文件中
project_path = '/home/yukina/Missile_Fault_Detection/project/Deep-SAD-OriginalPaper/log/'
file_path = project_path + 'cifar10,group/dataset=cifar10,normal=3,outlier=5,ratioNormal=0.01,ratioOutlier=0.01,seed=avg/'  # 文件路径

# 从文件中读取JSON数据
with open(file_path + 'config.json', 'r') as file:
    config = json.load(file)

with open(file_path + 'results.json', 'r') as file:
    results = json.load(file)

# 提取test_scores
test_scores = results['test_scores']

# 分离出真实标签和预测得分
indices = [score[0] for score in test_scores]  # 样本索引
y_true = [score[1] for score in test_scores]  # 真实标签
y_scores = [score[2] for score in test_scores]  # 预测得分
indices, y_true, y_scores = np.array(indices), np.array(y_true), np.array(y_scores)

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(y_true, y_scores)
pr_auc = auc(recall, precision)

# 计算FPR95
# 找到最接近0.95的TPR的索引
tpr_target = 0.95
closest_index = np.argmin(np.abs(tpr - tpr_target))
threshold_tpr95 = thresholds[closest_index]
fpr95 = fpr[closest_index]

print("AUROC: ", roc_auc)
print("AUPR: ", pr_auc)
print("FPR95: ", fpr95)

# 画出ROC曲线
plt.cla()
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(file_path + '/ROC curve')

# 画出PR曲线
plt.figure()
plt.plot(recall, precision, label='PR curve (area = %0.2f)' % pr_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower right")
plt.savefig(file_path + '/PR curve')

# 画出正常和异常数据得分的分布图
plt.cla()
plt.figure()
plt.hist([score[2] for score in test_scores if score[1] == 1], bins=100, alpha=0.5, label='Anomalies',
         color=matplotlib.colors.TABLEAU_COLORS['tab:orange'])
plt.hist([score[2] for score in test_scores if score[1] == 0], bins=100, alpha=0.5, label='Normal',
         color=matplotlib.colors.TABLEAU_COLORS['tab:blue'])
plt.axvline(threshold_tpr95, color='r', linestyle='dashed', linewidth=1, label='Threshold tpr=95')
plt.title('Distribution of Anomaly Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.savefig(file_path + '/Distribution of Anomaly Scores')

# 输出阈值附近的样本标号
print("Threshold for TPR=95:", threshold_tpr95)
# print("Sample indices above the threshold:")


normal_indices = indices[y_scores <= threshold_tpr95]
ordered_normal_indices = normal_indices[np.argsort(y_scores[normal_indices])]
anomaly_indices = indices[y_scores > threshold_tpr95]
ordered_anomaly_indices = anomaly_indices[np.argsort(y_scores[anomaly_indices])]

false_normal_indices = ordered_normal_indices[np.where(y_true[ordered_normal_indices] == 1)]
false_anomaly_indices = ordered_anomaly_indices[np.where(y_true[ordered_anomaly_indices] == 0)]

transform = transforms.ToTensor()
target_transform = transforms.Lambda(lambda x: int(x != config['normal_class']))
if config['dataset_name'] == 'cifar10':
    test_set = CIFAR10(root=config['data_path'], train=False, transform=transform, target_transform=target_transform,
                       download=True)
    false_normal = torch.tensor(
        np.transpose(test_set.data[false_normal_indices[-32:], ...], (0, 3, 1, 2)))
    false_anomaly = torch.tensor(
        np.transpose(test_set.data[false_anomaly_indices[:32], ...], (0, 3, 1, 2)))

if config['dataset_name'] == 'mnist':
    test_set = MNIST(root=config['data_path'], train=False, transform=transform, target_transform=target_transform,
                     download=True)
    false_normal = torch.tensor(test_set.data[false_normal_indices[-32:], ...].unsqueeze(1))
    false_anomaly = torch.tensor(test_set.data[false_anomaly_indices[:32], ...].unsqueeze(1))

# 输出y_true = 1的正常样本
plt.clf()
plot_images_grid(false_normal, export_img=file_path + '/False Normal', padding=2)

# 输出y_true = 0的异常样本
plt.clf()
plot_images_grid(false_anomaly, export_img=file_path + '/False Anomaly', padding=2)
