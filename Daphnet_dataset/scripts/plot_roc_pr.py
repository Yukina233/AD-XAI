import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

path_project = '/home/yukina/Missile_Fault_Detection/project'

model_name0 = 'AE'
score0 = np.load(os.path.join(path_project, 'Daphnet_dataset/autoencoder/results/window=100, step=10/scores.npy'))

model_name1 = 'DeepSAD'
score1 = np.load(os.path.join(path_project, 'Daphnet_dataset/log/Daphnet/train_result/fix_pretrain, net=Dense, std, window=100, step=10, n_epochs=50, ae_n_epochs=50, lr=0.001, ae_lr=0.001/scores/test.npz/scores_DeepSAD.npy'))

model_name2 = 'ensemble DeepSAD'
score2 = np.load(os.path.join(path_project, 'Daphnet_dataset/log/Daphnet/train_result/no_GAN, std, window=100, step=10, no_tau2_K=7,deepsad_epoch=50,gan_epoch=20,lam1=0.9,lam2=0.1,tau1=1/scores/0/scores_DeepSAD.npy'))

model_name3 = 'ensemble DeepSAD with aug'
score3 = np.load(os.path.join(path_project, 'Daphnet_dataset/log/Daphnet/train_result/no_GAN, std, window=100, step=10, no_tau2_K=7,deepsad_epoch=50,gan_epoch=20,lam1=0.9,lam2=0.1,tau1=1/scores/4/scores_DeepSAD.npy'))

# 假设你有以下数据
# 各分类器的异常分数
scores = {
    model_name0: score0,
    model_name1: score1,
    model_name2: score2,
    model_name3: score3
}

# 对应的标签
labels = np.load(os.path.join(path_project, 'Daphnet_dataset/log/Daphnet/train_result/fix_pretrain, net=Dense, std, window=100, step=10, n_epochs=50, ae_n_epochs=50, lr=0.001, ae_lr=0.001/scores/test.npz/labels_DeepSAD.npy'))

# 绘制ROC曲线
plt.figure()

for name, score in scores.items():
    # 计算ROC曲线和AUC值
    fpr, tpr, _ = roc_curve(labels, score)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

# 绘制对角线（代表随机分类器）
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# 设置图形信息
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

# 显示图形
plt.show()
plt.close()

# 绘制PR曲线
plt.figure()

for name, score in scores.items():
    # 计算PR曲线和平均精度
    precision, recall, _ = precision_recall_curve(labels, score)
    average_precision = average_precision_score(labels, score)

    # 绘制PR曲线
    plt.plot(recall, precision, lw=2, label=f'{name} (AP = {average_precision:.2f})')

# 设置图形信息
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')

# 显示图形
plt.show()