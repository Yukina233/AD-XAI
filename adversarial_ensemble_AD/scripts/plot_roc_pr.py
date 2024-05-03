import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
path_project = '/home/yukina/Missile_Fault_Detection/project'
# 假设y_true是真实的标签，y_score是模型预测的分数
# y_true = ...
# y_score = ...

label_SSLLE_AD = np.load(os.path.join(path_project, "data/scores/labels.npy"), allow_pickle=True)
scores_SSLLE_AD = np.load(os.path.join(path_project, "data/scores/scores_SSLLE-AD.npy"), allow_pickle=True)

faults = ['sf']
scores_DeepSAD = []
label_DeepSAD = []
for fault in faults:
    scores_list = []
    label_list = []
    path_fault_dir = os.path.join(path_project, 'adversarial_ensemble_AD/log/real_data/DeepSAD/DeepSAD,n_epoch=50', fault)
    for trajectory_dir in os.listdir(path_fault_dir):
        scores = pd.read_csv(os.path.join(path_fault_dir, trajectory_dir, 'results.csv'))['scores']
        scores_list.append(np.array(scores))

        labels = pd.read_csv(os.path.join(path_fault_dir, trajectory_dir, 'results.csv'))['labels']
        label_list.append(np.array(labels))
    scores_DeepSAD.append(scores_list)
    label_DeepSAD.append(label_list)

label_DeepSAD_Ensemble = np.load(os.path.join(path_project, "data/scores/labels_DeepSAD_Ensemble.npy"), allow_pickle=True)
scores_DeepSAD_Ensemble = np.load(os.path.join(path_project, "data/scores/scores_DeepSAD_Ensemble.npy"), allow_pickle=True)

y_true1 = np.concatenate(np.concatenate(label_SSLLE_AD.tolist()).tolist())
y_true2 = np.concatenate(np.concatenate(label_DeepSAD).tolist())
y_true3 = np.concatenate(np.concatenate(label_DeepSAD_Ensemble.tolist()).tolist())

y_scores_model1 = np.concatenate(np.concatenate(scores_SSLLE_AD.tolist()).tolist())
y_scores_model2 = np.concatenate(np.concatenate(scores_DeepSAD).tolist())
y_scores_model3 = np.concatenate(np.concatenate(scores_DeepSAD_Ensemble.tolist()).tolist())

# 计算ROC曲线的各个点
fpr1, tpr1, _ = roc_curve(y_true1, y_scores_model1)
fpr2, tpr2, _ = roc_curve(y_true2, y_scores_model2)
fpr3, tpr3, _ = roc_curve(y_true3, y_scores_model3)
# roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.figure()
lw = 2
# plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot(fpr1, tpr1, lw=lw, label='SSLLE-AD')
plt.plot(fpr2, tpr2, lw=lw, label='DeepSAD')
plt.plot(fpr3, tpr3, lw=lw, label='DeepSAD-Ensemble')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳率')
plt.ylabel('真阳率')
plt.title('各故障检测方法在所有测试数据上的ROC曲线对比图')
plt.legend(loc="lower right")
plt.show()

# 计算PR曲线的各个点
precision1, recall1, _ = precision_recall_curve(y_true1, y_scores_model1)
precision2, recall2, _ = precision_recall_curve(y_true2, y_scores_model2)
precision3, recall3, _ = precision_recall_curve(y_true3, y_scores_model3)

precision_threshold = 0.99
recall1_at_threshold = recall1[np.where(precision1 >= precision_threshold)[0][0]]
recall2_at_threshold = recall2[np.where(precision2 >= precision_threshold)[0][0]]
recall3_at_threshold = recall3[np.where(precision3 >= precision_threshold)[0][0]]

recall_threshold = 0.99
precision1_at_threshold = precision1[np.where(recall1 >= recall_threshold)[0][-1]]
precision2_at_threshold = precision1[np.where(recall2 >= recall_threshold)[0][-1]]
precision3_at_threshold = precision1[np.where(recall3 >= recall_threshold)[0][-1]]

# pr_auc = auc(recall, precision)
# 绘制PR曲线
plt.figure()
# plt.plot(recall, precision, color='darkorange', lw=lw, label='PR curve (area = %0.3f)' % pr_auc)
plt.plot(recall1, precision1, lw=lw, label='SSLLE-AD')
plt.plot(recall2, precision2, lw=lw, label='DeepSAD')
plt.plot(recall3, precision3, lw=lw, label='DeepSAD-Ensemble')
plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('召回率')
plt.ylabel('精确率')
plt.title('各故障检测方法在所有测试数据上的PR曲线对比图')
plt.legend(loc="lower left")
plt.show()

print('recall1_at_threshold:', recall1_at_threshold)
print('recall2_at_threshold:', recall2_at_threshold)
print('recall3_at_threshold:', recall3_at_threshold)

print('precision1_at_threshold:', precision1_at_threshold)
print('precision2_at_threshold:', precision2_at_threshold)
print('precision3_at_threshold:', precision3_at_threshold)