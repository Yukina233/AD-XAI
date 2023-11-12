import numpy as np

path_project = '/home/yukina/Missile_Fault_Detection/project/'
path_file = path_project + 'anomaly_detection/ProblemCheck/'

import gc
import os
import torch
import torchvision.datasets as datasets

device = "cuda" if torch.cuda.is_available() else "cpu"

mnist_testset = datasets.MNIST(root=path_file + 'data', train=False, download=True, transform=None) \
                     .data.reshape(-1, 1, 28, 28) / 255.
fashion_trainset = datasets.FashionMNIST(root=path_file + 'data', train=True, download=True,transform=None)\
                       .data.reshape(-1, 1, 28, 28) / 255.
fashion_testset = datasets.FashionMNIST(root=path_file + 'data', train=False, download=True,transform=None)\
                        .data.reshape(-1, 1, 28, 28) / 255.

from pythae_modified.models import AutoModel

VAE = AutoModel.load_from_folder(
    path_file + 'my_model/VAE_training_2023-11-08_16-47-35/final_model')
print('VAE_fashion-minist loaded')

def calculate_losses(dataset, model, batch_size=10000):
    losses = []
    num_batches = len(dataset) // batch_size

    for i in range(num_batches):
        batch_data = torch.tensor(dataset[i * batch_size:(i + 1) * batch_size], dtype=torch.float)
        input = dict(data=batch_data)
        model_output = model.calculate_elementwise_loss(input)
        losses.append(model_output.recon_loss.cpu().detach().numpy())
        del model_output, batch_data
        gc.collect()

    return losses
def concatenate_losses(losses):
    losses_concatenated = losses[0]
    for loss in losses[1:]:
        losses_concatenated = np.concatenate((losses_concatenated, loss))
    return losses_concatenated

losses_train = calculate_losses(dataset=fashion_trainset, model=VAE, batch_size=5000)
losses_train = concatenate_losses(losses_train)


losses_test = calculate_losses(dataset=fashion_testset, model=VAE, batch_size=5000)
losses_test = concatenate_losses(losses_test)


losses_another = calculate_losses(dataset=mnist_testset, model=VAE, batch_size=5000)
losses_another = concatenate_losses(losses_another)

import matplotlib.pyplot as plt

# 使用matplotlib的hist函数绘制直方图
plt.hist(losses_train, bins=30, edgecolor='black', alpha=0.7, density=True, label='FashionMNIST-TRAIN', color='black')
plt.hist(losses_test, bins=30, edgecolor='black', alpha=0.7, density=True, label='FashionMNIST-TEST', color='lightblue')
plt.hist(losses_another, bins=30, edgecolor='black', alpha=0.7, density=True, label='MNIST-TEST', color='pink')
# 添加标题和标签
plt.title('Loss distribution of VAE trained on FashionMNIST')
plt.xlabel('Loss')
plt.ylabel('Density')
# 添加图例
plt.legend()
# 显示图形
plt.show()

from sklearn.metrics import roc_curve, auc

# 真实标签和预测分数
y_id = np.zeros(losses_train.size + losses_test.size)
y_ood = np.ones(losses_another.size)
y_true = np.concatenate((y_id, y_ood))  # 真实标签，1表示正例，0表示负例
y_scores = np.concatenate((losses_train, losses_test, losses_another)) # 预测分数，对应每个样本是正例的概率

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# 计算AUROC
auroc = auc(fpr, tpr)

# 计算FPR95
# 找到最接近0.95的TPR的索引
index = np.argmin(np.abs(tpr - 0.95))
fpr95 = fpr[index]

print("AUROC: ", auroc)
print("FPR95: ", fpr95)