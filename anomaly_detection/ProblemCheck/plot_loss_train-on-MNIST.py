path_project = '/home/yukina/Missile_Fault_Detection/project/'
path_file = path_project + 'anomaly_detection/ProblemCheck/'

import gc
import os
import torch
import torchvision.datasets as datasets

device = "cuda" if torch.cuda.is_available() else "cpu"

mnist_trainset = datasets.MNIST(root=path_file + 'data', train=True, download=True, transform=None) \
                     .data.reshape(-1, 1, 28, 28) / 255.
mnist_testset = datasets.MNIST(root=path_file + 'data', train=False, download=True, transform=None) \
                    .data.reshape(-1, 1, 28, 28) / 255.
fashion_testset = datasets.FashionMNIST(root=path_file + 'data', train=False, download=True,transform=None)\
                       .data.reshape(-1, 1, 28, 28) / 255.

from pythae_modified.models import AutoModel

VAE_mnist = AutoModel.load_from_folder(
    path_file + 'model_to_check/MNIST_VQVAE_training_2023-09-22_00-25-40/final_model')
print('VAE_minist loaded')



mnist_mean = mnist_trainset.mean(dim=0).flatten()
mnist_var = mnist_trainset.var(dim=0).flatten()
fashion_mean = fashion_testset.mean(dim=0).flatten()
fashion_var = fashion_testset.var(dim=0).flatten()

import matplotlib.pyplot as plt
# 绘制数据分布直方图
import seaborn as sns
sns.kdeplot(mnist_mean, label='MNIST', color='black')
sns.kdeplot(fashion_mean, label='FASHIONMNIST', color='lightblue')
plt.legend()
plt.xlabel('mean')
plt.ylabel('frequency')
plt.show()

sns.kdeplot(mnist_var, label='MNIST', color='black')
sns.kdeplot(fashion_var, label='FASHIONMNIST', color='lightblue')
plt.legend()
plt.xlabel('var')
plt.ylabel('frequency')
plt.show()

del mnist_mean, mnist_var, fashion_mean, fashion_var
gc.collect()

input = dict(data=(mnist_trainset))
model_output = VAE_mnist.calculate_elementwise_loss(input)
losses_train = model_output.loss.cpu().detach().numpy()
del model_output
gc.collect()

input = dict(data=(mnist_testset))
model_output = VAE_mnist.calculate_elementwise_loss(input)
losses_test = model_output.loss.cpu().detach().numpy()
del model_output
gc.collect()

data_fashion = dict(data=(fashion_testset))
model_output = VAE_mnist.calculate_elementwise_loss(data_fashion)
losses_another = model_output.loss.cpu().detach().numpy()
del model_output
gc.collect()


# 使用matplotlib的hist函数绘制直方图
plt.hist(losses_train, bins=30, edgecolor='black', alpha=0.7, density=True, label='MNIST-TRAIN', color='black')
plt.hist(losses_test, bins=30, edgecolor='black', alpha=0.7, density=True, label='MNIST-TEST', color='lightblue')
plt.hist(losses_another, bins=30, edgecolor='black', alpha=0.7, density=True, label='FashionMNIST-TEST', color='pink')
# 添加标题和标签
plt.title('Loss distribution of VQVAE trained on MNIST')
plt.xlabel('Loss')
plt.ylabel('Density')
# 添加图例
plt.legend()
# 显示图形
plt.show()
