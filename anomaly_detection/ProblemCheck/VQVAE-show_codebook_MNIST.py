path_project = '/home/yukina/Missile_Fault_Detection/project/'
path_file = path_project + 'anomaly_detection/ProblemCheck/'

import gc
import os
import torch
import torchvision.datasets as datasets
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

from pythae_modified.models import AutoModel

# last_training = sorted(os.listdir(path_file + 'my_model'))[-1]
# path2load = os.path.join(path_file + 'my_model', last_training, 'final_model')
# trained_model = AutoModel.load_from_folder(path2load)
# print('Load model in ' + path2load)
path2load = 'model_to_check/MNIST_VQVAE_training_2023-09-22_00-25-40/final_model'
trained_model = AutoModel.load_from_folder(
    path_file + path2load)
print('Load model in ' + path2load)
# trained_model = AutoModel.load_from_folder(
#     path_file + 'model_to_check/SVHN_VAE_training_2023-09-20_21-51-15/final_model')
# print('VAE_SVHN loaded')

codebook_embeddings = trained_model.get_codebook()

import matplotlib.pyplot as plt

# 重构codebook中的编码，由于一张图片完整的编码是(4,4,16)，而codebook中的编码是(1,16)，因此其它位置都设为0
codebook_embeddings_images = torch.zeros(256, 16, 4, 4)
codebook_embeddings_images[:, :, 2:3, 2:3] = codebook_embeddings.unsqueeze(2).unsqueeze(3)
codebook_reconstructions = trained_model.decoder(codebook_embeddings_images).reconstruction

# show codebook reconstructions
fig, axes = plt.subplots(nrows=16, ncols=16, figsize=(40, 40))
fig.suptitle('codebook reconstructions')
for i in range(16):
    for j in range(16):
        axes[i][j].imshow(codebook_reconstructions.cpu().detach().numpy()[i * 16 + j].squeeze(0), cmap='gray')
        axes[i][j].set_title('index: ' + str(i * 16 + j))
plt.tight_layout()
plt.show()

FashionMNIST_testset = datasets.FashionMNIST(root=path_file + 'data', train=False, download=True,
                                             transform=None).data.reshape(-1, 1, 28, 28) / 255.

# 生成随机index
index = np.random.randint(0, FashionMNIST_testset.shape[0], size=25)
data_to_plot = FashionMNIST_testset[index]

# show the true data
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
for i in range(5):
    for j in range(5):
        axes[i][j].imshow(data_to_plot[i * 5 + j].squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout()
plt.show()

reconstructions = trained_model.reconstruct(
    data_to_plot).detach().cpu()
# show reconstructions
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(reconstructions[i * 5 + j].cpu().numpy().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout()
plt.show()

MNIST_testset = datasets.MNIST(root=path_file + 'data', train=False, download=True,
                               transform=None).data.reshape(-1, 1, 28, 28) / 255.

# 生成随机index
index = np.random.randint(0, MNIST_testset.shape[0], size=25)
data_to_plot = MNIST_testset[index]

# show the true data
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
for i in range(5):
    for j in range(5):
        axes[i][j].imshow(data_to_plot[i * 5 + j].squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout()
plt.show()

reconstructions = trained_model.reconstruct(
    data_to_plot).detach().cpu()
# show reconstructions
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(reconstructions[i * 5 + j].cpu().numpy().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout()
plt.show()
