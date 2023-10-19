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
path2load = 'my_model/VQVAE_training_2023-10-06_19-25-16/final_model'
trained_model = AutoModel.load_from_folder(
    path_file + path2load)
print('Load model in ' + path2load)
# trained_model = AutoModel.load_from_folder(
#     path_file + 'model_to_check/SVHN_VAE_training_2023-09-20_21-51-15/final_model')
# print('VAE_SVHN loaded')

codebook_embeddings = trained_model.get_codebook()

import matplotlib.pyplot as plt
# show codebook embeddings
# fig, axes = plt.subplots(nrows=6, ncols=10, figsize=(40, 12))
# fig.suptitle('codebook embeddings')
# for i in range(6):
#     for j in range(10):
#         axes[i][j].plot(codebook_embeddings.cpu().detach().numpy()[i * 10 + j])
#         axes[i][j].set_title('index: ' + str(i * 10 + j))
#         # axes[i][j].set_ylim(-15, 15)
# plt.tight_layout()
# plt.show()

# 重构codebook中的编码，由于一张图片完整的编码是(4,4,16)，而codebook中的编码是(1,16)，因此其它位置都设为0
codebook_embeddings_images = torch.zeros(256, 16, 4, 4)
codebook_embeddings_images[:, :, 2:3, 2:3] = codebook_embeddings.unsqueeze(2).unsqueeze(3)
codebook_reconstructions = trained_model.decoder(codebook_embeddings_images).reconstruction



# show codebook reconstructions
fig, axes = plt.subplots(nrows=16, ncols=16, figsize=(40, 40))
fig.suptitle('codebook reconstructions')
for i in range(16):
    for j in range(16):
        axes[i][j].imshow(codebook_reconstructions.cpu().detach().numpy()[i * 16 + j].transpose(1, 2, 0))
        axes[i][j].set_title('index: ' + str(i * 16 + j))
plt.tight_layout()
plt.show()



SVHN_testset = datasets.SVHN(root=path_file + 'data', split='test', download=True, transform=None).data / 255.

# 生成随机index
index = np.random.randint(0, SVHN_testset.shape[0], size=25)
data_to_plot = SVHN_testset[index]

# show the true data
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
for i in range(5):
    for j in range(5):
        axes[i][j].imshow(data_to_plot[i * 5 + j].transpose(1, 2, 0))
        axes[i][j].axis('off')
plt.tight_layout()
plt.show()

reconstructions = trained_model.reconstruct(
    torch.tensor(data_to_plot, dtype=torch.float)).detach().cpu()
# show reconstructions
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(reconstructions[i * 5 + j].cpu().numpy().transpose(1, 2, 0))
        axes[i][j].axis('off')
plt.tight_layout()
plt.show()



cifar10_testset = datasets.CIFAR10(root=path_file + 'data', train=False, download=True, transform=None).data.transpose(0, 3, 1, 2) / 255.

# 生成随机index
index = np.random.randint(0, cifar10_testset.shape[0], size=25)
data_to_plot = cifar10_testset[index]

# show the true data
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))
for i in range(5):
    for j in range(5):
        axes[i][j].imshow(data_to_plot[i * 5 + j].transpose(1, 2, 0))
        axes[i][j].axis('off')
plt.tight_layout()
plt.show()

reconstructions = trained_model.reconstruct(
    torch.tensor(data_to_plot, dtype=torch.float)).detach().cpu()
# show reconstructions
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(reconstructions[i * 5 + j].cpu().numpy().transpose(1, 2, 0))
        axes[i][j].axis('off')
plt.tight_layout()
plt.show()
