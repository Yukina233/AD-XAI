import numpy as np

path_project = '/home/yukina/Missile_Fault_Detection/project/'
path_file = path_project + 'anomaly_detection/ProblemCheck/'
#%%
import torch
import torchvision.datasets as datasets

device = "cuda" if torch.cuda.is_available() else "cpu"


from pythae_modified.models import AutoModel
import matplotlib.pyplot as plt

trained_model = AutoModel.load_from_folder(
    path_file + 'model_to_check/FASHIONMNIST_VAE_training_2023-09-19_12-26-17/final_model')
print('VAE_fashion-minist loaded')

mnist_testset = datasets.MNIST(root=path_file + 'data', train=False, download=True, transform=None) \
                    .data.reshape(-1, 1, 28, 28) / 255.

index = np.random.randint(0, mnist_testset.shape[0], size=25)
data_to_plot = mnist_testset[index]

# show the true data
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(data_to_plot[i*5 +j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout()
plt.show()

reconstructions = trained_model.reconstruct(data_to_plot[:25]).detach().cpu()
# show reconstructions
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(reconstructions[i*5 + j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout()
plt.show()

fashion_testset = datasets.FashionMNIST(root=path_file + 'data', train=False, download=True,transform=None)\
                       .data.reshape(-1, 1, 28, 28) / 255.

index = np.random.randint(0, fashion_testset.shape[0], size=25)
data_to_plot = fashion_testset[index]
# show the true data
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(data_to_plot[i*5 +j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout()
plt.show()

reconstructions = trained_model.reconstruct(data_to_plot[:25]).detach().cpu()
# show reconstructions
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(reconstructions[i*5 + j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout()
plt.show()