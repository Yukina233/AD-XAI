import pythae_modified.trainers.training_callbacks

path_project = '/home/yukina/Missile_Fault_Detection/project/'
path_file = path_project + 'anomaly_detection/ProblemCheck/'
# %%
import torchvision.datasets as datasets
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# trainset = datasets.SVHN(root=path_file + 'data', split='train', download=True, transform=None).data
# testset = datasets.SVHN(root=path_file + 'data', split='test', download=True, transform=None).data

trainset = datasets.CIFAR10(root=path_file + 'data', train=True, download=True, transform=None).data.transpose(0, 3, 1, 2)
testset = datasets.CIFAR10(root=path_file + 'data', train=False, download=True, transform=None).data.transpose(0, 3, 1, 2)

train_dataset = trainset / 255.
eval_dataset = testset / 255.

import matplotlib.pyplot as plt
# %%
from pythae_modified.models.vq_vae.vq_vae_config import Simplified_VQVAEConfig
from pythae_modified.models.vq_vae.vq_vae_model import Simplified_VQVAE
from pythae_modified.models import VQVAE, VQVAEConfig
from pythae_modified.trainers import BaseTrainerConfig
from pythae_modified.pipelines.training import TrainingPipeline
from pythae_modified.models.nn.benchmarks.cifar import Encoder_ResNet_VQVAE_CIFAR, Decoder_ResNet_VQVAE_CIFAR

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# %%


config = BaseTrainerConfig(
    output_dir=path_file + 'my_model',
    learning_rate=1e-3,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_epochs=1, # Change this to train the model a bit more
)

model_config = VQVAEConfig(
    input_dim=(3, 32, 32),
    latent_dim=16,
    reconstruction_loss="mse",
    commitment_loss_factor=0.25,
    quantization_loss_factor=1.0,
    num_embeddings=256,
    use_ema=True,
    decay=0.99,
    epsilon_prob=0,
    density_factor=0.25
)

model = VQVAE(
    model_config=model_config,
    encoder=Encoder_ResNet_VQVAE_CIFAR(model_config),
    decoder=Decoder_ResNet_VQVAE_CIFAR(model_config)
)
# %%
pipeline = TrainingPipeline(
    training_config=config,
    model=model
)
# %%
pipeline(
    train_data=train_dataset,
    eval_data=eval_dataset,
    callbacks=[pythae_modified.trainers.training_callbacks.WandbCallback]
)
# %%
import os
from pythae_modified.models import AutoModel

# %%
last_training = sorted(os.listdir(path_file + 'my_model'))[-1]
trained_model = AutoModel.load_from_folder(os.path.join(path_file + 'my_model', last_training, 'final_model'))

## Visualizing reconstructions
# %%
reconstructions = trained_model.reconstruct(
    torch.tensor(eval_dataset[:25], dtype=torch.float)).detach().cpu()
# %%
# show reconstructions
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(reconstructions[i * 5 + j].cpu().numpy().transpose(1, 2, 0))
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
plt.show()
# %%
# show the true data
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(eval_dataset[i * 5 + j].transpose(1, 2, 0))
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
plt.show()
# %% md
## Visualizing interpolations
# %%
interpolations = trained_model.interpolate(torch.tensor(eval_dataset[:5], dtype=torch.float),
                                           torch.tensor(eval_dataset[5:10], dtype=torch.float),
                                           granularity=10).detach().cpu()
# %%
# show interpolations
fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(10, 5))

for i in range(5):
    for j in range(10):
        axes[i][j].imshow(interpolations[i, j].cpu().numpy().transpose(1, 2, 0))
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
plt.show()
