
#%%
import torch
import torchvision.datasets as datasets

device = "cuda" if torch.cuda.is_available() else "cpu"



mnist_trainset = datasets.MNIST(root='../../data', train=True, download=True, transform=None)

train_dataset = mnist_trainset.data[:-50000].reshape(-1, 1, 28, 28) / 255.
eval_dataset = mnist_trainset.data[-10000:].reshape(-1, 1, 28, 28) / 255.
#%%
from pythae_modified.models import VQVAE, VQVAEConfig
from pythae_modified.trainers import BaseTrainerConfig
from pythae_modified.pipelines.training import TrainingPipeline
from pythae_modified.models.nn.benchmarks.mnist.resnets import Encoder_ResNet_VQVAE_MNIST, Decoder_ResNet_VQVAE_MNIST
#%%
config = BaseTrainerConfig(
    output_dir='my_model',
    learning_rate=1e-3,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_epochs=10, # Change this to train the model a bit more
)


model_config = VQVAEConfig(
    latent_dim=16,
    input_dim=(1, 28, 28),
    commitment_loss_factor=0.25,
    quantization_loss_factor=1.0,
    num_embeddings=128,
    use_ema=True,
    decay=0.99
)

model = VQVAE(
    model_config=model_config,
    encoder=Encoder_ResNet_VQVAE_MNIST(model_config),
    decoder=Decoder_ResNet_VQVAE_MNIST(model_config)
)
#%%
pipeline = TrainingPipeline(
    training_config=config,
    model=model
)
#%%
pipeline(
    train_data=train_dataset,
    eval_data=eval_dataset
)
#%%
import os
from pythae_modified.models import AutoModel
#%%
last_training = sorted(os.listdir('my_model'))[-1]
trained_model = AutoModel.load_from_folder(os.path.join('my_model', last_training, 'final_model')).to(device)
#%% md
## Visualizing reconstructions
#%%
reconstructions = trained_model.reconstruct(eval_dataset[:25].to(device)).detach().cpu()
#%%
import matplotlib.pyplot as plt

# show reconstructions
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(reconstructions[i*5 + j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
#%%
# show the true data
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(eval_dataset[i*5 +j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
#%% md
## Visualizing interpolations
#%%
interpolations = trained_model.interpolate(eval_dataset[:5].to(device), eval_dataset[5:10].to(device), granularity=10).detach().cpu()
#%%
# show interpolations
fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(10, 5))

for i in range(5):
    for j in range(10):
        axes[i][j].imshow(interpolations[i, j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)