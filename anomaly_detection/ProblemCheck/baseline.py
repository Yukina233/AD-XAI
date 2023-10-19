path_project = '/home/yukina/Missile_Fault_Detection/project/'
path_file = path_project + 'anomaly_detection/ProblemCheck/'
# %%
import torch
import torchvision.datasets as datasets

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# trainset = datasets.MNIST(root=path_file + 'data', train=True, download=True, transform=None)
# testset = datasets.MNIST(root=path_file + 'data', train=False, download=True, transform=None)
trainset = datasets.FashionMNIST(root=path_file + 'data', train=True, download=True, transform=None)
testset = datasets.FashionMNIST(root=path_file + 'data', train=False, download=True, transform=None)

train_dataset = trainset.data.reshape(-1, 1, 28, 28) / 255.
eval_dataset = testset.data.reshape(-1, 1, 28, 28) / 255.
# %%
from pythae_modified.models import VAE, VAEConfig
from pythae_modified.trainers import BaseTrainerConfig
from pythae_modified.pipelines.training import TrainingPipeline
from pythae_modified.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST, Decoder_ResNet_AE_MNIST, \
    Encoder_CNN_VAE_MNIST_baseline, Decoder_CNN_VAE_MNIST_baseline

# %%
config = BaseTrainerConfig(
    output_dir=path_file + 'my_model',
    learning_rate=1e-4,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_epochs=200,  # Change this to train the model a bit more
    optimizer_cls="RMSprop",
)

model_config = VAEConfig(
    input_dim=(1, 28, 28),
    latent_dim=50,
    reconstruction_loss="bce"
)

model = VAE(
    model_config=model_config,
    encoder=Encoder_CNN_VAE_MNIST_baseline(model_config),
    decoder=Decoder_CNN_VAE_MNIST_baseline(model_config)
)
# %%
pipeline = TrainingPipeline(
    training_config=config,
    model=model
)
# %%
pipeline(
    train_data=train_dataset,
    eval_data=eval_dataset
)
# %%
import os
from pythae_modified.models import AutoModel

# %%
last_training = sorted(os.listdir(path_file + 'my_model'))[-1]
trained_model = AutoModel.load_from_folder(os.path.join(path_file + 'my_model', last_training, 'final_model'))
# %%
from pythae_modified.samplers import NormalSampler

# %%
# create normal sampler
normal_samper = NormalSampler(
    model=trained_model
)
# %%
# sample
gen_data = normal_samper.sample(
    num_samples=25
).argmax(dim=1)/255.
# %%
import matplotlib.pyplot as plt

# %%
# show results with normal sampler
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(gen_data[i * 5 + j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
plt.show()
# %%
from pythae_modified.samplers import GaussianMixtureSampler, GaussianMixtureSamplerConfig

# %%
# set up GMM sampler config
gmm_sampler_config = GaussianMixtureSamplerConfig(
    n_components=10
)

# create gmm sampler
gmm_sampler = GaussianMixtureSampler(
    sampler_config=gmm_sampler_config,
    model=trained_model
)

# fit the sampler
gmm_sampler.fit(train_dataset)
# %%
# sample
gen_data = gmm_sampler.sample(
    num_samples=25
).argmax(dim=1)/255.
# %%
# show results with gmm sampler
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(gen_data[i * 5 + j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
plt.show()
# %% md
## ... the other samplers work the same
# %% md
## Visualizing reconstructions
# %%
reconstructions = trained_model.reconstruct(eval_dataset[:25].to(device)).detach().cpu().argmax(dim=1)/255.
# %%
# show reconstructions
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(reconstructions[i * 5 + j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
plt.show()
# %%
# show the true data
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(eval_dataset[i * 5 + j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
plt.show()
# %% md
## Visualizing interpolations
# %%
interpolations = trained_model.interpolate(eval_dataset[:5].to(device), eval_dataset[5:10].to(device),
                                           granularity=10).detach().cpu().argmax(dim=1)/255.
# %%
# show interpolations
fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(10, 5))

for i in range(5):
    for j in range(10):
        axes[i][j].imshow(interpolations[i, j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
plt.show()
