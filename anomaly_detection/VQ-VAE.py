import torch
import torch.nn as nn
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score

import pythae_modified
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from pythae_modified.models.nn import BaseEncoder, BaseDecoder
from pythae_modified.models.base.base_utils import ModelOutput
from pythae_modified.models.vq_vae.vq_vae_config import Simplified_VQVAEConfig
from pythae_modified.models.vq_vae.vq_vae_model import Simplified_VQVAE


def get_dat(path):
    a = pd.read_csv(path,delim_whitespace=True)
    b = a[['dOmega_ib_b[0]', 'dOmega_ib_b[1]', 'dOmega_ib_b[2]', 'Gama', 'Theta_k', 'Psi_t', 'fb[0]', 'fb[1]',
           'fb[2]', 'Alfa', 'Beta', 'zmb', 'P']]
    b = np.array(b, dtype=float)
    return b

class Encoder_Linear_VQVAE_Missile(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)

        self.input_dim = 13
        self.hidden_dim = 128
        self.latent_dim = args.latent_dim

        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim)
        )

    def forward(self, x: torch.Tensor):
        h = self.layers(x)
        output = ModelOutput(
            embedding=h,
        )
        return output

class Decoder_Linear_VQVAE_Missile(BaseDecoder):
    def __init__(self, args):
        BaseDecoder.__init__(self)

        self.input_dim = 13
        self.hidden_dim = 128
        self.latent_dim = args.latent_dim

        self.layers = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor):
        output = ModelOutput(
            reconstruction=self.layers(z)
        )
        return output

path_project = '/home/yukina/Missile_Fault_Detection/project/'
device = "cuda" if torch.cuda.is_available() else "cpu"

# load data
# mnist_trainset = datasets.MNIST(root='../../data', train=True, download=True, transform=None)

path1 = path_project + 'anomaly_detection/data/zc1.dat'
normal_data = get_dat(path1)
path2 = path_project + 'raw/ks/1/ks1_-1.dat'
ka_data = get_dat(path2)
path3 = path_project + 'raw/normal/zc3.dat'
new_data = get_dat(path3)
scaler = MinMaxScaler().fit(normal_data)

window_size = 1
train_dataset = scaler.transform(normal_data[:5000])
val_dataset = scaler.transform(ka_data[5000:])
test_dataset = scaler.transform(ka_data)
normal_dataset = scaler.transform(normal_data)

from pythae_modified.trainers import BaseTrainerConfig
from pythae_modified.pipelines.training import TrainingPipeline
from pythae_modified.models.nn.benchmarks.mnist.resnets import Encoder_ResNet_VQVAE_MNIST, Decoder_ResNet_VQVAE_MNIST
#%%
config = BaseTrainerConfig(
    output_dir=path_project + 'anomaly_detection/model/my_model',
    # learning_rate=1e-3,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_epochs=100, # Change this to train the model a bit more
)


model_config = Simplified_VQVAEConfig(
    latent_dim=9,
    input_dim=(1, 13),
    commitment_loss_factor=0.25,
    quantization_loss_factor=1.0,
    num_embeddings=60,
    use_ema=True,
    decay=0.99
)

model = Simplified_VQVAE(
    model_config=model_config,
    encoder=Encoder_Linear_VQVAE_Missile(model_config),
    decoder=Decoder_Linear_VQVAE_Missile(model_config)
)

#%%
pipeline = TrainingPipeline(
    training_config=config,
    model=model
)
#%%
pipeline(
    train_data=train_dataset,
    eval_data=val_dataset
)
#%%
import os
from pythae_modified.models import AutoModel
#%%
last_training = sorted(os.listdir(path_project + 'anomaly_detection/model/my_model'))[-1]
trained_model = AutoModel.load_from_folder(os.path.join(path_project + 'anomaly_detection/model/my_model', last_training, 'final_model'))
#%%
from pythae_modified.samplers import NormalSampler
#%%
# create normal sampler
normal_samper = NormalSampler(
    model=trained_model
)
# %%
# sample
gen_data = normal_samper.sample(
    num_samples=25
)
# %%
import matplotlib.pyplot as plt

ymin = 0
ymax = 1
# %%
# # show results with normal sampler
# fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 10))
# for i in range(5):
#     for j in range(5):
#         axes[i][j].plot(gen_data.cpu().detach().numpy()[i * 5 + j])
#         axes[i][j].set_ylim(ymin, ymax)
# fig.suptitle('normal sampler')
# plt.tight_layout()
# plt.show()

# 异常检测
ground_truth = np.concatenate((np.zeros(5000 - window_size), np.ones(test_dataset.shape[0] - 5000 + window_size)))
predict = np.zeros(test_dataset.shape[0])
inputs = dict(data=torch.tensor(test_dataset, dtype=torch.float).to(device))
model_output = trained_model.calculate_elementwise_loss(inputs)
losses = model_output.loss.cpu().detach().numpy()
threshold = 200
i = 0
for loss in losses:
    if loss > threshold:
        predict[i] = 1
    i += 1
print('data length: ', test_dataset.shape[0])
print('threshold:   ', threshold)
print('accuracy:    ', accuracy_score(ground_truth, predict))

codebook_embeddings = trained_model.get_codebook()
# %%
# show codebook embeddings
fig, axes = plt.subplots(nrows=6, ncols=10, figsize=(40, 12))
fig.suptitle('codebook embeddings')
for i in range(6):
    for j in range(10):
        axes[i][j].plot(codebook_embeddings.cpu().detach().numpy()[i * 10 + j])
        axes[i][j].set_title('index: ' + str(i * 10 + j))
        axes[i][j].set_ylim(-15, 15)
plt.tight_layout()
plt.show()

codebook_reconstructions = trained_model.decoder(trained_model.get_codebook()).reconstruction
# %%
# show codebook reconstructions
fig, axes = plt.subplots(nrows=6, ncols=10, figsize=(40, 12))
fig.suptitle('codebook reconstructions')
for i in range(6):
    for j in range(10):
        axes[i][j].plot(codebook_reconstructions.cpu().detach().numpy()[i * 10 + j])
        axes[i][j].set_title('index: ' + str(i * 10 + j))
        axes[i][j].set_ylim(ymin, ymax)
plt.tight_layout()
plt.show()


from pythae_modified.samplers import GaussianMixtureSampler, GaussianMixtureSamplerConfig

# # set up GMM sampler config
# gmm_sampler_config = GaussianMixtureSamplerConfig(
#     n_components=10
# )
#
# # create gmm sampler
# gmm_sampler = GaussianMixtureSampler(
#     sampler_config=gmm_sampler_config,
#     model=trained_model
# )
#
# # fit the sampler
# gmm_sampler.fit(train_dataset)
# #%%
# # sample
# gen_data = gmm_sampler.sample(
#     num_samples=25
# )
# #%%
# # show results with gmm sampler
# fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 10))
# count = 0
# for i in range(3):
#     for j in range(5):
#         if count < 13:
#             axes[i][j].plot(gen_data[i*5 +j].cpu().squeeze(0))
#             count += 1
#         else:
#             break
# plt.title('Normal Data')
# plt.tight_layout()
# plt.show()
# %% md
## ... the other samplers work the same
# %% md

# 正样本
# 生成随机index
index = np.random.randint(5500, normal_dataset.shape[0], size=25)
data_to_plot = normal_dataset[index]

# show the true data
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 10))
fig.suptitle('Positive Data: true data')
for i in range(5):
    for j in range(5):
        axes[i][j].plot(data_to_plot[i * 5 + j])
        axes[i][j].set_title('index: ' + str(index[i * 5 + j]))
        axes[i][j].set_ylim(ymin, ymax)
plt.tight_layout()
plt.show()
# fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 10))
# count = 0
# for i in range(3):
#     for j in range(5):
#         if count < 13:
#             axes[i][j].plot(eval_dataset[:, i*5 +j])
#             count += 1
#         else:
#             break
# fig.suptitle('true data')
# plt.tight_layout()
# plt.show()

inputs = dict(data=torch.tensor(data_to_plot, dtype=torch.float).to(device))
model_output = trained_model.calculate_elementwise_loss(inputs)
embeddings = model_output.embeddings.cpu().detach().numpy()
quantized_indices = model_output.quantized_indices.cpu().detach().numpy().squeeze()
reconstructions = model_output.recon_x.cpu().detach().numpy()
# show embeddings
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 10))
fig.suptitle('Positive Data: embeddings')
for i in range(5):
    for j in range(5):
        axes[i][j].plot(embeddings[i * 5 + j])
        axes[i][j].set_ylim(-15, 15)
        axes[i][j].set_title('index: ' + str(index[i * 5 + j]) + ', neighbor: ' + str(quantized_indices[i * 5 + j]))
plt.tight_layout()
plt.show()

## Visualizing reconstructions
# reconstructions = trained_model.reconstruct(torch.tensor(data_to_plot, dtype=torch.float).to(device)).detach().cpu()
# show reconstructions
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 10))
fig.suptitle('Positive Data: reconstructions')
for i in range(5):
    for j in range(5):
        axes[i][j].plot(reconstructions[i * 5 + j])
        axes[i][j].set_title('index: ' + str(index[i * 5 + j]) + ', neighbor: ' + str(quantized_indices[i * 5 + j]))
        axes[i][j].set_ylim(ymin, ymax)
plt.tight_layout()
plt.show()
# fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 10))
# count = 0
# for i in range(3):
#     for j in range(5):
#         if count < 13:
#             axes[i][j].plot(reconstructions.cpu().detach().numpy()[:, i*5 +j])
#             count += 1
#         else:
#             break
# fig.suptitle('reconstructions')
# plt.tight_layout()
# plt.show()

# 负样本
# 生成随机index
index = np.random.randint(5500, test_dataset.shape[0], size=25)
data_to_plot = test_dataset[index]

# show the true data
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 10))
fig.suptitle('Negative Data: true data')
for i in range(5):
    for j in range(5):
        axes[i][j].plot(data_to_plot[i * 5 + j])
        axes[i][j].set_title('index: ' + str(index[i * 5 + j]))
        axes[i][j].set_ylim(-100, 100)
plt.tight_layout()
plt.show()
# fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 10))
# count = 0
# for i in range(3):
#     for j in range(5):
#         if count < 13:
#             axes[i][j].plot(eval_dataset[:, i*5 +j])
#             count += 1
#         else:
#             break
# fig.suptitle('true data')
# plt.tight_layout()
# plt.show()

inputs = dict(data=torch.tensor(data_to_plot, dtype=torch.float).to(device))
model_output = trained_model.calculate_elementwise_loss(inputs)
embeddings = model_output.embeddings.cpu().detach().numpy()
quantized_indices = model_output.quantized_indices.cpu().detach().numpy().squeeze()
reconstructions = model_output.recon_x.cpu().detach().numpy()
# show embeddings
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 10))
fig.suptitle('Negative Data: embeddings')
for i in range(5):
    for j in range(5):
        axes[i][j].plot(embeddings[i * 5 + j])
        axes[i][j].set_title('index: ' + str(index[i * 5 + j]) + ', neighbor: ' + str(quantized_indices[i * 5 + j]))
        axes[i][j].set_ylim(-15, 15)
plt.tight_layout()
plt.show()

## Visualizing reconstructions
# reconstructions = trained_model.reconstruct(torch.tensor(data_to_plot, dtype=torch.float).to(device)).detach().cpu()
# show reconstructions
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 10))
fig.suptitle('Negative Data: reconstructions')
for i in range(5):
    for j in range(5):
        axes[i][j].plot(reconstructions[i * 5 + j])
        axes[i][j].set_title('index: ' + str(index[i * 5 + j]) + ', neighbor: ' + str(quantized_indices[i * 5 + j]))
        axes[i][j].set_ylim(ymin, ymax)
plt.tight_layout()
plt.show()


# #%% md
# ## Visualizing interpolations
# #%%
# interpolations = trained_model.interpolate(eval_dataset[:5].to(device), eval_dataset[5:10].to(device), granularity=10).detach().cpu()
# #%%
# # show interpolations
# fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(10, 5))
#
# for i in range(5):
#     for j in range(10):
#         axes[i][j].imshow(interpolations[i, j].cpu().squeeze(0), cmap='gray')
#         axes[i][j].axis('off')
# plt.tight_layout(pad=0.)
# plt.show()
# class VAE(nn.Module):
#     def __init__(self, input_dim, hidden_dim, latent_dim):
#         super(VAE, self).__init__()
#
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, latent_dim * 2),
#         )
#
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, input_dim),
#             nn.Sigmoid()
#         )
#
#     def reparameterize(self, mu, log_var):
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mu + eps * std
#
#     def forward(self, x):
#         h = self.encoder(x)
#         mu, log_var = torch.chunk(h, 2, dim=1)
#         z = self.reparameterize(mu, log_var)
#         return self.decoder(z), mu, log_var
#

#
#
# model = VAE(input_dim=784, hidden_dim=400, latent_dim=20)
# optimizer = torch.optim.Adam(model.parameters())
#
# for epoch in range(num_epochs):
#     for batch in dataloader:
#         optimizer.zero_grad()
#         recon_batch, mu, log_var = model(batch)
#         loss = vae_loss(recon_batch, batch, mu, log_var)
#         loss.backward()
#         optimizer.step()
# print('Finished!')
