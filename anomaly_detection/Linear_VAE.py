import torch
import torch.nn as nn
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score

import pythae_modified
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler, KBinsDiscretizer
from pythae_modified.models.nn import BaseEncoder, BaseDecoder
from pythae_modified.models.base.base_utils import ModelOutput


def get_dat(path):
    a = pd.read_csv(path, delim_whitespace=True)
    b = a[['dOmega_ib_b[0]', 'dOmega_ib_b[1]', 'dOmega_ib_b[2]', 'Gama', 'Theta_k', 'Psi_t', 'fb[0]', 'fb[1]',
           'fb[2]', 'Alfa', 'Beta', 'zmb', 'P']]
    b = np.array(b, dtype=float)
    return b


def create_window_dataset(dataset, k):
    # X是所有长度为k的窗口数据，Y是每个窗口对应的下一个数据点。
    dataX, dataY = [], []
    for i in range(len(dataset) - k):
        a = dataset[i:(i + k), :]
        dataX.append(a)
        dataY.append(dataset[i + k, :])
    return np.array(dataX), np.array(dataY)


def vae_loss(recon_x, x, mu, log_var):
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # 高斯分布KL散度计算
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_divergence

class Encoder_Linear_VAE_Missile(BaseEncoder):
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
            nn.Linear(self.hidden_dim, self.latent_dim * 2)
        )

    def forward(self, x: torch.Tensor):
        h = self.layers(x)
        mu, log_var = torch.chunk(h, 2, dim=1)
        output = ModelOutput(
            embedding=mu,
            log_covariance=log_var
        )
        return output


class Decoder_Linear_VAE_Missile(BaseDecoder):
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
path2 = path_project + 'anomaly_detection/data/ks2_7.dat'
ka_data = get_dat(path2)
path3 = path_project + 'raw/normal/zc3.dat'
new_data = get_dat(path3)
scaler = MinMaxScaler().fit(normal_data)

window_size = 1
train_dataset = scaler.transform(normal_data[:5000])
val_dataset = scaler.transform(ka_data[5000:])
test_dataset = scaler.transform(ka_data)

from pythae_modified.models import VAE, VAEConfig
from pythae_modified.trainers import BaseTrainerConfig
from pythae_modified.pipelines.training import TrainingPipeline
from pythae_modified.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST, Decoder_ResNet_AE_MNIST

config = BaseTrainerConfig(
    output_dir=path_project + 'anomaly_detection/model/my_model',
    # learning_rate=1e-3,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_epochs=1,  # Change this to train the model a bit more
    optimizer_cls="RMSprop",
)

model_config = VAEConfig(
    input_dim=(1, 13),
    latent_dim=9
)

model = VAE(
    model_config=model_config,
    encoder=Encoder_Linear_VAE_Missile(model_config),
    decoder=Decoder_Linear_VAE_Missile(model_config)
)
# %%
pipeline = TrainingPipeline(
    training_config=config,
    model=model
)
# %%
pipeline(
    train_data=train_dataset,
    eval_data=val_dataset
)
# %%
import os
from pythae_modified.models import AutoModel

# %%
last_training = sorted(os.listdir(path_project + 'anomaly_detection/model/my_model'))[-1]
trained_model = AutoModel.load_from_folder(
    os.path.join(path_project + 'anomaly_detection/model/my_model', last_training, 'final_model'))
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
)
# %%
import matplotlib.pyplot as plt

ymin = -260
ymax = 250
# %%
# show results with normal sampler
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 10))
for i in range(5):
    for j in range(5):
        axes[i][j].plot(gen_data.cpu().detach().numpy()[i * 5 + j])
        axes[i][j].set_ylim(ymin, ymax)
fig.suptitle('normal sampler')
plt.tight_layout()
plt.show()

# 异常检测
ground_truth = np.concatenate((np.zeros(5000 - window_size), np.ones(test_dataset.shape[0] - 5000 + window_size)))
predict = np.zeros(test_dataset.shape[0])
inputs = dict(data=torch.tensor(test_dataset, dtype=torch.float).to(device))
model_output = trained_model.calculate_elementwise_loss(inputs)
losses = model_output.loss.cpu().detach().numpy()
threshold = 20
i = 0
for loss in losses:
    if loss > threshold:
        predict[i] = 1
    i += 1
print('data length: ', test_dataset.shape[0])
print('threshold:   ', threshold)
print('accuracy:    ', accuracy_score(ground_truth, predict))

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
## Visualizing reconstructions
# %%
reconstructions = trained_model.reconstruct(torch.tensor(test_dataset[5500:], dtype=torch.float).to(device)).detach().cpu()
# %%
# show reconstructions
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 10))
fig.suptitle('reconstructions')
for i in range(5):
    for j in range(5):
        axes[i][j].plot(reconstructions.cpu().detach().numpy()[i * 5 + j])
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
# %%
# show the true data
data_to_plot = test_dataset[5500:]
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 10))
fig.suptitle('true data')
for i in range(5):
    for j in range(5):
        axes[i][j].plot(data_to_plot[i * 5 + j])
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
