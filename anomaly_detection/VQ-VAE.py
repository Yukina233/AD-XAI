import torch
import torch.nn as nn
import torchvision.datasets as datasets
import pythae
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
def get_dat(path):
    a = pd.read_csv(path,delim_whitespace=True)
    b = a[['dOmega_ib_b[0]', 'dOmega_ib_b[1]', 'dOmega_ib_b[2]', 'Gama', 'Theta_k', 'Psi_t', 'fb[0]', 'fb[1]',
           'fb[2]', 'Alfa', 'Beta', 'zmb', 'P']]
    b = np.array(b, dtype=float)
    return b

class Encoder_Linear_VAE_Missile(BaseEncoder):
    def __init__(self, args):
        BaseEncoder.__init__(self)

        self.input_dim = 13
        self.hidden_dim = 128
        self.latent_dim = args.latent_dim

        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
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
scaler = RobustScaler().fit(normal_data)


train_dataset = scaler.transform(normal_data[:6000])
eval_dataset = scaler.transform(normal_data[:6000])

from pythae.models import VQVAE, VQVAEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline
from pythae.models.nn.benchmarks.mnist.resnets import Encoder_ResNet_VQVAE_MNIST, Decoder_ResNet_VQVAE_MNIST
#%%
config = BaseTrainerConfig(
    output_dir=path_project + 'anomaly_detection/model/my_model',
    # learning_rate=1e-3,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_epochs=10, # Change this to train the model a bit more
)


model_config = VQVAEConfig(
    latent_dim=9,
    input_dim=(1, 13),
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
from pythae.models import AutoModel
#%%
last_training = sorted(os.listdir(path_project + 'anomaly_detection/model/my_model'))[-1]
trained_model = AutoModel.load_from_folder(os.path.join(path_project + 'anomaly_detection/model/my_model', last_training, 'final_model'))
#%%
from pythae.samplers import NormalSampler
#%%
# create normal sampler
normal_samper = NormalSampler(
    model=trained_model
)
#%%
# sample
gen_data = normal_samper.sample(
    num_samples=25
)
#%%
import matplotlib.pyplot as plt
#%%
# show results with normal sampler
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 10))
for i in range(5):
    for j in range(5):
            axes[i][j].plot(gen_data.cpu().detach().numpy()[i*5 +j])
fig.suptitle('normal sampler')
plt.tight_layout()
plt.show()

from pythae.samplers import GaussianMixtureSampler, GaussianMixtureSamplerConfig

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
#%% md
## ... the other samplers work the same
#%% md
## Visualizing reconstructions
#%%
reconstructions = trained_model.reconstruct(torch.tensor(eval_dataset, dtype=torch.float).to(device)).detach().cpu()
#%%
# show reconstructions
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 10))
count = 0
for i in range(3):
    for j in range(5):
        if count < 13:
            axes[i][j].plot(reconstructions.cpu().detach().numpy()[:, i*5 +j])
            count += 1
        else:
            break
fig.suptitle('reconstructions')
plt.tight_layout()
plt.show()
#%%
# show the true data
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(20, 10))
count = 0
for i in range(3):
    for j in range(5):
        if count < 13:
            axes[i][j].plot(eval_dataset[:, i*5 +j])
            count += 1
        else:
            break
fig.suptitle('true data')
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
#     def vae_loss(self, recon_x, x, mu, log_var):
#         recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
#         # 高斯分布KL散度计算
#         kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
#         return recon_loss + kl_divergence
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