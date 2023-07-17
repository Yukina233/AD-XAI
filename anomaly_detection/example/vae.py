#%%
# Install the library
from typing import List

import torch
import torch.nn as nn

from pythae.models.base import BaseAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn.base_architectures import BaseDecoder, BaseEncoder
from pythae.models.nn.benchmarks.utils import ResBlock
class Encoder_ResNet_VAE_MNIST(BaseEncoder):
    """
    A ResNet encoder suited for MNIST and Variational Autoencoder-based models.

    It can be built as follows:

    .. code-block::

    #     >>> from pythae.models.nn.benchmarks.mnist import Encoder_ResNet_VAE_MNIST
    #     >>> from pythae.models import VAEConfig
    #     >>> model_config = VAEConfig(input_dim=(1, 28, 28), latent_dim=16)
    #     >>> encoder = Encoder_ResNet_VAE_MNIST(model_config)
    #     >>> encoder
    #     ... Encoder_ResNet_VAE_MNIST(
    #     ...   (layers): ModuleList(
    #     ...     (0): Sequential(
    #     ...       (0): Conv2d(1, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    #     ...     )
    #     ...     (1): Sequential(
    #     ...       (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    #     ...     )
    #     ...     (2): Sequential(
    #     ...       (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    #     ...     )
    #     ...     (3): Sequential(
    #     ...       (0): ResBlock(
    #     ...         (conv_block): Sequential(
    #     ...           (0): ReLU()
    #     ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     ...           (2): ReLU()
    #     ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
    #     ...         )
    #     ...       )
    #     ...       (1): ResBlock(
    #     ...         (conv_block): Sequential(
    #     ...           (0): ReLU()
    #     ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     ...           (2): ReLU()
    #     ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
    #     ...         )
    #     ...       )
    #     ...     )
    #     ...   )
    #     ...   (embedding): Linear(in_features=2048, out_features=16, bias=True)
    #     ...   (log_var): Linear(in_features=2048, out_features=16, bias=True)
    #     ... )
    #
    #
    # and then passed to a :class:`pythae.models` instance
    #
    #     >>> from pythae.models import VAE
    #     >>> model = VAE(model_config=model_config, encoder=encoder)
    #     >>> model.encoder == encoder
    #     ... True
    #
    # .. note::
    #
    #     Please note that this encoder is only suitable for Autoencoder based models since it only
    #     outputs the embeddings of the input data under the key `embedding`.
    #
    #     .. code-block::
    #
    #         >>> import torch
    #         >>> input = torch.rand(2, 1, 28, 28)
    #         >>> out = encoder(input)
    #         >>> out.embedding.shape
    #         ... torch.Size([2, 16])

    """

    def __init__(self, args: BaseAEConfig):
        BaseEncoder.__init__(self)

        self.input_dim = (1, 28, 28)
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Conv2d(self.n_channels, 64, 4, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(64, 128, 4, 2, padding=1)))

        layers.append(nn.Sequential(nn.Conv2d(128, 128, 3, 2, padding=1)))

        layers.append(
            nn.Sequential(
                ResBlock(in_channels=128, out_channels=32),
                ResBlock(in_channels=128, out_channels=32),
            )
        )

        self.layers = layers
        self.depth = len(layers)

        self.embedding = nn.Linear(128 * 4 * 4, args.latent_dim)
        self.log_var = nn.Linear(128 * 4 * 4, args.latent_dim)

    def forward(self, x: torch.Tensor, output_layer_levels: List[int] = None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the embeddings of the input data
            under the key `embedding`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `embedding_layer_i` where
            i is the layer's level."""
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth})."
                f"Got ({output_layer_levels})."
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = x

        for i in range(max_depth):
            out = self.layers[i](out)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"embedding_layer_{i+1}"] = out
            if i + 1 == self.depth:
                output["embedding"] = self.embedding(out.reshape(x.shape[0], -1))
                output["log_covariance"] = self.log_var(out.reshape(x.shape[0], -1))

        return output

class Decoder_ResNet_AE_MNIST(BaseDecoder):
    """
    A ResNet decoder suited for MNIST and Autoencoder-based
    models.

    .. code-block::

    #     >>> from pythae.models.nn.benchmarks.mnist import Decoder_ResNet_AE_MNIST
    #     >>> from pythae.models import VAEConfig
    #     >>> model_config = VAEConfig(input_dim=(1, 28, 28), latent_dim=16)
    #     >>> decoder = Decoder_ResNet_AE_MNIST(model_config)
    #     >>> decoder
    #     ... Decoder_ResNet_AE_MNIST(
    #     ...   (layers): ModuleList(
    #     ...     (0): Linear(in_features=16, out_features=2048, bias=True)
    #     ...     (1): ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    #     ...     (2): Sequential(
    #     ...       (0): ResBlock(
    #     ...         (conv_block): Sequential(
    #     ...           (0): ReLU()
    #     ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     ...           (2): ReLU()
    #     ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
    #     ...         )
    #     ...       )
    #     ...       (1): ResBlock(
    #     ...         (conv_block): Sequential(
    #     ...           (0): ReLU()
    #     ...           (1): Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     ...           (2): ReLU()
    #     ...           (3): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
    #     ...         )
    #     ...       )
    #     ...       (2): ReLU()
    #     ...     )
    #     ...     (3): Sequential(
    #     ...       (0): ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    #     ...       (1): ReLU()
    #     ...     )
    #     ...     (4): Sequential(
    #     ...       (0): ConvTranspose2d(64, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
    #     ...       (1): Sigmoid()
    #     ...     )
    #     ...   )
    #     ... )
    #
    #
    #
    # and then passed to a :class:`pythae.models` instance
    #
    #     >>> from pythae.models import VAE
    #     >>> model = VAE(model_config=model_config, decoder=decoder)
    #     >>> model.decoder == decoder
    #     ... True
    #
    # .. note::
    #
    #     Please note that this decoder is suitable for **all** models.
    #
    #     .. code-block::
    #
    #         >>> import torch
    #         >>> input = torch.randn(2, 16)
    #         >>> out = decoder(input)
    #         >>> out.reconstruction.shape
    #         ... torch.Size([2, 1, 28, 28])
    """

    def __init__(self, args: BaseAEConfig):
        BaseDecoder.__init__(self)

        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim
        self.n_channels = 1

        layers = nn.ModuleList()

        layers.append(nn.Linear(args.latent_dim, 128 * 4 * 4))

        layers.append(nn.ConvTranspose2d(128, 128, 3, 2, padding=1))

        layers.append(
            nn.Sequential(
                ResBlock(in_channels=128, out_channels=32),
                ResBlock(in_channels=128, out_channels=32),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1),
                nn.ReLU(),
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    64, self.n_channels, 3, 2, padding=1, output_padding=1
                ),
                nn.Sigmoid(),
            )
        )

        self.layers = layers
        self.depth = len(layers)

    def forward(self, z: torch.Tensor, output_layer_levels: List[int] = None):
        """Forward method

        Args:
            output_layer_levels (List[int]): The levels of the layers where the outputs are
                extracted. If None, the last layer's output is returned. Default: None.

        Returns:
            ModelOutput: An instance of ModelOutput containing the reconstruction of the latent code
            under the key `reconstruction`. Optional: The outputs of the layers specified in
            `output_layer_levels` arguments are available under the keys `reconstruction_layer_i`
            where i is the layer's level.
        """
        output = ModelOutput()

        max_depth = self.depth

        if output_layer_levels is not None:

            assert all(
                self.depth >= levels > 0 or levels == -1
                for levels in output_layer_levels
            ), (
                f"Cannot output layer deeper than depth ({self.depth})."
                f"Got ({output_layer_levels})"
            )

            if -1 in output_layer_levels:
                max_depth = self.depth
            else:
                max_depth = max(output_layer_levels)

        out = z

        for i in range(max_depth):
            out = self.layers[i](out)

            if i == 0:
                out = out.reshape(z.shape[0], 128, 4, 4)

            if output_layer_levels is not None:
                if i + 1 in output_layer_levels:
                    output[f"reconstruction_layer_{i+1}"] = out

            if i + 1 == self.depth:
                output["reconstruction"] = out

        return output

#%%
import torch
import torchvision.datasets as datasets

device = "cuda" if torch.cuda.is_available() else "cpu"


#%%
mnist_trainset = datasets.MNIST(root='../../data', train=True, download=True, transform=None)

train_dataset = mnist_trainset.data[:-10000].reshape(-1, 1, 28, 28) / 255.
eval_dataset = mnist_trainset.data[-10000:].reshape(-1, 1, 28, 28) / 255.
#%%
from pythae.models import VAE, VAEConfig
from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline

#%%
config = BaseTrainerConfig(
    output_dir='my_model',
    learning_rate=1e-4,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_epochs=10, # Change this to train the model a bit more
    optimizer_cls="AdamW",
    optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.99)}
)


model_config = VAEConfig(
    input_dim=(1, 28, 28),
    latent_dim=16
)

model = VAE(
    model_config=model_config,
    encoder=Encoder_ResNet_VAE_MNIST(model_config),
    decoder=Decoder_ResNet_AE_MNIST(model_config)
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
last_training = sorted(os.listdir('my_model'))[-1]
trained_model = AutoModel.load_from_folder(os.path.join('my_model', last_training, 'final_model'))
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
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(gen_data[i*5 +j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
#%%
from pythae.samplers import GaussianMixtureSampler, GaussianMixtureSamplerConfig
#%%
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
#%%
# sample
gen_data = gmm_sampler.sample(
    num_samples=25
)
#%%
# show results with gmm sampler
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(10, 10))

for i in range(5):
    for j in range(5):
        axes[i][j].imshow(gen_data[i*5 +j].cpu().squeeze(0), cmap='gray')
        axes[i][j].axis('off')
plt.tight_layout(pad=0.)
#%% md
## ... the other samplers work the same
#%% md
## Visualizing reconstructions
#%%
reconstructions = trained_model.reconstruct(eval_dataset[:25].to(device)).detach().cpu()
#%%
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