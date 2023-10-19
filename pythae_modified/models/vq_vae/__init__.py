"""This module is the implementation of the Vector Quantized VAE proposed in 
(https://arxiv.org/abs/1711.00937).

Available samplers
-------------------

Normalizing flows sampler to come.

.. autosummary::
    ~pythae.samplers.GaussianMixtureSampler
    ~pythae.samplers.MAFSampler
    ~pythae.samplers.IAFSampler
    :nosignatures:
"""

from .vq_vae_config import VQVAEConfig, Simplified_VQVAEConfig
from .vq_vae_model import VQVAE, Simplified_VQVAE

__all__ = ["VQVAE", "Simplified_VQVAE", "VQVAEConfig", "Simplified_VQVAEConfig"]
