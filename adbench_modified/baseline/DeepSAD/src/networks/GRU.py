import torch.nn as nn
import torch.nn.functional as F

from adbench.baseline.DeepSAD.src.base import BaseNet

default_hidden_dim = 512
default_rep_dim = 256

class SMD_GRU_Encoder(BaseNet):

    def __init__(self, input_dim=36, hidden_dim=default_hidden_dim, rep_dim=default_rep_dim, bias=False):
        super().__init__()

        self.rep_dim = rep_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bias=bias)
        self.code = nn.Linear(hidden_dim, rep_dim, bias=bias)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch_size, 100, 36)
        h, _ = self.gru(x)
        h = h[:, -1, :]  # Take the output of the last time step
        return self.code(h)


class SMD_GRU_Decoder(BaseNet):

    def __init__(self, input_dim=36, hidden_dim=default_hidden_dim, rep_dim=default_rep_dim, bias=False):
        super().__init__()

        self.rep_dim = rep_dim
        self.gru = nn.GRU(rep_dim, hidden_dim, batch_first=True, bias=bias)
        self.reconstruction = nn.Linear(hidden_dim, input_dim, bias=bias)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, 100, 1)  # Repeat the latent vector for each time step
        h, _ = self.gru(x)
        h = self.reconstruction(h)
        h = h.transpose(1, 2)  # (batch_size, 36, 100)
        return self.output_activation(h)


class SMD_GRU_Autoencoder(BaseNet):

    def __init__(self, input_dim=36, hidden_dim=default_hidden_dim, rep_dim=default_rep_dim, bias=False):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = SMD_GRU_Encoder(input_dim, hidden_dim, rep_dim, bias)
        self.decoder = SMD_GRU_Decoder(input_dim, hidden_dim, rep_dim, bias)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
