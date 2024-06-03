import torch.nn as nn
import torch.nn.functional as F

from adbench.baseline.DeepSAD.src.base import BaseNet


class Simple_Dense(BaseNet):

    def __init__(self, x_dim, h_dims=[128, 64], rep_dim=32, bias=False):
        super().__init__()

        self.rep_dim = rep_dim

        self.code = nn.Linear(x_dim, rep_dim, bias=bias)

    def forward(self, x):
        x = x.view(int(x.size(0)), -1)
        return self.code(x)


class Simple_Dense_Decoder(BaseNet):

    def __init__(self, x_dim, h_dims=[64, 128], rep_dim=32, bias=False):
        super().__init__()

        self.rep_dim = rep_dim

        self.reconstruction = nn.Linear(rep_dim, x_dim, bias=bias)
        # self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = x.view(int(x.size(0)), -1)

        x = self.reconstruction(x)
        # return self.output_activation(x)
        return x

class Simple_Dense_Autoencoder(BaseNet):

    def __init__(self, x_dim, h_dims=[128, 64], rep_dim=32, bias=False):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = Simple_Dense(x_dim, h_dims, rep_dim, bias)
        self.decoder = Simple_Dense_Decoder(x_dim, list(reversed(h_dims)), rep_dim, bias)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Linear_BN_leakyReLU(nn.Module):
    """
    A nn.Module that consists of a Linear layer followed by BatchNorm1d and a leaky ReLu activation
    """

    def __init__(self, in_features, out_features, bias=False, eps=1e-04):
        super(Linear_BN_leakyReLU, self).__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features, eps=eps, affine=bias)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.linear(x)))
