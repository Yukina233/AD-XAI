import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

class Custom_LeNet(BaseNet):

    def __init__(self, rep_dim=32):
        super().__init__()

        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, (3, 5), bias=False, padding=(1, 2))
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, (3, 5), bias=False, padding=(1, 2))
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        # 假设输入尺寸为 (5, 36)，经过两个核大小为 2 的最大池化层后，输出尺寸将是 (1, 9)
        self.fc1 = nn.Linear(4 * 1 * 9, self.rep_dim, bias=False)

    def forward(self, x):
        x = x.view(-1, 1, 5, 36)
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(int(x.size(0)), -1)
        x = self.fc1(x)
        return x

class Custom_LeNet_Decoder(BaseNet):

    def __init__(self, rep_dim=32):
        super().__init__()

        self.rep_dim = rep_dim

        # 解码器网络
        self.deconv1 = nn.ConvTranspose2d(1, 8, (3, 5), bias=False, padding=(1, 2))
        self.bn3 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(8, 4, (3, 5), bias=False, padding=(1, 2))
        self.bn4 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(4, 1, (3, 5), bias=False, padding=(1, 2))

        # Fully connected layer to produce the final output shape
        self.fc2 = nn.Linear(256, 180)  # Adjust the input size as needed
    def forward(self, x):
        # 假设表示维度 rep_dim 为 32
        x = x.view(int(x.size(0)), 1, 2, 16)  # 因为 1 * 2 * 16 = 32
        x = F.interpolate(F.leaky_relu(x), scale_factor=(1, 2))
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=(1, 2))
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=(1, 2))
        x = self.deconv3(x)
        x = F.leaky_relu(x)

        x = x.view(int(x.size(0)), -1)  # Flatten the feature map
        x = self.fc2(x)  # Fully connected layer to get the output shape (180)
        return x

class Custom_LeNet_Autoencoder(BaseNet):

    def __init__(self, rep_dim=32):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = Custom_LeNet(rep_dim=rep_dim)
        self.decoder = Custom_LeNet_Decoder(rep_dim=rep_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x