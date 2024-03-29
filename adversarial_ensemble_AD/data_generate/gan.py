import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from adbench_modified.baseline.DeepSAD.src.run import DeepSAD

path_project = '/home/yukina/Missile_Fault_Detection/project'


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class Adversarial_Generator:
    def __init__(self, config=None):
        # 默认参数
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
        parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
        parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
        parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
        parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
        parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
        parser.add_argument("--channels", type=int, default=1, help="number of image channels")
        parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
        parser.add_argument("--path_detector", type=str,
                            default=os.path.join(path_project, 'adversarial_ensemble_AD/models/ensemble/n=2'),
                            help="path to load detector")
        default_params = parser.parse_args()
        for key, value in vars(default_params).items():
            setattr(self, key, value)

        # 设置传入参数
        if config is not None:
            for key, value in config.items():
                setattr(self, key, value)
        self.img_shape = (self.channels, self.img_size, self.img_size)
        self.generator = Generator(self.latent_dim, self.img_shape)
        self.discriminator = Discriminator(self.img_shape)
        self.adversarial_loss = torch.nn.BCELoss()
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.adversarial_loss.cuda()

    def calculate_entropy(self, X, y, tau=1, path_plot=None):
        detectors = []
        scores = []
        for model in os.listdir(self.path_detector):
            detector = DeepSAD(seed=self.seed, load_model=os.path.join(self.path_detector, model))
            detector.load_model_from_file()
            detectors.append(detector)
            score, _ = detector.predict_score(X)
            scores.append(score)
        scores = np.array(scores)
        prob = np.exp(1/(scores * tau)) / np.sum(np.exp(1/(scores * tau)), axis=0)
        entropys = -np.sum(prob * np.log(prob), axis=0)

        return entropys

    def calculate_scores(self, X, y, tau=1, path_plot=None):
        detectors = []
        scores = []
        for model in os.listdir(self.path_detector):
            detector = DeepSAD(seed=self.seed, load_model=os.path.join(self.path_detector, model))
            detector.load_model_from_file()
            detectors.append(detector)
            score, _ = detector.predict_score(X)
            scores.append(score)
        scores = np.array(scores)

        # 按照标签绘制scores直方图
        for num, score in enumerate(scores):
            for i in range(0, 2):
                plt.hist(score[y == i], bins=1000, alpha=0.5, label=str(i), density=True)
            plt.legend()
            plt.title("Scores Histogram")
            plt.xlabel("Scores")
            plt.ylabel("Frequency")
            plt.show()
            plt.savefig(path_plot + f'_model={num}.png')
            plt.close()

        return scores

    def calculate_reps(self, X, y, tau=1, path_plot=None):
        detectors = []
        scores = []
        for num, model in enumerate(os.listdir(self.path_detector)):
            detector = DeepSAD(seed=self.seed, load_model=os.path.join(self.path_detector, model))
            detector.load_model_from_file()
            detectors.append(detector)
            score, outputs = detector.predict_score(X)

            X_plot = np.concatenate((np.array(outputs), np.array(detector.deepSAD.c).reshape(1, -1)))
            tsne = TSNE(n_components=2, random_state=0)  # n_components表示目标维度

            X_2d = tsne.fit_transform(X_plot) # 对数据进行降维处理

            center = X_2d[-1]
            X_2d = X_2d[:-1]

            plt.figure(figsize=(8, 6))
            if y is not None:
                # 如果有目标数组，根据不同的类别用不同的颜色绘制
                for i in np.unique(y):
                    plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], label=i, alpha=0.5)
                plt.legend()
            else:
                # 如果没有目标数组，直接绘制
                plt.scatter(X_2d[:, 0], X_2d[:, 1])
            plt.scatter(center[0], center[1], c='red', marker='x', label='center')
            plt.title('t-SNE Visualization of rep after training')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.show()
            plt.savefig(path_plot + f'_model={num}.png')
            plt.close()

        return outputs
    # def train(self, dataloader):
    #     for epoch in range(self.n_epochs):
    #         for i, (imgs, _) in enumerate(dataloader):
    #
    #             # Adversarial ground truths
    #             valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
    #             fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
    #
    #             # Configure input
    #             real_imgs = Variable(imgs.type(Tensor))
    #
    #             # -----------------
    #             #  Train Generator
    #             # -----------------
    #
    #             self.optimizer_G.zero_grad()
    #
    #             # Sample noise as generator input
    #             z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))))
    #
    #             # Generate a batch of images
    #             gen_imgs = self.generator(z)
    #
    #             # Loss measures generator's ability to fool the discriminator
    #             g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)
    #
    #             g_loss.backward()
    #             self.optimizer_G.step()
    #
    #             # ---------------------
    #             #  Train Discriminator
    #             # ---------------------
    #
    #             self.optimizer_D.zero_grad()
    #
    #             # Measure discriminator's ability to classify real from generated samples
    #             real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
    #             fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
    #             d_loss = (real_loss + fake_loss) / 2
    #
    #             d_loss.backward()
    #             self.optimizer_D.step()
    #
    #             print(
    #                 "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
    #                 % (epoch, self.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
    #             )
    #
    #             batches_done = epoch * len(dataloader) + i
    #             if batches_done % self.sample_interval == 0:
    #                 save_image(gen_imgs.data[:25], os.path.join(path_save, "%d.png" % batches_done), nrow=5,
    #                            normalize=True)
    #
    #     return self.generator



# path_data = os.path.join(path_project, 'data/mnist')
# path_save = os.path.join(path_project, 'adversarial_ensemble_AD/original_gan', "samples")
# os.makedirs(path_save, exist_ok=True)
#
# # Configure data loader
#
# os.makedirs(path_data, exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         path_data,
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#         ),
#     ),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )
#
# # Optimizers
#
#
# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#
# # ----------
# #  Training
# # ----------
#
# for epoch in range(opt.n_epochs):
#     for i, (imgs, _) in enumerate(dataloader):
#
#         # Adversarial ground truths
#         valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
#         fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
#
#         # Configure input
#         real_imgs = Variable(imgs.type(Tensor))
#
#         # -----------------
#         #  Train Generator
#         # -----------------
#
#         optimizer_G.zero_grad()
#
#         # Sample noise as generator input
#         z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
#
#         # Generate a batch of images
#         gen_imgs = generator(z)
#
#         # Loss measures generator's ability to fool the discriminator
#         g_loss = adversarial_loss(discriminator(gen_imgs), valid)
#
#         g_loss.backward()
#         optimizer_G.step()
#
#         # ---------------------
#         #  Train Discriminator
#         # ---------------------
#
#         optimizer_D.zero_grad()
#
#         # Measure discriminator's ability to classify real from generated samples
#         real_loss = adversarial_loss(discriminator(real_imgs), valid)
#         fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
#         d_loss = (real_loss + fake_loss) / 2
#
#         d_loss.backward()
#         optimizer_D.step()
#
#         print(
#             "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
#             % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
#         )
#
#         batches_done = epoch * len(dataloader) + i
#         if batches_done % opt.sample_interval == 0:
#             save_image(gen_imgs.data[:25], os.path.join(path_save, "%d.png" % batches_done), nrow=5, normalize=True)
