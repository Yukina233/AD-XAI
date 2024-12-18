import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torchvision.utils import save_image

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from adbench_modified.baseline.DeepSAD.src.run import DeepSAD

path_project = '/home/yukina/Missile_Fault_Detection/project'


# class Generator(nn.Module):
#     def __init__(self, latent_dim, img_shape):
#         super(Generator, self).__init__()
#         self.latent_dim = latent_dim
#         self.img_shape = img_shape
#
#         self.model = nn.Sequential(
#             nn.Linear(self.latent_dim, 500),
#             nn.ReLU(),
#             nn.Linear(500, 500),
#             nn.ReLU(),
#             nn.Linear(500, int(np.prod(self.img_shape)))
#         )
#
#     def forward(self, z):
#         img = self.model(z)
#         img = img.view(img.size(0), *self.img_shape)
#         return img
#
#
# class Discriminator(nn.Module):
#     def __init__(self, img_shape):
#         super(Discriminator, self).__init__()
#         self.img_shape = img_shape
#
#         self.model = nn.Sequential(
#             nn.Linear(int(np.prod(self.img_shape)), 500),
#             nn.ReLU(),
#             nn.Linear(500, 500),
#             nn.ReLU(),
#             nn.Linear(500, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, img):
#         img_flat = img.view(img.size(0), -1)
#         validity = self.model(img_flat)
#         return validity

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
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
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


# class Generator(nn.Module):
#     def __init__(self, latent_dim, img_shape):
#         super(Generator, self).__init__()
#         self.latent_dim = latent_dim
#         self.img_shape = img_shape
#
#         def block(in_feat, out_feat, normalize=True):
#             layers = [nn.Linear(in_feat, out_feat)]
#             if normalize:
#                 layers.append(nn.BatchNorm1d(out_feat, 0.8))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers
#
#         self.model = nn.Sequential(
#             nn.Linear(self.latent_dim, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(inplace=True),
#             nn.Linear(64, int(64)),
#             nn.BatchNorm1d(int(64)),
#             nn.ReLU(inplace=True),
#             nn.Linear(int(64), int(np.prod(self.img_shape)))
#         )
#         # for layer in self.model:
#         #     if isinstance(layer, nn.Linear):
#         #         layer.bias = None
#         # self.model = nn.Sequential(
#         #     *block(self.latent_dim, 64, normalize=False),
#         #     *block(64, 128),
#         #     *block(128, 256),
#         #     # *block(512, 1024),
#         #     nn.Linear(256, int(np.prod(self.img_shape))),
#         #     nn.Tanh()
#         # )
#
#     def forward(self, z):
#         img = self.model(z)
#         img = img.view(img.size(0), *self.img_shape)
#         return img
#
#
# class Discriminator(nn.Module):
#     def __init__(self, img_shape):
#         super(Discriminator, self).__init__()
#         self.img_shape = img_shape
#
#         self.model = nn.Sequential(
#             nn.Linear(int(np.prod(self.img_shape)), 128),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Linear(128, int(128)),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Linear(int(128), 1),
#             nn.Sigmoid()
#         )
#         # self.model = nn.Sequential(
#         #     nn.Linear(int(np.prod(self.img_shape)), 128),
#         #     nn.LeakyReLU(0.01, inplace=True),
#         #     nn.Linear(128, 64),
#         #     nn.LeakyReLU(0.01, inplace=True),
#         #     nn.Linear(64, 1),
#         #     nn.Sigmoid(),
#         # )
#
#     def forward(self, img):
#         img_flat = img.view(img.size(0), -1)
#         validity = self.model(img_flat)
#
#         return validity


class Adversarial_Generator:
    def __init__(self, config=None, DeepSAD_config=None):
        # 默认参数
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--n_epochs", type=int, default=5, help="number of epochs of training")
        parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
        parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
        parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--lam1", type=float, default=2, help="lambda parameter for entropy loss")
        parser.add_argument("--lam2", type=float, default=1, help="lambda parameter for mean ensemble loss")
        parser.add_argument("--tau1", type=float, default=1, help="tau for entropy calculation")
        parser.add_argument("--tau2", type=float, default=0.001, help="tau for mean ensemble loss calculation")
        parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
        parser.add_argument("--latent_dim", type=int, default=32, help="dimensionality of the latent space")
        parser.add_argument("--img_size", type=int, default=39, help="size of each image dimension")
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

        self.DeepSAD_config = DeepSAD_config

        self.img_shape = (self.channels, self.img_size)
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
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def load_model(self, path):
        self.generator.load_state_dict(torch.load(os.path.join(path, 'generator.pth')))
        self.discriminator.load_state_dict(torch.load(os.path.join(path, 'discriminator.pth')))

    def save_model(self, path):
        torch.save(self.generator.state_dict(), os.path.join(path, 'generator.pth'))
        torch.save(self.discriminator.state_dict(), os.path.join(path, 'discriminator.pth'))

    def calculate_entropy_numpy(self, X, tau=1):
        detectors = []
        scores = []
        for model in os.listdir(self.path_detector):
            detector = DeepSAD(seed=self.seed, load_model=os.path.join(self.path_detector, model))
            detector.load_model_from_file()
            detectors.append(detector)
            score, _ = detector.predict_score(X)
            scores.append(score)
        scores = np.array(scores)
        prob = np.exp(1 / (scores * tau)) / np.sum(np.exp(1 / (scores * tau)), axis=0)
        entropys = -np.sum(prob * np.log(prob), axis=0)

        return entropys

    def calculate_regular_loss(self, X, tau1=1):
        detectors = []
        scores = []
        for model in os.listdir(self.path_detector):
            if not model.startswith('DeepSAD'):
                continue
            detector = DeepSAD(seed=self.seed, load_model=os.path.join(self.path_detector, model),
                               config=self.DeepSAD_config)
            detector.load_model_from_file(input_size=self.img_shape[1])
            detectors.append(detector)

            # detector.deepSAD.net.eval()
            detector.deepSAD.net.to('cuda')
            outputs = detector.deepSAD.net(X)
            center = torch.tensor(detector.deepSAD.c, device='cuda')
            score = torch.sum((outputs - center) ** 2, dim=1)
            scores.append(score)
        scores = torch.stack(scores)
        # prob = torch.exp(1 / (scores * tau1)) / torch.sum(torch.exp(1 / (scores * tau1)), dim=0)
        # if torch.isnan(prob).any():
        #     print('prob nan')
        #     entropys = torch.tensor([0.0], device='cuda')
        # else:
        #     entropys = -torch.sum(prob * torch.log(prob), dim=0)
        #     if torch.isnan(entropys).any():
        #         print('entropys nan')
        #         entropys = torch.tensor([0.0], device='cuda')

        var_ensemble_loss = torch.std(scores, dim=0)

        mean_ensemble_loss = -torch.mean(scores, dim=0)
        # return entropys, mean_ensemble_loss

        return var_ensemble_loss, mean_ensemble_loss

    def calculate_pull_away_loss(self, X):
        """
            Calculate the pull-away term for a batch of features.

            Args:
                features (torch.Tensor): A tensor of shape (batch_size, feature_dim).

            Returns:
                torch.Tensor: The pull-away term.
            """
        batch_size = X.size(0)

        X = X.view(batch_size, -1)

        # Normalize the features
        normalized_features = F.normalize(X, p=2, dim=1)

        # Calculate the cosine similarity matrix
        cosine_similarity = torch.mm(normalized_features, normalized_features.t())

        # Subtract the identity matrix to remove self-similarity
        identity_matrix = torch.eye(batch_size, device=X.device)
        cosine_similarity = cosine_similarity - identity_matrix

        # Calculate the pull-away term
        pull_away_term = torch.sum(cosine_similarity ** 2) / (batch_size * (batch_size - 1))

        # # Calculate the Euclidean distance matrix
        # dist_matrix = torch.cdist(normalized_features, normalized_features, p=2)
        #
        # # Subtract the identity matrix to remove self-similarity by setting diagonal to infinity
        # mask = torch.eye(batch_size, device=X.device).bool()
        # # dist_matrix = dist_matrix.masked_fill(mask, float('inf'))
        # dist_matrix = dist_matrix.masked_fill(mask, float(0))
        #
        # # Calculate the pull-away term
        # pull_away_term = torch.sum((dist_matrix ** 2)) / (batch_size * (batch_size - 1))

        return pull_away_term

    def calculate_entropy_test(self, X, tau=1):
        detectors = []
        scores = []
        for model in os.listdir(self.path_detector):
            detector = DeepSAD(seed=self.seed, load_model=os.path.join(self.path_detector, model),
                               config=self.DeepSAD_config)
            detector.load_model_from_file()
            detectors.append(detector)

            detector.deepSAD.net.eval()
            detector.deepSAD.net.to('cuda')
            outputs = detector.deepSAD.net(X)
            center = torch.tensor(detector.deepSAD.c, device='cuda')
            score = torch.sum((outputs - center) ** 2, dim=1)
            scores.append(score)
        scores = torch.stack(scores)
        prob = torch.exp(1 / (scores * tau)) / torch.sum(torch.exp(1 / (scores * tau)), dim=0)
        entropys = -torch.sum(prob * torch.log(prob), dim=0)

        return entropys

    def calculate_scores(self, X, y, tau=1, path_plot=None):
        detectors = []
        scores = []
        for model in os.listdir(self.path_detector):
            detector = DeepSAD(seed=self.seed, load_model=os.path.join(self.path_detector, model),
                               config=self.DeepSAD_config)
            detector.load_model_from_file()
            detectors.append(detector)
            score, _ = detector.predict_score(X)
            scores.append(score)
        scores = np.array(scores)

        # 按照标签绘制scores直方图
        for num, score in enumerate(scores):
            for i in range(0, 2):
                plt.hist(score[y == i], bins=50, alpha=0.5, label=str(i), density=False)
            plt.legend()
            plt.title("Scores Histogram")
            plt.xlabel("Scores")
            plt.ylabel("Frequency")
            plt.show()
            plt.savefig(path_plot + f'_model={num}.png')
            plt.close()

        return scores

    def train_origin(self, dataloader):
        loss_gen = []
        loss_dis = []

        # 分别记录生成器的两部分loss
        loss_gen_adv = []
        loss_gen_var_ensemble = []
        loss_gen_mean_ensemble = []
        loss_gen_pull_away = []
        loss_dis_wass = []
        loss_dis_gp = []
        batches_done = 0
        for epoch in range(self.pre_epochs):
            loss_gen_batch = []
            loss_dis_batch = []

            # 分别记录生成器的两部分loss
            loss_gen_adv_batch = []
            loss_gen_var_ensemble_batch = []
            loss_gen_mean_ensemble_batch = []
            loss_gen_pull_away_batch = []
            loss_dis_wass_batch = []
            loss_dis_gp_batch = []

            for i, (samples, _) in enumerate(dataloader):

                # Configure input
                real_samples = Variable(samples.type(self.Tensor))

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Sample noise as generator input
                z = Variable(self.Tensor(np.random.normal(0, 1, (samples.shape[0], self.latent_dim))))

                # Generate a batch of images
                gen_samples = self.generator(z)

                # Real images
                real_validity = self.discriminator(real_samples)
                # Fake images
                fake_validity = self.discriminator(gen_samples)
                # Gradient penalty
                gradient_penalty = self.compute_gradient_penalty(self.discriminator, real_samples.data,
                                                                 gen_samples.data)
                # Adversarial loss
                wassestein_distance = torch.mean(real_validity) - torch.mean(fake_validity)
                d_loss = -wassestein_distance + self.lambda_gp * gradient_penalty

                d_loss.backward()
                self.optimizer_D.step()

                self.optimizer_G.zero_grad()

                # Train the generator every n_critic steps
                if i % self.n_critic == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of images
                    gen_samples = self.generator(z)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = self.discriminator(gen_samples)
                    adv_loss = -torch.mean(fake_validity)

                    var_ensemble_loss, mean_ensemble_loss = self.calculate_regular_loss(X=gen_samples, tau1=self.tau1)
                    var_ensemble_loss = torch.mean(var_ensemble_loss)
                    mean_ensemble_loss = torch.mean(mean_ensemble_loss)

                    pull_away_loss = self.calculate_pull_away_loss(gen_samples)

                    g_loss = adv_loss

                    g_loss.backward()
                    self.optimizer_G.step()

                    batches_done += self.n_critic

                    if i % 70 == 0:
                        print(
                            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [adv loss: %f] [var loss: %f] [mean loss: %f] [pull away loss: %f] [Wass: %f] [GP: %f]"
                            % (epoch, self.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), adv_loss.item(),
                               var_ensemble_loss.item(), mean_ensemble_loss.item(), pull_away_loss.item(),
                               wassestein_distance.item(), gradient_penalty.item()
                               ))

                    loss_gen_batch.append(g_loss.item())
                    loss_dis_batch.append(d_loss.item())
                    loss_gen_adv_batch.append(adv_loss.item())
                    loss_gen_var_ensemble_batch.append(var_ensemble_loss.item())
                    loss_gen_mean_ensemble_batch.append(mean_ensemble_loss.item())
                    loss_gen_pull_away_batch.append(pull_away_loss.item())
                    loss_dis_wass_batch.append(wassestein_distance.item())
                    loss_dis_gp_batch.append(gradient_penalty.item())

            loss_gen.append(np.mean(loss_gen_batch))
            loss_dis.append(np.mean(loss_dis_batch))
            loss_gen_adv.append(np.mean(loss_gen_adv_batch))
            loss_gen_var_ensemble.append(np.mean(loss_gen_var_ensemble_batch))
            loss_gen_mean_ensemble.append(np.mean(loss_gen_mean_ensemble_batch))
            loss_gen_pull_away.append(np.mean(loss_gen_pull_away_batch))
            loss_dis_wass.append(np.mean(loss_dis_wass_batch))
            loss_dis_gp.append(np.mean(loss_dis_gp_batch))

        loss_train = {
            'loss_gen': np.array(loss_gen),
            'loss_dis': np.array(loss_dis),
            'loss_gen_adv': np.array(loss_gen_adv),
            'loss_gen_entropy': np.array(loss_gen_var_ensemble),
            'loss_gen_mean_ensemble': np.array(loss_gen_mean_ensemble),
            'loss_gen_pull_away': np.array(loss_gen_pull_away),
            'loss_dis_wass': np.array(loss_dis_wass),
            'loss_dis_gp': np.array(loss_dis_gp)
        }

        return loss_train

    def train(self, dataloader):
        loss_gen = []
        loss_dis = []

        # 分别记录生成器的两部分loss
        loss_gen_adv = []
        loss_gen_var_ensemble = []
        loss_gen_mean_ensemble = []
        loss_gen_pull_away = []
        for epoch in range(self.n_epochs):
            loss_gen_batch = []
            loss_dis_batch = []

            # 分别记录生成器的两部分loss
            loss_gen_adv_batch = []
            loss_gen_var_ensemble_batch = []
            loss_gen_mean_ensemble_batch = []
            loss_gen_pull_away_batch = []
            for i, (samples, _) in enumerate(dataloader):
                if samples.shape[0] != self.batch_size:
                    continue
                # Adversarial ground truths
                valid = Variable(self.Tensor(samples.size(0), 1).fill_(1.0), requires_grad=False)
                fake = Variable(self.Tensor(samples.size(0), 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_samples = Variable(samples.type(self.Tensor))

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(self.Tensor(np.random.normal(0, 1, (samples.shape[0], self.latent_dim))))

                # Generate a batch of images
                gen_samples = self.generator(z)

                pull_away_loss = self.calculate_pull_away_loss(gen_samples)

                # Loss measures generator's ability to fool the discriminator
                adv_loss = self.adversarial_loss(self.discriminator(gen_samples), valid)
                var_ensemble_loss, mean_ensemble_loss = self.calculate_regular_loss(X=gen_samples, tau1=self.tau1)
                var_ensemble_loss = torch.mean(var_ensemble_loss)
                mean_ensemble_loss = torch.mean(mean_ensemble_loss)

                g_loss = adv_loss + self.lam1 * var_ensemble_loss + self.lam2 * torch.mean(
                    mean_ensemble_loss) + self.lam3 * pull_away_loss
                # g_loss = self.lam1 * entropy_loss + self.lam2 * torch.mean(mean_ensemble_loss)

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.adversarial_loss(self.discriminator(real_samples), valid)
                fake_loss = self.adversarial_loss(self.discriminator(gen_samples.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                self.optimizer_D.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [adv loss: %f] [var loss: %f] [mean loss: %f] [pull away loss: %f]"
                    % (epoch, self.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), adv_loss.item(),
                       var_ensemble_loss.item(), mean_ensemble_loss.item(), pull_away_loss.item()
                       ))

                loss_gen_batch.append(g_loss.item())
                loss_dis_batch.append(d_loss.item())
                loss_gen_adv_batch.append(adv_loss.item())
                loss_gen_var_ensemble_batch.append(var_ensemble_loss.item())
                loss_gen_mean_ensemble_batch.append(mean_ensemble_loss.item())
                loss_gen_pull_away_batch.append(pull_away_loss.item())

            loss_gen.append(np.mean(loss_gen_batch))
            loss_dis.append(np.mean(loss_dis_batch))
            loss_gen_adv.append(np.mean(loss_gen_adv_batch))
            loss_gen_var_ensemble.append(np.mean(loss_gen_var_ensemble_batch))
            loss_gen_mean_ensemble.append(np.mean(loss_gen_mean_ensemble_batch))
            loss_gen_pull_away.append(np.mean(loss_gen_pull_away_batch))

        loss_train = {
            'loss_gen': np.array(loss_gen),
            'loss_dis': np.array(loss_dis),
            'loss_gen_adv': np.array(loss_gen_adv),
            'loss_gen_entropy': np.array(loss_gen_var_ensemble),
            'loss_gen_mean_ensemble': np.array(loss_gen_mean_ensemble),
            'loss_gen_pull_away': np.array(loss_gen_pull_away)
        }

        return loss_train

        # batches_done = epoch * len(dataloader) + i
        # if batches_done % self.sample_interval == 0:
        #     path_save = os.path.join(path_project, 'adversarial_ensemble_AD/log/gan', "samples")
        #     save_image(gen_samples.data[:25], os.path.join(path_save, "%d.png" % batches_done), nrow=5,
        #                normalize=True)

    def train_no_GAN(self, dataloader):
        loss_gen = []

        # 分别记录生成器的两部分loss
        loss_gen_entropy = []
        loss_gen_mean_ensemble = []
        loss_gen_pull_away = []
        for epoch in range(self.n_epochs):
            loss_gen_batch = []
            loss_dis_batch = []

            # 分别记录生成器的两部分loss
            loss_gen_adv_batch = []
            loss_gen_entropy_batch = []
            loss_gen_mean_ensemble_batch = []
            loss_gen_pull_away_batch = []
            for i, (samples, _) in enumerate(dataloader):
                if samples.shape[0] != self.batch_size:
                    continue
                # Adversarial ground truths
                valid = Variable(self.Tensor(samples.size(0), 1).fill_(1.0), requires_grad=False)
                fake = Variable(self.Tensor(samples.size(0), 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_samples = Variable(samples.type(self.Tensor))

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(self.Tensor(np.random.normal(0, 1, (samples.shape[0], self.latent_dim))))

                # Generate a batch of images
                gen_samples = self.generator(z)

                pull_away_loss = self.calculate_pull_away_loss(gen_samples)

                # Loss measures generator's ability to fool the discriminator
                entropy_loss, mean_ensemble_loss = self.calculate_regular_loss(X=gen_samples, tau1=self.tau1,
                                                                               tau2=self.tau2)
                entropy_loss = torch.mean(entropy_loss)
                mean_ensemble_loss = torch.mean(mean_ensemble_loss)

                g_loss = self.lam1 * entropy_loss + self.lam2 * torch.mean(
                    mean_ensemble_loss) + self.lam3 * pull_away_loss

                g_loss.backward()
                self.optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [G loss: %f] [entr loss: %f] [mean loss: %f] [pull away loss: %f] "
                    % (epoch, self.n_epochs, i, len(dataloader), g_loss.item(),
                       entropy_loss.item(), mean_ensemble_loss.item(), pull_away_loss.item()
                       ))

                loss_gen_batch.append(g_loss.item())
                loss_gen_entropy_batch.append(entropy_loss.item())
                loss_gen_mean_ensemble_batch.append(mean_ensemble_loss.item())
                loss_gen_pull_away_batch.append(pull_away_loss.item())

            loss_gen.append(np.mean(loss_gen_batch))
            loss_gen_entropy.append(np.mean(loss_gen_entropy_batch))
            loss_gen_mean_ensemble.append(np.mean(loss_gen_mean_ensemble_batch))
            loss_gen_pull_away.append(np.mean(loss_gen_pull_away_batch))

        loss_train = {
            'loss_gen': np.array(loss_gen),
            'loss_gen_entropy': np.array(loss_gen_entropy),
            'loss_gen_mean_ensemble': np.array(loss_gen_mean_ensemble),
            'loss_gen_pull_away': np.array(loss_gen_pull_away)
        }

        return loss_train

    def sample_generate(self, num):
        z = Variable(self.Tensor(np.random.normal(0, 1, (num, self.latent_dim))))

        gen_samples = self.generator(z)

        return gen_samples


if __name__ == '__main__':
    path_detector = os.path.join(path_project, f'adversarial_ensemble_AD/models/ensemble/n=2/after_train')
    params = {
        "path_detector": path_detector
    }
    ad_g = Adversarial_Generator(params)

    # 加载数据
    X_train = np.load(os.path.join(path_project, 'data/banwuli_data/yukina_data/normal/features.npy'))
    y_train = np.load(os.path.join(path_project, 'data/banwuli_data/yukina_data/normal/labels.npy'))
    train_dataset = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=ad_g.batch_size, shuffle=True)

    ad_g.train(dataloader=train_dataloader)
    gen_samples = ad_g.sample_generate(10)

    # 保存生成的样本
    path_save = os.path.join(path_project, 'adversarial_ensemble_AD/log/gan', "samples")
