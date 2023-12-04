import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import random
from random import shuffle
import os
from itertools import product
import json

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import datasets,transforms
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Subset

from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import logging

os.chdir('/content/drive/MyDrive/PhD Research/ADBench')
from mvtec_ad import MVTecAD

if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    print(f'number of gpu: {n_gpu}')
    print(f'cuda name: {torch.cuda.get_device_name(0)}')
    print('GPU is on')
else:
    print('GPU is off')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
    # os.environ['TF_DETERMINISTIC_OPS'] = 'true'

    #basic seed
    np.random.seed(seed)
    random.seed(seed)

    #pytorch seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def imshow(img):
  img = img / 2 + 0.5   # unnormalize
  npimg = img.numpy()   # convert from tensor

  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()


def embedding_MVTec_AD(img_size=32, encoder=None, batch_size=16, subsample=10000, dataset_name=None, plot=True):
    assert dataset_name == 'MVTec-AD'

    transformation = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])

    def _convert_label(x):
        '''
        convert anomaly label. 0: normal; 1: anomaly.
        :param x (int): class label
        :return: 0 or 1
        '''
        return 0 if x == 0 else 1

    # define target transforms
    target_transform = transforms.Lambda(_convert_label)

    # class list only for the MVTec-AD dataset
    class_list = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
                  'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

    fig = plt.figure(figsize=(14, 8))
    for n, c in enumerate(class_list):
        # load data (notice that the MVTec-AD training set exists only normal samples)
        data_train = MVTecAD('data',
                             subset_name=c,
                             train=True,
                             transform=transformation,
                             mask_transform=transformation,
                             target_transform=target_transform,
                             download=True)

        data_test = MVTecAD('data',
                            subset_name=c,
                            train=False,
                            transform=transformation,
                            mask_transform=transformation,
                            target_transform=target_transform,
                            download=True)

        train_tensor = []
        train_tensor.extend([(_[0], _[-1]) for _ in data_train])  # drop the mask data
        train_tensor.extend([(_[0], _[-1]) for _ in data_test])

        # shuffle
        shuffle(train_tensor)

        train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=False, num_workers=3, drop_last=False)

        # encoder for extracting embedding
        encoder.eval()

        embeddings = []
        labels = []
        with torch.no_grad():
            for i, data in enumerate(tqdm(train_loader)):
                X, y = data
                X = X.to(device)

                embeddings.append(encoder(X).squeeze().cpu().numpy())
                labels.append(y.numpy())

        X = np.vstack(embeddings)
        y = np.hstack(labels)

        print(
            f'Class: {c}, Samples: {len(y)}, Anomalies: {sum(y)}, Anomaly Ratio(%): {round(sum(y) / len(y) * 100, 2)}')
        np.savez_compressed(os.path.join('/content/drive/MyDrive/PhD Research/ADBench/datasets_NLP_CV',
                                         dataset_name + '_' + c + '.npz'), X=X, y=y)

        if plot:
            fig.add_subplot(3, 5, n + 1)
            # visualization
            X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X[:1000])
            y_tsne = np.array(y[:1000])

            plt.scatter(X_tsne[y_tsne == 0, 0], X_tsne[y_tsne == 0, 1], color='blue', edgecolors='k', alpha=0.8)
            plt.scatter(X_tsne[y_tsne == 1, 0], X_tsne[y_tsne == 1, 1], color='red', edgecolors='k', alpha=0.8)
            plt.title(f'Category: {c}')

    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    plt.suptitle(f'Dataset: {dataset_name}', y=0.98, fontsize=16)
    plt.show()






set_seed(42)
encoder_name = 'ResNet-18'

if encoder_name == 'ResNet-18':
  img_size = 32

  # resnet18 pretrained on the ImageNet (embedding dimension: 512)
  encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
  encoder = nn.Sequential(*list(encoder.children())[:-1])
  encoder.to(device)

# elif encoder_name == 'ViT':
#   img_size = 384
#
#   # Load ViT
#   from pytorch_pretrained_vit import ViT
#   encoder = ViT('B_16_imagenet1k', pretrained=True)
#   encoder.to(device)

else:
  raise NotImplementedError


dataset_name = 'MVTec-AD'
# for dataset_name in ['MNIST-C', 'FashionMNIST', 'CIFAR10', 'SVHN', 'MVTec-AD']:
embedding_MVTec_AD(img_size=img_size, encoder=encoder, dataset_name=dataset_name)
