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
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Subset

from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import logging

if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    print(f'number of gpu: {n_gpu}')
    print(f'cuda name: {torch.cuda.get_device_name(0)}')
    print('GPU is on')
else:
    print('GPU is off')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
    # os.environ['TF_DETERMINISTIC_OPS'] = 'true'

    # basic seed
    np.random.seed(seed)
    random.seed(seed)

    # pytorch seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()  # convert from tensor

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


import os
import numpy as np
from torch.utils.data import DataLoader, Dataset


class MVTecDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file, allow_pickle=True)
        self.X = data['X']
        self.y = data['y']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def embedding_MVTec_AD(encoder=None, batch_size=16, dataset_name=None, plot=True,
                       path_datasets=None, output_path=None):
    fig = plt.figure(figsize=(14, 8))
    # 加载所有的数据集
    n = 0
    for filename in os.listdir(path_datasets):
        data_name = filename.split('.')[0]
        dataset = MVTecDataset(npz_file=os.path.join(path_datasets, filename))
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=3, drop_last=False)

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
            f'Class: {data_name}, Samples: {len(y)}, Anomalies: {sum(y)}, Anomaly Ratio(%): {round(sum(y) / len(y) * 100, 2)}')
        np.savez_compressed(os.path.join(output_path, data_name + '.npz'), X=X, y=y)

        if plot:
            fig.add_subplot(3, 5, n + 1)
            # visualization
            X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X[:1000])
            y_tsne = np.array(y[:1000])

            plt.scatter(X_tsne[y_tsne == 0, 0], X_tsne[y_tsne == 0, 1], color='blue', edgecolors='k', alpha=0.8)
            plt.scatter(X_tsne[y_tsne == 1, 0], X_tsne[y_tsne == 1, 1], color='red', edgecolors='k', alpha=0.8)
            plt.title(f'Category: {data_name}')

        n += 1

    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    plt.suptitle(f'Dataset: {dataset_name}', y=0.98, fontsize=16)
    plt.show()


path_project = '/home/yukina/Missile_Fault_Detection/project'
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

embedding_MVTec_AD(encoder=encoder,
                   path_datasets=path_project + '/data/mvtec_ad',
                   output_path=path_project + '/data/mvtec_ad_preprocess')
