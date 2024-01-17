import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.nn import AdaptiveAvgPool2d
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
    def __init__(self, data):
        self.X = data['X']
        self.y = data['y']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def embedding_data(encoder_name='resnet18', layers=['avgpool'], interpolate=False, batch_size=4, X=None, y=None):
    outputs = {}
    encoder = torch.hub.load('pytorch/vision:v0.10.0', encoder_name, pretrained=True)
    # Create dataloader
    data = {
        'X': X,
        'y': y
    }
    dataset = MVTecDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=3, drop_last=False)
    def get_embedding(layer_name):
        def hook(module, input, output):
            outputs[str(layer_name)] = output.detach()

        return hook

    # Register the hook on the first and second layer
    for layer in layers:
        if layer == 'avgpool':
            encoder.avgpool.register_forward_hook(get_embedding(layer))
        else:
            encoder.__dict__["_modules"][layer][-1].register_forward_hook(get_embedding(layer))

    encoder = nn.Sequential(*list(encoder.children())[:-1])
    encoder.to(device)
    avgpool = AdaptiveAvgPool2d(output_size=(1, 1))
    avgpool.to(device)

    # encoder for extracting embedding
    encoder.eval()

    def generate_embedding(data_loader):
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                outputs.clear()
                X, y = data
                X = X.to(device)

                encoder(X)
                if interpolate:
                    print(embedding.hhh)
                    shallow_embed_size = [outputs[layer] for layer in layers][0].shape
                    embedding = torch.cat([
                        F.interpolate(outputs[layer],
                                      size=(shallow_embed_size[-2], shallow_embed_size[-1]),
                                      mode="bilinear",
                                      align_corners=False) for layer in layers], dim=1)
                    embedding = avgpool(embedding)
                else:
                    embedding = torch.cat([avgpool(outputs[layer]) for layer in layers], dim=1)
                embedding_list.append(embedding.squeeze().cpu().numpy())
                label_list.append(y.numpy())
        return np.vstack(embedding_list), np.hstack(label_list)

    return generate_embedding(data_loader)


def embedding_datasets(encoder_name='resnet18', layers=['avgpool'], interpolate=False, batch_size=4, dataset_name=None,
                       plot=True,
                       path_datasets=None, output_path=None):
    outputs = {}
    encoder = torch.hub.load('pytorch/vision:v0.10.0', encoder_name, pretrained=True)

    def get_embedding(layer_name):
        def hook(module, input, output):
            outputs[str(layer_name)] = output.detach()

        return hook

    # Register the hook on the first and second layer
    for layer in layers:
        if layer == 'avgpool':
            encoder.avgpool.register_forward_hook(get_embedding(layer))
        else:
            encoder.__dict__["_modules"][layer][-1].register_forward_hook(get_embedding(layer))

    encoder = nn.Sequential(*list(encoder.children())[:-1])
    encoder.to(device)
    avgpool = AdaptiveAvgPool2d(output_size=(1, 1))
    avgpool.to(device)

    fig = plt.figure(figsize=(14, 8))

    # 加载所有的数据集
    n = 0
    for filename in tqdm(os.listdir(path_datasets)):
        data_name = filename.split('.')[0]
        npz_file = os.path.join(path_datasets, filename)
        data_all = np.load(npz_file, allow_pickle=True)
        data_train = {
            'X': data_all['X_train'],
            'y': data_all['y_train']
        }
        data_test = {
            'X': data_all['X_test'],
            'y': data_all['y_test']
        }
        dataset = MVTecDataset(data_all)
        dataset_train = MVTecDataset(data_train)
        dataset_test = MVTecDataset(data_test)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=3, drop_last=False)
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=3, drop_last=False)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=3, drop_last=False)

        # encoder for extracting embedding
        encoder.eval()

        def generate_embedding(data_loader):
            embedding_list = []
            label_list = []
            with torch.no_grad():
                for i, data in enumerate(data_loader):
                    outputs.clear()
                    X, y = data
                    X = X.to(device)

                    encoder(X)
                    if interpolate:
                        shallow_embed_size = [outputs[layer] for layer in layers][0].shape
                        embedding = torch.cat([
                            F.interpolate(outputs[layer],
                                          size=(shallow_embed_size[-2], shallow_embed_size[-1]),
                                          mode="bilinear",
                                          align_corners=False) for layer in layers], dim=1)
                        embedding = avgpool(embedding)
                    else:
                        embedding = torch.cat([avgpool(outputs[layer]) for layer in layers], dim=1)
                    embedding_list.append(embedding.squeeze().cpu().numpy())
                    label_list.append(y.numpy())
            return np.vstack(embedding_list), np.hstack(label_list)

        X, y = generate_embedding(data_loader)
        X_train, y_train = generate_embedding(train_loader)
        X_test, y_test = generate_embedding(test_loader)

        print(
            f'Class: {data_name}, Samples: {len(y)}, Anomalies: {sum(y)}, Anomaly Ratio(%): {round(sum(y) / len(y) * 100, 2)}')
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        np.savez_compressed(os.path.join(output_path, data_name + '.npz'), X=X, y=y, X_train=X_train, y_train=y_train,
                            X_test=X_test, y_test=y_test)

        if plot:
            fig.add_subplot(3, 5, n + 1)
            # visualization
            X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X[:1000])
            y_tsne = np.array(y[:1000])

            plt.scatter(X_tsne[y_tsne == 0, 0], X_tsne[y_tsne == 0, 1], color='blue', edgecolors='k', alpha=0.8)
            plt.scatter(X_tsne[y_tsne == 1, 0], X_tsne[y_tsne == 1, 1], color='red', edgecolors='k', alpha=0.8)
            plt.title(f'Category: {data_name}')

        n += 1
    if plot:
        plt.subplots_adjust(wspace=0.4, hspace=0.5)
        plt.suptitle(f'Dataset: {dataset_name}', y=0.98, fontsize=16)
        plt.savefig(os.path.join(output_path, f'TSNE.png'))


if __name__ == '__main__':
    path_project = '/home/yukina/Missile_Fault_Detection/project'
    set_seed(42)
    encoder_name = 'resnet152'
    layers = ['layer1', 'layer3']
    imgsize = 224
    interpolate = False
    path_datasets = path_project + f'/data/mvtec_ad_imgsize={imgsize}'

    if layers == ['avgpool']:
        layers_name = ''
    else:
        layers_name = layers.__str__().replace('[', '').replace(']', '').replace(' ', '').replace('\'', '')

    if interpolate:
        output_path = path_project + f'/data/interpolate/mvtec_ad_preprocessed_{encoder_name}_{layers_name}_imgsize={imgsize}'
    else:
        output_path = path_project + f'/data/mvtec_ad_preprocessed_{encoder_name}_{layers_name}_imgsize={imgsize}'

    embedding_datasets(encoder_name=encoder_name,
                       layers=layers,
                       interpolate=interpolate,
                       path_datasets=path_datasets,
                       output_path=output_path)
