#! /usr/bin/env python3
path_project = '/home/yukina/Missile_Fault_Detection/project/'
path_file = path_project + 'pytorch_influence_functions-master/examples/'

import pytorch_influence_functions as ptif
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

def load_model():
    from pythae_modified.models import AutoModel

    model = AutoModel.load_from_folder(
        path_project + 'anomaly_detection/ProblemCheck/model_to_check/' +
                       'CIFAR10_VQVAE_training10_2023-10-20_14-14-37/final_model')
    model.cuda()
    print('model loaded')
    return model


def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    indices_train = torch.arange(0, 100)

    indices_test = torch.arange(0, 1)

    trainset = torchvision.datasets.CIFAR10(root=path_project + 'anomaly_detection/ProblemCheck/data', train=True,
                                            download=True, transform=transform)
    trainset = Subset(trainset, indices_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=path_project + 'anomaly_detection/ProblemCheck/data', train=False,
                                           download=True, transform=transform)
    testset = Subset(testset, indices_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader


if __name__ == "__main__":
    config = {
        'outdir': path_file + 'outdir',
        'seed': 42,
        'gpu': 0,
        'dataset': 'CIFAR10',
        'num_classes': 10,
        'test_sample_num': False,
        'test_start_index': 0,
        'recursion_depth': 5,
        'r_averaging': 10,
        'scale': None,
        'damp': None,
        'calc_method': 'img_wise',
        'log_filename': None,
    }
    model = load_model()
    trainloader, testloader = load_data()
    ptif.init_logging('logfile.log')
    ptif.calc_img_wise(config, model, trainloader, testloader)
