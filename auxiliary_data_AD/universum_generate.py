import random

import numpy as np
import torch
from matplotlib import pyplot as plt


def rand_bbox(size, lam):
    '''Getting the random box in CutMix'''
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_universum(image1, image2, aug_type, lamda):
    if_plot = False
    if image1.shape != image2.shape:
        if image2.shape[0] == 1:
            image2 = image2.repeat(3, axis=0)
        elif image2.shape[0] == 3:
            image2 = 0.299 * image2[0, :, :] + 0.587 * image2[1, :, :] + 0.114 * image2[2, :, :]
            image2 = np.expand_dims(image2, axis=0)
    """Calculating Mixup-induced universum from a batch of images"""
    universum = image1.copy()
    if aug_type == 'mixup':
        # Using Mixup
        universum = lamda * universum + (1 - lamda) * image2
    else:
        # Using CutMix
        lam = 0
        while lam < lamda - 0.05 or lam > lamda + 0.05:
            # Since it is hard to control the value of lambda in CutMix,
            # we accept lambda in [lambda-0.05, lambda+0.05].
            bbx1, bby1, bbx2, bby2 = rand_bbox(image2.shape, lamda)
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image2.shape[-1] * image2.shape[-2]))
        universum[:, bbx1:bbx2, bby1:bby2] = image2[:, bbx1:bbx2, bby1:bby2]

    # for debug
    if if_plot:
        plt.imshow(image1.transpose(1, 2, 0), cmap='gray')
        plt.savefig('/home/yukina/Missile_Fault_Detection/project/auxiliary_data_AD/log/example/image1.png')
        plt.imshow(image2.transpose(1, 2, 0), cmap='gray')
        plt.savefig('/home/yukina/Missile_Fault_Detection/project/auxiliary_data_AD/log/example/image2.png')
        plt.imshow(universum.transpose(1, 2, 0), cmap='gray')
        plt.savefig('/home/yukina/Missile_Fault_Detection/project/auxiliary_data_AD/log/example/universum.png')
    return universum
