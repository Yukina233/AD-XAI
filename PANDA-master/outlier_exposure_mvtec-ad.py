import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse

from tqdm import tqdm

import utils


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


def train_model(model, train_loader, outliers_loader, test_loader, device, epochs, lr, outputdir, seed, use_OE=True):
    model.eval()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    bce = torch.nn.BCELoss()
    epoch_output = [epoch for epoch in range(1, epochs + 1)]
    aucroc_output = []
    for epoch in tqdm(range(epochs), desc='Epochs'):
        run_epoch(model, train_loader, outliers_loader, optimizer, bce, device, use_OE=use_OE)
        # print('Epoch: {}'.format(epoch + 1))
        # print('Epoch: {}, AUROC is: {}'.format(epoch + 1, auc))
        auc = get_score(model, device, test_loader)
        aucroc_output.append(auc)

    df_output = pd.DataFrame(data={'epoch': epoch_output, 'aucroc': aucroc_output})
    df_output.to_csv(os.path.join(outputdir, f'AUCROC_PANDA_epoch_seed={seed}.csv'), index=False)

    return aucroc_output[-1]

def run_epoch(model, train_loader, outliers_loader, optimizer, bce, device, use_OE=True):
    # running_loss = 0.0
    for i, (imgs, ls) in enumerate(train_loader):
        if use_OE:
            imgs = imgs.to(device)

            out_imgs, _ = next(iter(outliers_loader))

            outlier_im = out_imgs.to(device)

            optimizer.zero_grad()

            pred, _ = model(imgs)
            if pred.dim() == 0:
                pred = pred.unsqueeze(0)
            outlier_pred, _ = model(outlier_im)

            batch_1 = pred.size()[0]
            batch_2 = outlier_pred.size()[0]

            labels = torch.zeros(size=(batch_1 + batch_2,), device=device)
            labels[batch_1:] = torch.ones(size=(batch_2,))

            loss = bce(torch.sigmoid(torch.cat([pred, outlier_pred])), labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

            optimizer.step()

        else:
            imgs = imgs.to(device)
            ls = ls.float().to(device)
            optimizer.zero_grad()

            pred, _ = model(imgs)

            loss = bce(torch.sigmoid(torch.cat([pred])), ls)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

            optimizer.step()

    #     running_loss += loss.item()
    #
    # return running_loss / (i + 1)

def main_origin(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_path = os.path.join(path_project,
                             f'data/mvtec_ad_imgsize=224')
    for target_dataset_name in tqdm(utils.mvtype, desc='Total progress'):

        output_dir = os.path.join(path_project, f'PANDA-master/log/PANDA_origin_resnet{args.resnet_type}',
                                  'MVTec-AD_' + target_dataset_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        npz_file = os.path.join(data_path, 'MVTec-AD_' + target_dataset_name + '.npz')
        if os.path.exists(npz_file):
            npz_data = np.load(npz_file, allow_pickle=True)
            X_train = npz_data['X_train']
            y_train = npz_data['y_train']
            X_test = npz_data['X_test']
            y_test = npz_data['y_test']
        else:
            raise FileNotFoundError('No data found in the file or directory: ' + npz_file)

        seed_output = []
        aucroc_output = []

        for i in tqdm(range(1, args.seed + 1), desc='Different seed'):
            model = utils.get_resnet_model(resnet_type=args.resnet_type)

            # Change last layer
            model.fc = torch.nn.Linear(args.latent_dim_size, 1)

            model = model.to(device)
            utils.freeze_parameters(model, train_fc=True)

            set_seed(i)
            seed_output.append(i)
            from auxiliary_data_AD.preprocessing import MVTecDataset
            train_dataset = MVTecDataset({'X': X_train, 'y': y_train})
            test_dataset = MVTecDataset({'X': X_test, 'y': y_test})
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=2, drop_last=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                      num_workers=2, drop_last=False)
            outliers_loader = utils.get_outliers_loader(args.batch_size)

            aucroc = train_model(model, train_loader, outliers_loader, test_loader, device, args.epochs, args.lr, output_dir, i, use_OE=True)
            aucroc_output.append(aucroc)

            del model


        df_output = pd.DataFrame(data={'seed': seed_output, 'Customized': aucroc_output})
        df_output.to_csv(os.path.join(output_dir, f'AUCROC_PANDA_type(None)_noise(None)_unsupervise.csv'), index=False)

def get_score(model, device, test_loader):
    model.eval()
    anom_labels = []
    predictions = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.numpy()
            pred, _ = model(imgs)
            pred = torch.sigmoid(pred)
            if pred.dim() == 0:
                pred = pred.unsqueeze(0)
            batch_size = imgs.shape[0]
            for j in range(batch_size):
                predictions.append(pred[j].detach().cpu().numpy())
                anom_labels.append(labels[j])

    test_set_predictions = np.array(predictions)
    test_labels = np.array(anom_labels)

    auc = roc_auc_score(test_labels, test_set_predictions)

    return auc


def main_aug(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for target_dataset_name in tqdm(utils.mvtype, desc='Total progress'):

        output_dir = os.path.join(path_project,
                                  f'PANDA-master/log/n_samples_threshold={args.n_samples_threshold},imgsize={args.imagesize}',
                                  f'PANDA_{args.aug_type},lamda={args.lamda},aux_size={args.aux_size}_resnet{args.resnet_type}',
                                  'MVTec-AD_' + target_dataset_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        seed_output = []
        aucroc_output = []

        for i in tqdm(range(1, args.seed + 1), desc='Different seed'):
            model = utils.get_resnet_model(resnet_type=args.resnet_type)

            # Change last layer
            model.fc = torch.nn.Linear(args.latent_dim_size, 1)

            model = model.to(device)
            utils.freeze_parameters(model, train_fc=True)

            set_seed(i)
            seed_output.append(i)
            data_path = os.path.join(path_project,
                                     f'auxiliary_data_AD/aug_data/n_samples_threshold={args.n_samples_threshold},aug_type={args.aug_type},aux_size={args.aux_size},lamda={args.lamda},seed={i}')

            npz_file = os.path.join(data_path, 'MVTec-AD_' + target_dataset_name + '.npz')
            if os.path.exists(npz_file):
                npz_data = np.load(npz_file, allow_pickle=True)
                X_train = npz_data['X_train']
                y_train = npz_data['y_train']
                X_test = npz_data['X_test']
                y_test = npz_data['y_test']
            else:
                raise FileNotFoundError('No data found in the file or directory: ' + npz_file)

            from auxiliary_data_AD.preprocessing import MVTecDataset
            train_dataset = MVTecDataset({'X': X_train, 'y': y_train})
            test_dataset = MVTecDataset({'X': X_test, 'y': y_test})
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                       num_workers=2, drop_last=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                      num_workers=2, drop_last=False)

            aucroc = train_model(model, train_loader, None, test_loader, device, args.epochs, args.lr, output_dir, i, use_OE=False)
            aucroc_output.append(aucroc)

            del model

        df_output = pd.DataFrame(data={'seed': seed_output, 'Customized': aucroc_output})
        df_output.to_csv(os.path.join(output_dir, f'AUCROC_PANDA_type(None)_noise(None)_unsupervise.csv'), index=False)



path_project = '/home/yukina/Missile_Fault_Detection/project'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epochs', default=20, type=int, metavar='epochs', help='number of epochs')
    parser.add_argument('--label', default=0, type=int, help='The normal class')
    parser.add_argument('--lr', type=float, default=1e-1, help='The initial learning rate.')
    parser.add_argument('--resnet_type', default=50, type=int, help='which resnet to use')
    parser.add_argument('--latent_dim_size', default=2048, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--n_samples_threshold', default=0, type=int)
    parser.add_argument('--imagesize', default=224, type=int)
    parser.add_argument('--aug_type', default='cutmix', type=str)
    parser.add_argument('--lamda', default=1, type=float)
    parser.add_argument('--aux_size', default=1, type=int)
    parser.add_argument('--seed', default=2, type=int)
    args = parser.parse_args()

    # main_origin(args)
    main_aug(args)
