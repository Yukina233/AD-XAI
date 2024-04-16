import argparse
import pickle

import torch
from tqdm import tqdm

from adbench_modified.baseline.DeepSAD.src.run import DeepSAD

from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import time
import glob

from adversarial_ensemble_AD.data_generate.gan import Adversarial_Generator

# logging.basicConfig(level=logging.INFO)

# 设置项目路径
path_project = '/home/yukina/Missile_Fault_Detection/project'

print("All down!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--n_epochs", type=int, default=5, help="number of epochs of overall training")
    parser.add_argument("--path_train_data", type=str,
                        default=os.path.join(path_project, 'data/banwuli_data/yukina_data/train_seperate'))
    parser.add_argument("--dir_model", type=str,
                        default=os.path.join(path_project, f'adversarial_ensemble_AD/models/ensemble'))
    parser.add_argument("--path_output", type=str,
                        default=os.path.join(path_project, f'adversarial_ensemble_AD/log/train_result'))
    parser.add_argument("--DeepSAD_config", type=dict, default={
        "n_epochs": 20,
        "ae_n_epochs": 20
    }, help="config of DeepSAD")
    parser.add_argument("--GAN_config", type=dict, default={
        "n_epochs": 100,
        "lam": 0.5,
        "tau": 1
    }, help="config of GAN")

    config = parser.parse_args()

    # 生成特定参数的文件夹
    param_dir = f'K=2,gan_epoch={config.GAN_config["n_epochs"]},lam={config.GAN_config["lam"]},tau={config.GAN_config["tau"]}'
    config.dir_model = os.path.join(config.dir_model, param_dir)
    config.path_output = os.path.join(config.path_output, param_dir)

    path_data_init = os.path.join(config.path_train_data, 'init')
    X_train_init = None
    y_train_init = None
    for train_dataset in os.listdir(path_data_init):
        data = np.load(os.path.join(path_data_init, train_dataset))
        if X_train_init is None:
            X_train_init = data['X_train']
            y_train_init = data['y_train']
        else:
            X_train_init = np.concatenate((X_train_init, data['X_train']))
            y_train_init = np.concatenate((y_train_init, data['y_train']))


    model_list = []

    for iteration in tqdm(range(0, config.n_epochs), desc='Main epochs'):
        if iteration == 0:
            path_train = os.path.join(config.path_train_data, 'init')
        else:
            path_train = os.path.join(config.path_train_data, 'augment', param_dir, f'{iteration - 1}')
        # 遍历所有数据集文件，分别训练各子模型
        for train_dataset in tqdm(os.listdir(path_train), desc='Seperate model train'):
            base_name = os.path.basename(train_dataset).replace('.npz', '')

            # 加载数据集
            data = np.load(os.path.join(path_train, train_dataset))
            X_train = data['X_train']
            y_train = data['y_train']

            dir_model_save = os.path.join(config.dir_model, f'{iteration}')
            os.makedirs(dir_model_save, exist_ok=True)
            # 初始迭代
            if iteration == 0:
                # 创建结果文件夹
                path_model_save = os.path.join(dir_model_save, f'DeepSAD-{base_name}.pth')
                # 实例化DeepSAD并存到model_list中
                model = DeepSAD(seed=config.seed, config=config.DeepSAD_config)
                model.fit(X_train=X_train, y_train=y_train)

                model.deepSAD.save_model(export_model=path_model_save, save_ae=True)
            else:

                path_model_save = os.path.join(dir_model_save, f'DeepSAD-{base_name}.pth')
                # 加载模型
                path_model_load = os.path.join(config.dir_model, f'{iteration - 1}/DeepSAD-{base_name}.pth')
                model = DeepSAD(seed=config.seed, load_model=path_model_load, config=config.DeepSAD_config)

                # 训练模型
                model.fit_encoder(X_train=X_train, y_train=y_train)
                model.deepSAD.save_model(export_model=path_model_save, save_ae=True)

        # 训练对抗样本生成器
        print("Train Adversarial Generator")
        path_detector = os.path.join(config.dir_model, f'{iteration}')
        config.GAN_config["path_detector"] = path_detector
        ad_g = Adversarial_Generator(config=config.GAN_config)

        train_dataset_GAN = torch.utils.data.TensorDataset(torch.Tensor(X_train_init), torch.Tensor(y_train_init))
        train_dataloader_GAN = torch.utils.data.DataLoader(train_dataset_GAN, batch_size=ad_g.batch_size, shuffle=True)

        loss_train = ad_g.train(dataloader=train_dataloader_GAN)

        # 将字典存储到文件中
        path_train_result_save = os.path.join(config.path_output, 'loss')
        os.makedirs(path_train_result_save, exist_ok=True)
        with open(os.path.join(path_train_result_save, f'{iteration}.pkl'), 'wb') as file:
            pickle.dump(loss_train, file)

        # 构造新的训练数据集
        for train_dataset in tqdm(os.listdir(path_data_init), desc='Create new train data'):
            data = np.load(os.path.join(path_data_init, train_dataset))
            X_train = data['X_train']
            y_train = data['y_train']

            num_generate = y_train.shape[0]

            gen_samples = ad_g.sample_generate(num=num_generate)

            X_train_new = np.concatenate((X_train, np.array(gen_samples.detach().cpu()).squeeze(1)))
            y_train_new = np.concatenate((y_train, np.ones(num_generate)))

            # 保存新的训练数据集
            path_train_new = os.path.join(config.path_train_data, 'augment', param_dir, f'{iteration}')
            os.makedirs(path_train_new, exist_ok=True)
            np.savez(os.path.join(path_train_new, train_dataset), X_train=X_train_new, y_train=y_train_new)

        del ad_g
