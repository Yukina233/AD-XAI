import argparse
import gc
import pickle
import shutil
import pandas as pd
import torch
from tqdm import tqdm

from SMD_ensemble_DeepSAD_test_new import ensemble_test
from scripts.combine_seed_result import combine_seed_results
from scripts.auto_select_epoch import auto_select_epoch
from GHL_dataset.scripts.combine_epoch_result import combine_epoch_results
from adbench_modified.baseline.DeepSAD.src.run import DeepSAD

from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import time
import glob

# from adversarial_ensemble_AD.data_generate.gan import Adversarial_Generator
from GHL_dataset.data_generate.wgan_gp import Adversarial_Generator

# logging.basicConfig(level=logging.INFO)

# 设置项目路径
path_project = '/home/yukina/Missile_Fault_Detection/project'

def merge_dict(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1:
            if isinstance(value, np.ndarray):
                dict1[key] = np.concatenate((dict1[key], value))
            elif isinstance(value, list):
                dict1[key].extend(value)
        else:
            if isinstance(value, np.ndarray):
                dict1[key] = value
            elif isinstance(value, list):
                dict1[key] = value.copy()

def group_loss(loss_dir, iterations):

    for i in range(0, iterations):
        loss_i = dict()
        loss_i['machine'] = []
        for machine_dir in glob.glob(os.path.join(loss_dir, 'machine-*')):
            loss_machine_i = dict()
            path_loss = os.path.join(machine_dir, 'loss', 'loss.csv')
            loss = pd.read_csv(path_loss)
            # 将第i次迭代的loss存储到loss_i中
            loss_machine_i['machine'] = [machine_dir.split('/')[-1].split('machine-')[-1]]
            for key, value in loss.items():
                    loss_machine_i[key] = np.array([value[i]])
            merge_dict(loss_i, loss_machine_i)

        loss_i['machine'].append('mean')
        for key in loss_i.keys():
            if key == 'machine':
                continue
            loss_i[key] = np.concatenate((loss_i[key], np.array([np.mean(loss_i[key])])))
        df = pd.DataFrame(loss_i)
        path_loss_all = os.path.join(loss_dir, 'loss_i', f'iteration={i}.csv')
        os.makedirs(os.path.dirname(path_loss_all), exist_ok=True)
        df.to_csv(path_loss_all, index=False)



if __name__ == '__main__':
    train_set_name = 'SMD_group4'
    for seed in range(3):
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=seed, help="seed")
        parser.add_argument("--K", type=int, default=7, help="number of sub-models")
        parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs of overall training")
        parser.add_argument("--path_train_data", type=str,
                            default=os.path.join(path_project,

                                                 f'data/{train_set_name}/yukina_data/ensemble_data, window=100, step=10'))
        parser.add_argument("--dir_model", type=str,
                            default=os.path.join(path_project,
                                                 f'{train_set_name}_dataset/models/{train_set_name}/ensemble'))
        parser.add_argument("--path_output", type=str,
                            default=os.path.join(path_project,
                                                 f'{train_set_name}_dataset/log/{train_set_name}/train_result'))
        parser.add_argument("--DeepSAD_config", type=dict, default={
            "n_epochs": 5,
            "ae_n_epochs": 20,
            "net_name": 'Dense'
        }, help="config of DeepSAD")
        parser.add_argument("--GAN_config", type=dict, default={
            "seed": seed,
            "latent_dim": 180,
            "lr": 0.002,
            "clip_value": 0.01,
            "lambda_gp": 1000,
            "n_epochs": 1,
            "lam1": 10,
            "lam2": 1,
            "alpha": 1,
            "lam3": 0,
            "tau1": 1,
            "img_size": 180
        }, help="config of GAN")

        config = parser.parse_args()

        # 生成特定参数的文件夹
        param_dir = f'{config.DeepSAD_config["net_name"]}, WGAN-GP, euc, window=100, step=10, no_tau2_K={config.K},deepsad_epoch={config.DeepSAD_config["n_epochs"]},ae_epoch={config.DeepSAD_config["ae_n_epochs"]},gan_epoch={config.GAN_config["n_epochs"]},lam1={config.GAN_config["lam1"]},lam2={config.GAN_config["lam2"]},alpha={config.GAN_config["alpha"]},latent_dim={config.GAN_config["latent_dim"]},lr={config.GAN_config["lr"]},clip_value={config.GAN_config["clip_value"]},lambda_gp={config.GAN_config["lambda_gp"]},seed={config.seed}'
        config.dir_model = os.path.join(config.dir_model, param_dir)
        config.path_output = os.path.join(config.path_output, param_dir)

        path_data_init = os.path.join(config.path_train_data, 'init', f'K={config.K}')

        for machine in os.listdir(path_data_init):
            path_machine = os.path.join(path_data_init, machine)

            X_train_init = None
            y_train_init = None
            for train_dataset in os.listdir(path_machine):
                data = np.load(os.path.join(path_machine, train_dataset))
                if X_train_init is None:
                    X_train_init = data['X_train']
                    y_train_init = data['y_train']
                else:
                    X_train_init = np.concatenate((X_train_init, data['X_train']))
                    y_train_init = np.concatenate((y_train_init, data['y_train']))

            model_list = []
            loss_train_all = dict()
            for iteration in tqdm(range(0, config.n_epochs), desc='Main epochs'):
                path_train = path_machine
                # 遍历所有数据集文件，分别训练各子模型
                for train_dataset in tqdm(os.listdir(path_train), desc='Seperate model train'):
                    base_name = os.path.basename(train_dataset).replace('.npz', '')

                    config.DeepSAD_config["loss_output_path"] = os.path.join(config.path_output, f'{machine}',
                                                                             'deepsad_loss',
                                                                             f'iteration={iteration}',
                                                                             f'model_id={base_name}')
                    os.makedirs(config.DeepSAD_config["loss_output_path"], exist_ok=True)

                    # 加载数据集
                    data = np.load(os.path.join(path_train, train_dataset))

                    if iteration == 0:
                        X_train = data['X_train']
                        y_train = data['y_train']
                    else:
                        # 加载生成模型
                        generator = Adversarial_Generator(config=config.GAN_config, DeepSAD_config=config.DeepSAD_config)
                        generator.load_model(os.path.join(config.dir_model, f'{machine}', f'{iteration - 1}'))
                        num_generate = int(data['y_train'].shape[0] * config.GAN_config["alpha"])
                        gen_samples = generator.sample_generate(num=num_generate)

                        X_train = np.concatenate((data['X_train'], np.array(gen_samples.detach().cpu()).squeeze(1)))
                        y_train = np.concatenate((data['y_train'], np.ones(num_generate)))

                    dir_model_save = os.path.join(config.dir_model, f'{machine}', f'{iteration}')
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
                        path_model_load = os.path.join(config.dir_model, f'{machine}',
                                                       f'{iteration - 1}/DeepSAD-{base_name}.pth')
                        model = DeepSAD(seed=config.seed, load_model=path_model_load, config=config.DeepSAD_config)

                        # 训练模型
                        model.fit_encoder(X_train=X_train, y_train=y_train)
                        model.deepSAD.save_model(export_model=path_model_save, save_ae=True)

                    del model

                # 跳过最后一次迭代
                if iteration == config.n_epochs - 1:
                    break
                # 训练对抗样本生成器
                print("Train Adversarial Generator")
                path_detector = os.path.join(config.dir_model, f'{machine}', f'{iteration}')
                config.GAN_config["path_detector"] = path_detector
                ad_g = Adversarial_Generator(config=config.GAN_config, DeepSAD_config=config.DeepSAD_config)

                if iteration != 0:
                    ad_g.load_model(os.path.join(config.dir_model, f'{machine}', f'{iteration - 1}'))
                train_dataset_GAN = torch.utils.data.TensorDataset(torch.Tensor(X_train_init), torch.Tensor(y_train_init))
                train_dataloader_GAN = torch.utils.data.DataLoader(train_dataset_GAN, batch_size=ad_g.batch_size,
                                                                   shuffle=True, drop_last=True)
                loss_train = {
                    'iteration': np.array([iteration]),
                }
                loss_train.update(ad_g.train(dataloader=train_dataloader_GAN))
                ad_g.save_model(os.path.join(config.dir_model, f'{machine}', f'{iteration}'))
                # loss_train = ad_g.train_no_GAN(dataloader=train_dataloader_GAN)

                merge_dict(loss_train_all, loss_train)

                del ad_g
                del loss_train
                del train_dataset_GAN
                del train_dataloader_GAN
            # 将字典存储到文件中

            path_train_result_save = os.path.join(config.path_output, f'{machine}', 'loss')
            os.makedirs(path_train_result_save, exist_ok=True)
            # 将字典转换为 DataFrame
            df = pd.DataFrame(loss_train_all)
            # 导出为 CSV 文件
            df.to_csv(os.path.join(path_train_result_save, f'loss.csv'), index=False)

            del loss_train_all
            gc.collect()

        group_loss(config.path_output, config.n_epochs - 1)
        ensemble_test(param_dir, seed, DeepSAD_config=config.DeepSAD_config)

    base_dir = os.path.join(path_project, f'{train_set_name}_dataset/log/{train_set_name}/ensemble/DeepSAD')
    prefix = param_dir.split('seed')[0] + 'seed'
    # 调用函数
    combine_seed_results(base_dir, prefix)

    seed_dir = os.path.join(base_dir, 'seed_group', prefix)
    combine_epoch_results(seed_dir)
    auto_select_epoch(seed_dir)

