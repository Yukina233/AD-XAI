import argparse
import gc
import pickle
import shutil
import pandas as pd
import torch
from tqdm import tqdm

from MNIST_dataset.show_samples import show_samples
from MNIST_ensemble_DeepSAD_test import ensemble_test
from adbench_modified.baseline.DeepSAD.src.run import DeepSAD

from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import time
import glob

from adversarial_ensemble_AD.data_generate.gan_mnist import Adversarial_Generator

# from GHL_dataset.data_generate.wgan_gp import Adversarial_Generator

# logging.basicConfig(level=logging.INFO)

# 设置项目路径
path_project = '/media/test/d/Yukina/AD-XAI_data'

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    window = 1
    step = 1
    time_start = time.time()
    for seed in range(1):
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=seed, help="seed")
        parser.add_argument("--K", type=int, default=7, help="number of sub-models")
        parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of overall training")
        parser.add_argument("--train_set_name", type=str, default='MNIST_nonorm_1')
        parser.add_argument("--DeepSAD_config", type=dict, default={
            "n_epochs": 1,
            "ae_n_epochs": 10,
            "net_name": 'Dense'
        }, help="config of DeepSAD")
        parser.add_argument("--GAN_config", type=dict, default={
            "seed": seed,
            "latent_dim": 96,
            "lr": 0.003,
            "clip_value": 0.01,
            "lambda_gp": 10000,
            "n_epochs": 1,
            "pre_epochs": 0,
            "lam1": 0,
            "lam2": 5,
            "lam3": 0,
            "tau1": 1,
            "img_size": 784
        }, help="config of GAN")

        config = parser.parse_args()

        # 生成特定参数的文件夹
        train_set_name = config.train_set_name
        config.path_train_data = os.path.join(path_project,

                                              f'data/{train_set_name}/yukina_data/ensemble_data, window={window}, step={step}')
        config.dir_model = os.path.join(path_project,
                                                 f'{train_set_name}_dataset/models/{train_set_name}/ensemble')
        config.path_output = os.path.join(path_project,
                                                 f'{train_set_name}_dataset/log/{train_set_name}/train_result')
        param_dir = f'GAN_web_w_decay, euc, window={window}, step={step}, K={config.K},deepsad_ae_epoch={config.DeepSAD_config["ae_n_epochs"]},gan_epoch={config.GAN_config["n_epochs"]},pre_epochs={config.GAN_config["pre_epochs"]},lam1={config.GAN_config["lam1"]},lam2={config.GAN_config["lam2"]},latent_dim={config.GAN_config["latent_dim"]},lr={config.GAN_config["lr"]},seed={config.seed}'
        config.dir_model = os.path.join(config.dir_model, param_dir)
        config.path_output = os.path.join(config.path_output, param_dir)

        path_data_init = os.path.join(config.path_train_data, 'init', f'K={config.K}')

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
            path_train = os.path.join(config.path_train_data, 'init', f'K={config.K}')
            # 遍历所有数据集文件，分别训练各子模型
            for train_dataset in tqdm(os.listdir(path_train), desc='Seperate model train'):
                base_name = os.path.basename(train_dataset).replace('.npz', '')

                config.DeepSAD_config["loss_output_path"] = os.path.join(config.path_output, 'deepsad_loss',
                                                                         f'iteration={iteration}',
                                                                         f'model_id={base_name}')
                os.makedirs(config.DeepSAD_config["loss_output_path"], exist_ok=True)

                # 加载数据集F
                data = np.load(os.path.join(path_train, train_dataset))

                if iteration == 0:
                    X_train = data['X_train']
                    y_train = data['y_train']
                else:
                    # 加载生成模型
                    generator = Adversarial_Generator(config=config.GAN_config)
                    generator.load_model(os.path.join(config.dir_model, f'{iteration - 1}'))
                    num_generate = data['y_train'].shape[0]
                    gen_samples = generator.sample_generate(num=num_generate)

                    X_train = np.concatenate((data['X_train'], np.array(gen_samples.detach().cpu()).squeeze(1)))
                    y_train = np.concatenate((data['y_train'], np.ones(num_generate)))

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

                del model

            # 跳过最后一次迭代
            if iteration == config.n_epochs - 1:
                break
            # 训练对抗样本生成器
            print("Train Adversarial Generator")
            path_detector = os.path.join(config.dir_model, f'{iteration}')
            config.GAN_config["path_detector"] = path_detector
            ad_g = Adversarial_Generator(config=config.GAN_config)

            if iteration != 0:
                ad_g.load_model(os.path.join(config.dir_model, f'{iteration - 1}'))
            train_dataset_GAN = torch.utils.data.TensorDataset(torch.Tensor(X_train_init), torch.Tensor(y_train_init))
            train_dataloader_GAN = torch.utils.data.DataLoader(train_dataset_GAN, batch_size=ad_g.batch_size,
                                                               shuffle=True)

            if iteration == 0:
                ad_g.pretrain(dataloader=train_dataloader_GAN)
            loss_train = ad_g.train(dataloader=train_dataloader_GAN)
            ad_g.save_model(os.path.join(config.dir_model, f'{iteration}'))
            # loss_train = ad_g.train_no_GAN(dataloader=train_dataloader_GAN)

            # 将字典存储到文件中
            path_train_result_save = os.path.join(config.path_output, 'loss')
            os.makedirs(path_train_result_save, exist_ok=True)
            # 将字典转换为 DataFrame
            df = pd.DataFrame(loss_train)
            # 导出为 CSV 文件
            df.to_csv(os.path.join(path_train_result_save, f'{iteration}.csv'), index=False)

            del ad_g
            del loss_train
            del train_dataset_GAN
            del train_dataloader_GAN

            gc.collect()

        # set the plot epoch
        if config.GAN_config["lam1"] == 0 and config.GAN_config["lam2"] == 0:
            e = 47
        else:
            e = 5
        show_samples(dataset_name=train_set_name, param_dir=param_dir, y=int(train_set_name.split('_')[-1]), epoch=e, GAN_config=config.GAN_config)
        print(f'{param_dir}')
        # ensemble_test(param_dir, config.K)

    # base_dir = os.path.join(path_project, f'{train_set_name}_dataset/log/{train_set_name}/ensemble/DeepSAD')
    # prefix = param_dir.split('seed')[0] + 'seed'
    #
    # from GHL_dataset.scripts.combine_seed_result import combine_seed_results
    # from GHL_dataset.scripts.combine_epoch_result import combine_epoch_results
    # # 调用函数
    # combine_seed_results(base_dir, prefix)
    #
    # seed_dir = os.path.join(base_dir, 'seed_group', prefix)
    # combine_epoch_results(seed_dir)
    #
    # time_end = time.time()
    # print(f"Time total cost: {time_end - time_start}")
