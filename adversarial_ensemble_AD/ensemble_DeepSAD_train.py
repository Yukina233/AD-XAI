import argparse


from tqdm import tqdm


from adbench_modified.baseline.DeepSAD.src.run import DeepSAD

import os
import numpy as np
import time
import glob

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
    parser.add_argument("--DeepSAD_config", type=dict, default={
        "n_epochs": 0,
        "ae_n_epochs": 0
    }, help="config of DeepSAD")
    parser.add_argument("--GAN_config", type=dict, default={
        "n_epochs": 20
    }, help="config of GAN")

    config = parser.parse_args()


    model_list = []
    for iteration in tqdm(range(0, config.n_epochs), desc='Main epochs'):
        if iteration == 0:
            path_train = os.path.join(config.path_train_data, 'init')
        else:
            path_train = os.path.join(config.path_train_data, 'augment')
        # 遍历所有数据集文件，分别训练各子模型
        for train_dataset in tqdm(os.listdir(path_train), desc='Seperate model train'):
            base_name = os.path.basename(train_dataset).replace('.npz', '')

            # 加载数据集
            data = np.load(os.path.join(path_train, train_dataset))
            X_train = data['X_train']
            y_train = data['y_train']

            dir_model_save = os.path.join(path_project, f'adversarial_ensemble_AD/models/ensemble/n=2')
            os.makedirs(dir_model_save, exist_ok=True)  # 创建结果文件夹
            path_model = os.path.join(dir_model_save, f'DeepSAD-{base_name}.pth')
            # 初始迭代
            if iteration == 0:
                # 实例化DeepSAD并存到model_list中
                model = DeepSAD(seed=config.seed, config=config.DeepSAD_config)
                model.fit(X_train=X_train, y_train=y_train)

                model.deepSAD.save_model(export_model=path_model, save_ae=True)
            else:
                # 加载模型
                load_model = path_model
                model = DeepSAD(seed=config.seed, load_model=load_model, config=config.DeepSAD_config)

                # 训练模型
                model.fit_encoder(X_train=X_train, y_train=y_train)
                model.deepSAD.save_model(export_model=path_model, save_ae=True)

            # TODO: 训练对抗样本生成器

            # TODO: 构造新的训练数据集