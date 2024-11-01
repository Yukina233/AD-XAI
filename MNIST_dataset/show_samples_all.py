# from GHL_dataset.data_generate.wgan_gp import Adversarial_Generator
from adversarial_ensemble_AD.data_generate.gan_mnist import Adversarial_Generator
import os
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use('TkAgg')
path_project = '/media/test/d/Yukina/AD-XAI_data'


def get_sample(dataset_name, param_dir, y, epoch, GAN_config):
    model_dir = os.path.join(path_project,
                             f'{dataset_name}_dataset/models/{dataset_name}/ensemble/',
                             param_dir,
                             f'{epoch}')
    # 加载生成模型
    generator = Adversarial_Generator(config=GAN_config)
    generator.load_model(model_dir)
    num_generate = 2
    gen_samples = generator.sample_generate(num=num_generate)

    return gen_samples.cpu().detach().numpy()[0]  # 如果你使用torch，请取消注释这一行


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    epoch = 5
    GAN_config = {
        "seed": 0,
        "latent_dim": 96,
        "lr": 0.0003,
        "clip_value": 0.01,
        "lambda_gp": 10000,
        "n_epochs": 1,
        "lam1": 100,
        "lam2": 10,
        "lam3": 0,
        "tau1": 1,
        "img_size": 784
    }

    param_dir = 'GAN_web_w_decay, euc, window=1, step=1, K=7,deepsad_ae_epoch=10,gan_epoch=1,pre_epochs=0,lam1=0,lam2=20,latent_dim=96,lr=0.003,seed=0'

    data = []
    for y in range(0, 10):
        dataset_name = f'MNIST_nonorm_{y}'
        data.append(get_sample(dataset_name, param_dir, y, epoch, GAN_config))

        # 设置图像的行和列数，5行4列，总共20张图像
        fig, axs = plt.subplots(1, 10, figsize=(20, 2))

    # 逐一展示每个生成的图像
    for i in range(0, 10):
        ax = axs[i]  # 计算子图的位置
        image = data[i].reshape(28, 28)  # 提取第i个样本并重塑为28x28
        ax.imshow(image, cmap='gray')  # 以灰度显示图像
        ax.axis('off')  # 不显示坐标轴

    # 调整子图间距
    plt.tight_layout()

    # 展示图像
    # plt.show()

    # 如果需要保存这张图像，可以取消以下注释
    item1 = dataset_name.split('_')[0]
    item2 = dataset_name.split('_')[1]
    name_no_id = f'{item1}_{item2}'
    plt_dir = os.path.join(path_project, f'{name_no_id}_dataset/plots/all')
    os.makedirs(plt_dir, exist_ok=True)
    plt.savefig(os.path.join(plt_dir, f'{param_dir}.png'))
