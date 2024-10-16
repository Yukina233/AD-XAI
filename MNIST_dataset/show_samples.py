# from GHL_dataset.data_generate.wgan_gp import Adversarial_Generator
from adversarial_ensemble_AD.data_generate.gan_mnist import Adversarial_Generator
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

path_project = '/media/test/d/Yukina/AD-XAI'
GAN_config = {
            "seed": 0,
            "latent_dim": 50,
            "lr": 0.0003,
            "clip_value": 0.01,
            "lambda_gp": 10000,
            "n_epochs": 1,
            "lam1": 0,
            "lam2": 0,
            "lam3": 0,
            "tau1": 1,
            "img_size": 784
}

epoch = 2
model_dir = os.path.join(path_project,
                         'MNIST_dataset/models/MNIST/ensemble/GAN, euc, window=1, step=1, K=7,deepsad_ae_epoch=1,gan_epoch=20,pre_epochs=0,lam1=0,lam2=0,lam3=0,latent_dim=50,lr=0.0003,clip_value=0.01,lambda_gp=10000,seed=0',
                         f'{epoch}')
# 加载生成模型
generator = Adversarial_Generator(config=GAN_config)
generator.load_model(model_dir)
num_generate = 20
gen_samples = generator.sample_generate(num=num_generate)

data = gen_samples.cpu().detach().numpy()  # 如果你使用torch，请取消注释这一行
# 展示前5个样本的图像
for i in range(num_generate):  # 只显示前5张图片
    image = data[i, 0, :].reshape(28, 28)  # 提取第i个样本并重塑为28x28
    # plt.figure(figsize=(2, 2))  # 设置图像大小
    plt.imshow(image, cmap='gray')  # 以灰度显示图像
    plt.axis('off')  # 不显示坐标轴
    # plt.title(f'Sample {i+1}')  # 添加标题，标记是第几个样本
    plt.show()
    # plt.savefig(os.path.join(path_project, f'MNIST_dataset/plots/GAN-{i}'))
