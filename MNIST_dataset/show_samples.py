# from GHL_dataset.data_generate.wgan_gp import Adversarial_Generator
from adversarial_ensemble_AD.data_generate.gan_mnist import Adversarial_Generator
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

dataset_name = 'MNIST_all_nonorm'
path_project = '/media/test/d/Yukina/AD-XAI_data'
GAN_config = {
            "seed": 0,
            "latent_dim": 96,
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

epoch = 1
model_dir = os.path.join(path_project,
                         f'{dataset_name}_dataset/models/{dataset_name}/ensemble/',
                         'GAN_sigmoid, euc, window=1, step=1, K=7,deepsad_ae_epoch=1,gan_epoch=1,pre_epochs=0,lam1=0,lam2=0,lam3=0,latent_dim=96,lr=0.0003,clip_value=0.01,lambda_gp=10000,seed=0',
                         f'{epoch}')
# 加载生成模型
generator = Adversarial_Generator(config=GAN_config)
generator.load_model(model_dir)
num_generate = 15
gen_samples = generator.sample_generate(num=num_generate)

data = gen_samples.cpu().detach().numpy()  # 如果你使用torch，请取消注释这一行

# 设置图像的行和列数，5行4列，总共20张图像
fig, axs = plt.subplots(1, 15, figsize=(30, 2))

# 逐一展示每个生成的图像
for i in range(num_generate):
    ax = axs[i]  # 计算子图的位置
    image = data[i, 0, :].reshape(28, 28)  # 提取第i个样本并重塑为28x28
    ax.imshow(image, cmap='gray')  # 以灰度显示图像
    ax.axis('off')  # 不显示坐标轴

# 调整子图间距
plt.tight_layout()

# 展示图像
plt.show()

# 如果需要保存这张图像，可以取消以下注释
# plt.savefig(os.path.join(path_project, 'MNIST_dataset/plots/GAN_samples.png'))
