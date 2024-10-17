import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader  # 用于加载数据，可实例化
import os


path_project = '/media/test/d/Yukina/AD-XAI_data'

image_size = [1, 28, 28]
latent_dim = 96
batch_size = 64
use_gpu = torch.cuda.is_available()



class Generator(nn.Module):
    def __init__(self):  # 定义一些模块
        super(Generator, self).__init__()  # 对父类实例化，因为是继承自nn.Module
        # 用Sequential来装网络层
        self.model = nn.Sequential(
            # 堆很多全连接层
            # 将参数in_dim(图片大小)写明成torch.prod(image_size,dtype=torch.int32)
            nn.Linear(latent_dim, 128),
            # 引入batchnorm可以提高收敛速度，具体做法是在生成器的Linear层后面添加BatchNorm1d，
            # 最后一层除外，判别器不要加
            torch.nn.BatchNorm1d(128),
            torch.nn.GELU(),  # 将激活函数ReLU换成GELU效果更好

            nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.GELU(),

            nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.GELU(),

            nn.Linear(512, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.GELU(),

            nn.Linear(1024, np.prod(image_size, dtype=np.int32)),  # 映射成图片大小（输出）
            #  nn.Tanh(),
            nn.Sigmoid(),
        )

    def forward(self, z):  # 将以上模块串联起来
        # z就是高斯噪声,shape of z:[batchsize,latent_dim]
        output = self.model(z)
        image = output.reshape(z.shape[0], *image_size)  # image_size是一个列表，加*变成元组

        return image

class Discriminator(nn.Module):  # 判别器接受一张图片作为输入，输出是经Sigmoid后的概率值
    def __init__(self):
        super(Discriminator, self).__init__()  # 对父类实例化，因为是继承自nn.Module
        # 用Sequential来装网络层
        self.model = nn.Sequential(
            # 堆很多全连接层
            nn.Linear(np.prod(image_size, dtype=np.int32), 512),  # 输入是图片
            torch.nn.GELU(),

            nn.Linear(512, 256),
            torch.nn.GELU(),

            nn.Linear(256, 128),
            torch.nn.GELU(),

            nn.Linear(128, 64),
            torch.nn.GELU(),

            nn.Linear(64, 32),
            torch.nn.GELU(),

            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, image):  # 将以上模块串联起来
        # shape of image:[batchsize,1,28,28]
        prob = self.model(image.reshape(image.shape[0], -1))  # reshape 4维成2维的，-1表示其他的放在一维

        return prob


# Training
# 真实图片
dataset = torchvision.datasets.MNIST(root=os.path.join(path_project, 'data/MNIST'), train=True, download=False,
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.Resize(28),
                                         torchvision.transforms.ToTensor(),  # 将PIL转换Tensor
                                         # 实际算出来的mean和std的效果不如0.5,0.5（也可以不用做归一化的transform）
                                         #                                         torchvision.transforms.Normalize(mean=[0.5],std=[0.5]),
                                     ]))
# print(len(dataset))
# for i in range(len(dataset)):
#     if i<5:
#         print(dataset[i][0].shape) # 每个样本大小是1*28*28，1为通道数
#     else:
#         break
# DataLoder的作用就是将dataset中数据构成mimi_batch来做批训练
# 当你使用了batch normalization的时候，如果batch_size设置得不合适，又没有使用 drop_last = true，易出错
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
# 2022.07.14注意这里是batch_size=batch_size(即64)


# 先实例化，因为后面优化器中要求参数是迭代器
generator = Generator()
discriminator = Discriminator()

# betas表示参数β，weight_decay表示权重衰减(正则化)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0003, betas=(0.4, 0.8), weight_decay=0.0001)

loss_fn = nn.BCELoss()
labels_one = torch.ones(batch_size, 1)
labels_zero = torch.zeros(batch_size, 1)

if use_gpu:  # 有cuda
    print("use gpu for training")
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    loss_fn = loss_fn.cuda()
    labels_one = labels_one.to("cuda")
    labels_zero = labels_zero.to("cuda")

num_epoch = 200

# 对照论文写博客
for epoch in range(num_epoch):
    for i, mimi_batch in enumerate(dataloader):
        gt_images, _ = mimi_batch  # 真实图片，label就不要了，写成_
        z = torch.randn(batch_size, latent_dim)  # 随机生成高斯变量，latent_dim表示隐变量维度

        if use_gpu:
            gt_images = gt_images.to("cuda")
            z = z.to("cuda")

        pred_images = generator(z)  # 预测出的图片
        g_optimizer.zero_grad()  # 先置零
        #         print(len(gt_images))
        #         print(len(pred_images))

        recons_loss = torch.abs(pred_images - gt_images).mean()
        # 把预测出的图片送入discriminator,即论文中的 D(G(z))
        g_loss = loss_fn(discriminator(pred_images),
                                              labels_one)  # 对生成器为进行优化,即生成的图片越接近真实的越好，所以target为1

        g_loss.backward()
        g_optimizer.step()

        d_optimizer.zero_grad()
        # 对照论文的优化目标函数来理解
        #         d_loss = 0.5*(loss_fn(discriminator(gt_images)
        #                          ,torch.ones(batch_size,1)) # 因为是BCELoss，所以要传入标签，希望判别器是能够把真实的图片判别为1
        # + loss_fn(discriminator(pred_images.detach()) # 因为计算d_loss不需要计算生成器部分的梯度，所以detach，从计算图中分离
        #                         ,torch.zeros(batch_size,1))) # 此时希望判别器将真实图片判别为0

        # 为了方便观察判别器是否训练好，应该看以下两个loss
        # real_loss基于真实图片，fake_loss基于生成图片
        # 观察real_loss与fake_loss同时下降且同时达到最小值，并且差不多大，说明判别器已经稳定
        real_loss = loss_fn(discriminator(gt_images), labels_one)
        fake_loss = loss_fn(discriminator(pred_images.detach()), labels_zero)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        d_optimizer.step()

        # 建议引入loss打印语句
        if i % 1000 == 0:
            print(
                f"step:{len(dataloader) * epoch + i}, recons_loss:{recons_loss.item()}, g_loss:{g_loss.item()}, d_loss:{d_loss.item()},real_loss:{real_loss.item()},fake_loss:{fake_loss.item()}")

    if epoch % 5 == 0:
        image = pred_images[:16].data
        torchvision.utils.save_image(image, os.path.join(path_project, "MNIST_dataset/plots/without_recon_loss", f"image_{len(dataloader) * epoch + i}.png"), nrow=4)
