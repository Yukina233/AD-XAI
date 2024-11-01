import os.path
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
import numpy as np



# matplotlib.use('TkAgg')

path_project = '/media/test/d/Yukina/AD-XAI_data'


def dataset_to_dataframe(dataset):
    images = []
    labels = []
    for img, label in dataset:
        images.append(img.view(-1).numpy())
        labels.append(0)
    # 将图像和标签转换为DataFrame
    df = pd.DataFrame(images)
    df['is_anomaly'] = labels
    return df


def dataset_train_to_dataframe(dataset, y):
    images = []
    labels = []
    for img, label in dataset:
        if label == y:
            # 将图像展平为一维向量
            images.append(img.view(-1).numpy())
            labels.append(0)
    # 将图像和标签转换为DataFrame
    df = pd.DataFrame(images)
    df['is_anomaly'] = labels
    return df


def dataset_test_to_dataframe(dataset, y):
    images = []
    labels = []
    for img, label in dataset:
        if label == y:
            # 将图像展平为一维向量
            images.append(img.view(-1).numpy())
            labels.append(0)
        else:
            images.append(img.view(-1).numpy())
            labels.append(1)
    # 将图像和标签转换为DataFrame
    df = pd.DataFrame(images)
    df['is_anomaly'] = labels
    return df

dataset_name = 'MNIST_nonorm'
y = 9
# 定义数据预处理（如归一化）
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
])
# 加载训练和测试数据集
train_dataset = datasets.MNIST(root=os.path.join(path_project, 'data/MNIST'), train=True, download=True,
                               transform=transform)
test_dataset = datasets.MNIST(root=os.path.join(path_project, 'data/MNIST'), train=False, download=True,
                              transform=transform)
# 转换训练集和测试集
# train_df = dataset_to_dataframe(train_dataset)
train_df = dataset_train_to_dataframe(train_dataset, y)
test_df = dataset_test_to_dataframe(test_dataset, y)
# 保存为CSV文件
output_dir = os.path.join(path_project, f'data/{dataset_name}_{str(y)}')
os.makedirs(output_dir, exist_ok=True)
train_csv_path = os.path.join(path_project, output_dir, f'mnist.train.csv')
test_csv_path = os.path.join(path_project, output_dir, f'mnist.test.csv')

train_df.to_csv(train_csv_path, index=True)
test_df.to_csv(test_csv_path, index=True)
print(f"训练集已保存到 {train_csv_path}")
print(f"测试集已保存到 {test_csv_path}")


num_generate = 15
sample_train = train_df.sample(n=num_generate, random_state=0)

fig, axes = plt.subplots(1, 15, figsize=(30, 2))

# 遍历样本并将其显示在子图中
for i, (index, row) in enumerate(sample_train.iterrows()):
    img = row.values[:-1].reshape(28, 28)

    ax = axes[i]  # 确定子图位置
    ax.imshow(img, cmap='gray')
    ax.axis('off')  # 不显示坐标轴

# 调整子图间的空隙
plt.subplots_adjust(wspace=0.1, hspace=0.1)

plt.tight_layout()

# 显示图像
# plt.show()

# 保存成一张图像
plt_dir = os.path.join(path_project, f'{dataset_name}_dataset/plots/{str(y)}')
os.makedirs(plt_dir, exist_ok=True)
plt.savefig(os.path.join(plt_dir, f'origin.png'))

