import os.path
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

path_project = '/media/test/d/Yukina/AD-XAI'
def dataset_train_to_dataframe(dataset):
    images = []
    labels = []
    for img, label in dataset:
        if label == 0:
            # 将图像展平为一维向量
            images.append(img.view(-1).numpy())
            labels.append(0)
    # 将图像和标签转换为DataFrame
    df = pd.DataFrame(images)
    df['is_anomaly'] = labels
    return df

def dataset_test_to_dataframe(dataset):
    images = []
    labels = []
    for img, label in dataset:
        if label == 0:
            # 将图像展平为一维向量
            images.append(img.view(-1).numpy())
            labels.append(0)
        elif label == 1:
            images.append(img.view(-1).numpy())
            labels.append(1)
    # 将图像和标签转换为DataFrame
    df = pd.DataFrame(images)
    df['is_anomaly'] = labels
    return df
# 定义数据预处理（如归一化）
transform = transforms.ToTensor()
# 加载训练和测试数据集
train_dataset = datasets.MNIST(root=os.path.join(path_project, 'data/MNIST'), train=True, download=True,
                               transform=transform)
test_dataset = datasets.MNIST(root=os.path.join(path_project, 'data/MNIST'), train=False, download=True,
                              transform=transform)
# 转换训练集和测试集
train_df = dataset_train_to_dataframe(train_dataset)
test_df = dataset_test_to_dataframe(test_dataset)
# 保存为CSV文件
train_csv_path = os.path.join(path_project, 'data/MNIST/mnist.train.csv')
test_csv_path = os.path.join(path_project, 'data/MNIST/mnist.test.csv')

train_df.to_csv(train_csv_path, index=True)
test_df.to_csv(test_csv_path, index=True)
print(f"训练集已保存到 {train_csv_path}")
print(f"测试集已保存到 {test_csv_path}")

sample_train = train_df.sample(n=5, random_state=0)
for i, row in sample_train.iterrows():
    img = row.values[:-1].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # 不显示坐标轴
    # plt.title(f"Sample: {i}")
    # plt.show()
    plt.savefig(os.path.join(path_project, f'MNIST_dataset/plots/origin-{i}'))
