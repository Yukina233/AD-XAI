import os

import numpy as np
from PIL import Image
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MVTecADDataset(Dataset):
    def __init__(self, root_dir, category, is_train=True, transform=None):
        self.root_dir = root_dir
        self.category = category
        self.is_train = is_train
        self.transform = transform

        self.data_type = 'train' if self.is_train else 'test'
        self.img_dir = os.path.join(self.root_dir, self.category, self.data_type)

        self.img_paths = []
        self.labels = []

        # Load all image file paths and their labels
        if self.is_train:
            # For training data, we consider all images as normal
            normal_dir = os.path.join(self.img_dir, 'good')
            for img_name in os.listdir(normal_dir):
                self.img_paths.append(os.path.join(normal_dir, img_name))
                self.labels.append(0)  # 0 for normal class
        else:
            # For test data, load both normal and abnormal images
            for class_name in os.listdir(self.img_dir):
                class_dir = os.path.join(self.img_dir, class_name)
                for img_name in os.listdir(class_dir):
                    self.img_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(0 if class_name == 'good' else 1)  # 1 for anomaly class

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def generate_dataset(root_dir, category, transform):
    train_dataset = MVTecADDataset(root_dir=root_dir, category=category, is_train=True, transform=transform)
    test_dataset = MVTecADDataset(root_dir=root_dir, category=category, is_train=False, transform=transform)

    # Create a DataLoader
    train_loader = DataLoader(train_dataset, batch_size=500, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

    X = torch.Tensor()
    y = torch.Tensor()

    # Now you can iterate over train_loader and test_loader
    for images, labels in train_loader:
        X = torch.concat((X, images))
        y = torch.concat((y, labels))

    for images, labels in test_loader:
        X = torch.concat((X, images))
        y = torch.concat((y, labels))

    # Convert tensors to numpy arrays
    X_np = X.numpy()
    y_np = y.numpy()

    # Save as NPZ
    np.savez(path_project + f"/data/mvtec_ad/MVTec-AD_{category}.npz", X=X_np, y=y_np)


path_project = '/home/yukina/Missile_Fault_Detection/project'
# Define a transform to apply to each image
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# Create the dataset
root_dir = path_project + '/data/mvtec_ad_raw'
category_list = ['bottle', 'cable', 'capsule', 'carpet', 'hazelnut', 'leather',
                 'metal_nut', 'pill', 'tile', 'toothbrush', 'transistor', 'wood']
for category in category_list:
    generate_dataset(root_dir, category, transform)

print('Data generate complete.')