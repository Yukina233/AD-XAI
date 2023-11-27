import json

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from torch.utils.data import Subset

from base.base_dataset import BaseADDataset
from networks.main import build_network, build_autoencoder
from optim.DeepSAD_trainer import DeepSADTrainer
from optim.ae_trainer import AETrainer


class DeepSAD(object):
    """A class for the Deep SAD method.

    Attributes:
        eta: Deep SAD hyperparameter eta (must be 0 < eta).
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network phi.
        trainer: DeepSADTrainer to train a Deep SAD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SAD network.
        ae_net: The autoencoder network corresponding to phi for network weights pretraining.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
        ae_results: A dictionary to save the autoencoder results.
    """

    def __init__(self, eta: float = 1.0):
        """Inits DeepSAD with hyperparameter eta."""

        self.eta = eta
        self.c = None  # hypersphere center c

        self.net_name = None
        self.net = None  # neural network phi

        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.normal_ae_net = None
        self.anomaly_ae_net = None  # autoencoder network for anomaly samples

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

        self.ae_results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None
        }

    def set_network(self, net_name):
        """Builds the neural network phi."""
        self.net_name = net_name
        self.net = build_network(net_name)

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        """Trains the Deep SAD model on the training data."""

        self.optimizer_name = optimizer_name
        self.trainer = DeepSADTrainer(self.c, self.eta, optimizer_name=optimizer_name, lr=lr, n_epochs=n_epochs,
                                      lr_milestones=lr_milestones, batch_size=batch_size, weight_decay=weight_decay,
                                      device=device, n_jobs_dataloader=n_jobs_dataloader)
        # Get the model
        self.net = self.trainer.train(dataset, self.net)
        self.results['train_time'] = self.trainer.train_time
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get as list

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Deep SAD model on the test data."""

        if self.trainer is None:
            self.trainer = DeepSADTrainer(self.c, self.eta, device=device, n_jobs_dataloader=n_jobs_dataloader)

        self.trainer.test(dataset, self.net)

        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores
    def test_on_trainset_at_start(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Deep SAD model on the train data."""

        if self.trainer is None:
            self.trainer = DeepSADTrainer(self.c, self.eta, device=device, n_jobs_dataloader=n_jobs_dataloader)

        self.trainer.test_on_trainset(dataset, self.net)

        # Get results
        self.results['train_set_size_0'] = self.trainer.train_scores.__len__()
        self.results['train_scores_0'] = self.trainer.train_scores

    def test_on_trainset(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Deep SAD model on the train data."""

        if self.trainer is None:
            self.trainer = DeepSADTrainer(self.c, self.eta, device=device, n_jobs_dataloader=n_jobs_dataloader)

        self.trainer.test_on_trainset(dataset, self.net)

        # Get results
        self.results['train_set_size'] = self.trainer.train_scores.__len__()
        self.results['train_scores'] = self.trainer.train_scores

    def pretrain_ae_normal(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001,
                            n_epochs: int = 100, lr_milestones: tuple = (), batch_size: int = 128,
                            weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        self.normal_ae_net = build_autoencoder(self.net_name)
        # Train
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.normal_ae_net = self.ae_trainer.train_normal(dataset, self.normal_ae_net)
    def pretrain_ae_anomaly(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001,
                            n_epochs: int = 100, lr_milestones: tuple = (), batch_size: int = 128,
                            weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        self.anomaly_ae_net = build_autoencoder(self.net_name)
        # Train
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.anomaly_ae_net = self.ae_trainer.train_anomaly(dataset, self.anomaly_ae_net)

    def remove_anomaly_in_normal(self, dataset: BaseADDataset, remove_threshold: float = 0.06, n_clusters: int = 4,
                                 seed: int = 0, batch_size: int = 128,
                                 device: str = 'cuda', n_jobs_dataloader: int = 0, xp_path: str = None):
        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=batch_size, num_workers=n_jobs_dataloader)
        train_anomaly_loader = dataset.anomaly_data_loader(batch_size=dataset.train_set_anomaly.indices.__len__(),
                                                           num_workers=n_jobs_dataloader)
        idx_label_semi_target_score = []
        net = self.anomaly_ae_net.encoder.to(device)
        net.eval()
        with torch.no_grad():
            for data in train_anomaly_loader:
                # get the inputs of the batch
                inputs, _, _, _ = data
                inputs = inputs.to(device)
                outputs = net(inputs)
        # 使用KMeans++进行聚类
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=seed)
        anomaly_data = outputs.cpu().detach().numpy()
        anomaly_y = kmeans.fit_predict(anomaly_data)
        normal_data = torch.tensor([], dtype=torch.int64, device=device)
        # 删去在表示空间与异常点接近的正常点
        remove_indices = torch.tensor([], dtype=torch.int64, device=device)
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, labels, semi_targets, idx = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                idx = idx.to(device)
                outputs = net(inputs)
                idx_normal = torch.where(labels == 0)[0]
                outputs = outputs[idx_normal]
                idx = idx[idx_normal]
                distance_list = []
                normal_data = torch.cat((normal_data, outputs))

                for center in kmeans.cluster_centers_:  # 计算每个点到聚类中心的距离
                    expanded_center = torch.tensor(center, device=device).unsqueeze(0).expand_as(outputs)
                    distance_list.append(torch.norm(outputs - expanded_center, dim=1, p=2))
                min_distance = torch.min(torch.stack(distance_list), dim=0).values
                idx_under_threshold = torch.where(min_distance < remove_threshold)[0]
                remove_indices = torch.cat((remove_indices, idx[idx_under_threshold]))
                # Save triples of (idx, label, score) in a list
                idx_label_semi_target_score += list(zip(idx.cpu().data.numpy().tolist(),
                                                        labels.cpu().data.numpy().tolist(),
                                                        semi_targets.cpu().data.numpy().tolist(),
                                                        min_distance.cpu().data.numpy().tolist()))
        # self.ae_results['remove_threshold'] = remove_threshold
        self.results['train_normal_dis_to_anomaly'] = idx_label_semi_target_score
        # 将NumPy tensor转换为set，用于快速查找
        remove_set = set(remove_indices.cpu().detach().numpy())
        # 使用列表推导式来构建一个新的列表，只包含不在remove_set中的indices的元素
        indices_filtered = [item for item in dataset.train_set.indices if item not in remove_set]
        dataset.train_set.indices = indices_filtered

        from sklearn.manifold import TSNE
        normal_y = -torch.ones(normal_data.shape[0], dtype=torch.int64, device=device).cpu().detach().numpy()
        X = np.concatenate((normal_data.cpu().detach().numpy(), anomaly_data), axis=0)
        y = np.concatenate((normal_y, anomaly_y), axis=0)
        tsne = TSNE(n_components=2, random_state=0)  # n_components表示目标维度
        X_2d = tsne.fit_transform(X)  # 对数据进行降维处理
        plt.figure(figsize=(8, 6))
        if y is not None:
            # 如果有目标数组，根据不同的类别用不同的颜色绘制
            for i in np.unique(y):
                plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], label=i)
            plt.legend()
        else:
            # 如果没有目标数组，直接绘制
            plt.scatter(X_2d[:, 0], X_2d[:, 1])
        plt.title('t-SNE Visualization of latent space')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.savefig(xp_path + '/t-SNE Visualization')
    def pretrain(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        """Pretrains the weights for the Deep SAD network phi via autoencoder."""

        # Set autoencoder network
        self.ae_net = build_autoencoder(self.net_name)

        # Train
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train_normal(dataset, self.ae_net)

        # Get train results
        self.ae_results['train_time'] = self.ae_trainer.train_time

        # Test
        self.ae_trainer.test(dataset, self.ae_net)

        # Get test results
        self.ae_results['test_auc'] = self.ae_trainer.test_auc
        self.ae_results['test_time'] = self.ae_trainer.test_time

        # Initialize Deep SAD network weights from pre-trained encoder
        self.init_network_weights_from_pretraining()

    def init_network_weights_from_pretraining(self):
        """Initialize the Deep SAD network weights from the encoder weights of the pretraining autoencoder."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)

    def save_model(self, export_model, save_ae=True):
        """Save Deep SAD model to export_model."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict() if save_ae else None

        torch.save({'c': self.c,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, export_model)

    def load_model(self, model_path, load_ae=False, map_location='cpu'):
        """Load Deep SAD model from model_path."""

        model_dict = torch.load(model_path, map_location=map_location)

        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])

        # load autoencoder parameters if specified
        if load_ae:
            if self.ae_net is None:
                self.ae_net = build_autoencoder(self.net_name)
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)

    def save_ae_results(self, export_json):
        """Save autoencoder results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.ae_results, fp, indent=4)
