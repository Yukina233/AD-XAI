import torch
import logging
import random
import numpy as np
import pandas as pd
import os
from .utils.config import Config
from .utils.visualization.plot_images_grid import plot_images_grid
from .deepsad import deepsad
from .datasets.main import load_dataset
from adbench_modified.myutils import Utils


class DeepSAD():
    def __init__(self, seed, model_name='DeepSAD', load_model=None, config=None):
        self.utils = Utils()
        self.device = self.utils.get_device(gpu_specific=True)  # get device
        self.seed = seed

        self.xp_path = None
        self.load_config = None
        self.load_model = load_model
        self.eta = 1.0  # eta in the loss function
        self.optimizer_name = 'adam'
        self.lr = 0.001
        self.n_epochs = 20
        self.lr_milestone = [50, ]
        self.batch_size = 128
        self.weight_decay = 1e-6
        self.pretrain = True  # whether to use auto-encoder for pretraining
        self.ae_optimizer_name = 'adam'
        self.ae_lr = 0.001
        self.ae_n_epochs = 20
        self.ae_lr_milestone = [50, ]
        self.ae_batch_size = 128
        self.ae_weight_decay = 1e-6
        self.num_threads = 0
        self.n_jobs_dataloader = 0

        if config is not None:
            for key, value in config.items():
                setattr(self, key, value)

    def fit(self, X_train, y_train, ratio=None):
        """
        Deep SAD, a method for deep semi-supervised anomaly detection.

        :arg DATASET_NAME: Name of the dataset to load.
        :arg NET_NAME: Name of the neural network to use.
        :arg XP_PATH: Export path for logging the experiment.
        """

        # Set seed (using myutils)
        self.utils.set_seed(self.seed)

        # Set the number of threads used for parallelizing CPU operations
        if self.num_threads > 0:
            torch.set_num_threads(self.num_threads)
        logging.info('Computation device: %s' % self.device)
        logging.info('Number of threads: %d' % self.num_threads)
        logging.info('Number of dataloader workers: %d' % self.n_jobs_dataloader)

        # Load data
        data = {'X_train': X_train, 'y_train': y_train}
        normal_train_data = {'X_train': X_train[y_train == 0], 'y_train': y_train[y_train == 0]}
        dataset = load_dataset(data=data, train=True)
        ae_dataset = load_dataset(data=normal_train_data, train=True)
        input_size = dataset.train_set.data.size(1)  # input size

        # Initialize DeepSAD model and set neural network phi
        self.deepSAD = deepsad(self.eta, ae_dataset)

        if X_train.ndim == 2:
            # 对应use_preprocess==True时，即pretrain_encoder=ResNet18
            self.net_name = 'dense'
        else:
            if X_train.shape[1] == 3:
                self.net_name = 'cifar10_LeNet'
            elif X_train.shape[1] == 1:
                self.net_name = 'cifar10_LeNet_1'
        self.deepSAD.set_network(self.net_name, input_size)

        # If specified, load Deep SAD model (center c, network weights, and possibly autoencoder weights)
        if self.load_model:
            self.deepSAD.load_model(model_path=self.load_model, load_ae=True, map_location=self.device, input_size=input_size)
            logging.info('Loading model from %s.' % self.load_model)

        else:
            logging.info('Pretraining: %s' % self.pretrain)
            if self.pretrain:
                # Pretrain model on dataset (via autoencoder)
                self.deepSAD.pretrain(ae_dataset,
                                      input_size,
                                      optimizer_name=self.ae_optimizer_name,
                                      lr=self.ae_lr,
                                      n_epochs=self.ae_n_epochs,
                                      lr_milestones=self.ae_lr_milestone,
                                      batch_size=self.ae_batch_size,
                                      weight_decay=self.ae_weight_decay,
                                      device=self.device,
                                      n_jobs_dataloader=self.n_jobs_dataloader)

            # Train model on dataset
            self.deepSAD.train(dataset,
                               optimizer_name=self.optimizer_name,
                               lr=self.lr,
                               n_epochs=self.n_epochs,
                               lr_milestones=self.lr_milestone,
                               batch_size=self.batch_size,
                               weight_decay=self.weight_decay,
                               device=self.device,
                               n_jobs_dataloader=self.n_jobs_dataloader)

        # Save results, model, and configuration
        # deepSAD.save_results(export_json=xp_path + '/results.json')
        # deepSAD.save_model(export_model=xp_path + '/model.tar')
        # cfg.save_config(export_json=xp_path + '/config.json')

        # Plot most anomalous and most normal test samples
        # indices, labels, scores = zip(*deepSAD.results['test_scores'])
        # indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
        # idx_all_sorted = indices[np.argsort(scores)]  # from lowest to highest score
        # idx_normal_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # from lowest to highest score

        return self

    # 只训练encoder部分
    def fit_encoder(self, X_train, y_train, ratio=None):
        """
        Deep SAD, a method for deep semi-supervised anomaly detection.

        :arg DATASET_NAME: Name of the dataset to load.
        :arg NET_NAME: Name of the neural network to use.
        :arg XP_PATH: Export path for logging the experiment.
        """

        # Set seed (using myutils)
        self.utils.set_seed(self.seed)

        # Set the number of threads used for parallelizing CPU operations
        if self.num_threads > 0:
            torch.set_num_threads(self.num_threads)
        logging.info('Computation device: %s' % self.device)
        logging.info('Number of threads: %d' % self.num_threads)
        logging.info('Number of dataloader workers: %d' % self.n_jobs_dataloader)

        # Load data
        data = {'X_train': X_train, 'y_train': y_train}
        dataset = load_dataset(data=data, train=True)
        input_size = dataset.train_set.data.size(1)  # input size

        # Initialize DeepSAD model and set neural network phi
        self.deepSAD = deepsad(self.eta)

        if X_train.ndim == 2:
            # 对应use_preprocess==True时，即pretrain_encoder=ResNet18
            self.net_name = 'dense'
        else:
            if X_train.shape[1] == 3:
                self.net_name = 'cifar10_LeNet'
            elif X_train.shape[1] == 1:
                self.net_name = 'cifar10_LeNet_1'
        self.deepSAD.set_network(self.net_name, input_size)

        # If specified, load Deep SAD model (center c, network weights, and possibly autoencoder weights)
        if self.load_model:
            self.deepSAD.load_model(model_path=self.load_model, load_ae=True, map_location=self.device, input_size=input_size)
            logging.info('Loading model from %s.' % self.load_model)

            # Train model on dataset
            self.deepSAD.train(dataset,
                               optimizer_name=self.optimizer_name,
                               lr=self.lr,
                               n_epochs=self.n_epochs,
                               lr_milestones=self.lr_milestone,
                               batch_size=self.batch_size,
                               weight_decay=self.weight_decay,
                               device=self.device,
                               n_jobs_dataloader=self.n_jobs_dataloader)
        else:
            raise NotImplemented


        # Save results, model, and configuration
        # deepSAD.save_results(export_json=xp_path + '/results.json')
        # deepSAD.save_model(export_model=xp_path + '/model.tar')
        # cfg.save_config(export_json=xp_path + '/config.json')

        # Plot most anomalous and most normal test samples
        # indices, labels, scores = zip(*deepSAD.results['test_scores'])
        # indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
        # idx_all_sorted = indices[np.argsort(scores)]  # from lowest to highest score
        # idx_normal_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # from lowest to highest score

        return self

    def load_model_from_file(self, input_size=39):
        """
                Deep SAD, a method for deep semi-supervised anomaly detection.

                :arg DATASET_NAME: Name of the dataset to load.
                :arg NET_NAME: Name of the neural network to use.
                :arg XP_PATH: Export path for logging the experiment.
                """


        self.net_name = 'dense'
        # Set seed (using myutils)
        self.utils.set_seed(self.seed)

        # Set the number of threads used for parallelizing CPU operations
        if self.num_threads > 0:
            torch.set_num_threads(self.num_threads)
        logging.info('Computation device: %s' % self.device)
        logging.info('Number of threads: %d' % self.num_threads)
        logging.info('Number of dataloader workers: %d' % self.n_jobs_dataloader)

        # Initialize DeepSAD model and set neural network phi
        self.deepSAD = deepsad(self.eta)


        self.deepSAD.set_network(self.net_name, input_size)

        # If specified, load Deep SAD model (center c, network weights, and possibly autoencoder weights)
        if self.load_model:
            self.deepSAD.load_model(model_path=self.load_model, load_ae=True, map_location=self.device,
                                    input_size=input_size)
            logging.info('Loading model from %s.' % self.load_model)
    def predict_score(self, X):
        # input randomly generated y label for consistence
        dataset = load_dataset(data={'X_test': X, 'y_test': np.random.choice([0, 1], X.shape[0])}, train=False)
        score, outputs = self.deepSAD.test(dataset, device=self.device, n_jobs_dataloader=self.n_jobs_dataloader)

        return score, outputs
