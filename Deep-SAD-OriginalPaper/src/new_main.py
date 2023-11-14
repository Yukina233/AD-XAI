import argparse
import multiprocessing
import os
import time

import click
import torch
import logging
import random
import numpy as np

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from DeepSAD import DeepSAD
from datasets.main import load_dataset


def run(config=None):
    path_project = '/home/yukina/Missile_Fault_Detection/project/Deep-SAD-OriginalPaper'
    parser = argparse.ArgumentParser(description="Run Deep SAD experiments with specified parameters.")

    # Add arguments to the parser
    parser.add_argument('--dataset_name', type=str, default='cifar10',
                        help='Name of the dataset to load.{mnist, fmnist, cifar10, arrhythmia, cardio, satellite,'
                             ' satimage-2, shuttle, thyroid}')
    parser.add_argument('--net_name', type=str, default='cifar10_LeNet',
                        help='Name of the neural network to use.{mnist_LeNet, fmnist_LeNet, cifar10_LeNet, '
                             'arrhythmia_mlp, cardio_mlp, satellite_mlp, satimage-2_mlp, shuttle_mlp, thyroid_mlp}')
    parser.add_argument('--xp_path', type=str, default=path_project + '/log/test/log_' + time.strftime('%Y-%m-%d_%H-%M-%S'),
                        help='Export path for logging the experiment.')
    parser.add_argument('--data_path', type=str, default=path_project + '/data', help='Root path of data.')

    parser.add_argument('--load_config', type=str, default=None, help='Config JSON-file path (default: None).')
    parser.add_argument('--load_model', type=str, default=None, help='Model file path (default: None).')

    parser.add_argument('--eta', type=float, default=1.0, help='Deep SAD hyperparameter eta (must be 0 < eta).')
    parser.add_argument('--ratio_known_normal', type=float, default=0.2,
                        help='Ratio of known (labeled) normal training examples in all train examples.')
    parser.add_argument('--ratio_known_outlier', type=float, default=0.2,
                        help='Ratio of known (labeled) anomalous training examples in all train examples.')
    parser.add_argument('--ratio_pollution', type=float, default=0.0,
                        help='Pollution ratio of unlabeled training data with unknown (unlabeled) anomalies.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')

    parser.add_argument('--seed', type=int, default=42, help='Set seed. If -1, use randomization.')
    parser.add_argument('--optimizer_name', type=str, default='adam',
                        help='Name of the optimizer to use for Deep SAD network training.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate for Deep SAD network training. Default=0.001')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--lr_milestone', type=tuple, default=(50,),
                        help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay (L2 penalty) hyperparameter for Deep SAD objective.')

    parser.add_argument('--pretrain', type=bool, default=True,
                        help='Pretrain neural network parameters via autoencoder.')
    parser.add_argument('--ae_optimizer_name', type=str, default='adam',
                        help='Name of the optimizer to use for autoencoder pretraining.')
    parser.add_argument('--ae_lr', type=float, default=0.001,
                        help='Initial learning rate for autoencoder pretraining. Default=0.001')
    parser.add_argument('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
    parser.add_argument('--ae_lr_milestone', type=tuple, default=(50,),
                        help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
    parser.add_argument('--ae_batch_size', type=int, default=128,
                        help='Batch size for mini-batch autoencoder training.')
    parser.add_argument('--ae_weight_decay', type=float, default=1e-6,
                        help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')

    parser.add_argument('--num_threads', type=int, default=0,
                        help='Number of threads used for parallelizing CPU operations. 0 means that all resources are used.')
    parser.add_argument('--n_jobs_dataloader', type=int, default=0,
                        help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
    parser.add_argument('--normal_class', type=int, default=0,
                        help='Specify the normal class of the dataset (all other classes are considered anomalous).')
    parser.add_argument('--known_outlier_class', type=int, default=1,
                        help='Specify the known outlier class of the dataset for semi-supervised anomaly detection.')
    parser.add_argument('--n_known_outlier_classes', type=int, default=1,
                        help='Number of known outlier classes.'
                             'If 0, no anomalies are known.'
                             'If 1, outlier class as specified in --known_outlier_class option.'
                             'If > 1, the specified number of outlier classes will be sampled at random.')

    # Parse arguments
    args = parser.parse_args()

    # Load experiment config
    if config is not None:
        print(f"Running experiment with config: {config}")
        for key, value in config.items():
            setattr(args, key, value)
    del config

    # Run experiment with parsed arguments
    run_experiment(args)


def run_experiment(args):
    print(f"Running experiment with: {args}")
    # Get script arguments
    dataset_name = args.dataset_name
    net_name = args.net_name
    xp_path = args.xp_path
    if not os.path.exists(xp_path):
        os.makedirs(xp_path)
    data_path = args.data_path

    load_config = args.load_config
    load_model = args.load_model

    eta = args.eta
    ratio_known_normal = args.ratio_known_normal
    ratio_known_outlier = args.ratio_known_outlier
    ratio_pollution = args.ratio_pollution
    device = args.device

    seed = args.seed
    optimizer_name = args.optimizer_name
    lr = args.lr
    n_epochs = args.n_epochs
    lr_milestone = args.lr_milestone
    batch_size = args.batch_size
    weight_decay = args.weight_decay

    pretrain = args.pretrain
    ae_optimizer_name = args.ae_optimizer_name
    ae_lr = args.ae_lr
    ae_n_epochs = args.ae_n_epochs
    ae_lr_milestone = args.ae_lr_milestone
    ae_batch_size = args.ae_batch_size
    ae_weight_decay = args.ae_weight_decay

    num_threads = args.num_threads
    n_jobs_dataloader = args.n_jobs_dataloader
    normal_class = args.normal_class
    known_outlier_class = args.known_outlier_class
    n_known_outlier_classes = args.n_known_outlier_classes
    del args

    # Get configuration
    cfg = Config(locals().copy())
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print paths
    logger.info('Log file is %s' % log_file)
    logger.info('Data path is %s' % data_path)
    logger.info('Export path is %s' % xp_path)

    # Print experimental setup
    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Ratio of labeled normal train samples: %.2f' % ratio_known_normal)
    logger.info('Ratio of labeled anomalous samples: %.2f' % ratio_known_outlier)
    logger.info('Pollution ratio of unlabeled train data: %.2f' % ratio_pollution)
    if n_known_outlier_classes == 1:
        logger.info('Known anomaly class: %d' % known_outlier_class)
    else:
        logger.info('Number of known anomaly classes: %d' % n_known_outlier_classes)
    logger.info('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print model configuration
    logger.info('Eta-parameter: %.2f' % cfg.settings['eta'])

    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        torch.cuda.manual_seed(cfg.settings['seed'])
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    # Set the number of threads used for parallelizing CPU operations
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    logger.info('Computation device: %s' % device)
    logger.info('Number of threads: %d' % num_threads)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                           ratio_known_normal, ratio_known_outlier, ratio_pollution,
                           random_state=np.random.RandomState(cfg.settings['seed']))
    # Log random sample of known anomaly classes if more than 1 class
    if n_known_outlier_classes > 1:
        logger.info('Known anomaly classes: %s' % (dataset.known_outlier_classes,))

    # Initialize DeepSAD model and set neural network phi
    deepSAD = DeepSAD(cfg.settings['eta'])
    deepSAD.set_network(net_name)

    # If specified, load Deep SAD model (center c, network weights, and possibly autoencoder weights)
    if load_model:
        deepSAD.load_model(model_path=load_model, load_ae=True, map_location=device)
        logger.info('Loading model from %s.' % load_model)

    logger.info('Pretraining: %s' % pretrain)
    if pretrain:
        # Log pretraining details
        logger.info('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
        logger.info('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
        logger.info('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
        logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
        logger.info('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
        logger.info('Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])

        # Pretrain model on dataset (via autoencoder)
        deepSAD.pretrain(dataset,
                         optimizer_name=cfg.settings['ae_optimizer_name'],
                         lr=cfg.settings['ae_lr'],
                         n_epochs=cfg.settings['ae_n_epochs'],
                         lr_milestones=cfg.settings['ae_lr_milestone'],
                         batch_size=cfg.settings['ae_batch_size'],
                         weight_decay=cfg.settings['ae_weight_decay'],
                         device=device,
                         n_jobs_dataloader=n_jobs_dataloader)

        # Save pretraining results
        deepSAD.save_ae_results(export_json=xp_path + '/ae_results.json')

    # Log training details
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

    # Train model on dataset
    deepSAD.train(dataset,
                  optimizer_name=cfg.settings['optimizer_name'],
                  lr=cfg.settings['lr'],
                  n_epochs=cfg.settings['n_epochs'],
                  lr_milestones=cfg.settings['lr_milestone'],
                  batch_size=cfg.settings['batch_size'],
                  weight_decay=cfg.settings['weight_decay'],
                  device=device,
                  n_jobs_dataloader=n_jobs_dataloader)

    # Test model
    deepSAD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    # Save results, model, and configuration
    deepSAD.save_results(export_json=xp_path + '/results.json')
    deepSAD.save_model(export_model=xp_path + '/model.tar')
    cfg.save_config(export_json=xp_path + '/config.json')

    # Plot most anomalous and most normal test samples
    indices, labels, scores = zip(*deepSAD.results['test_scores'])
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    idx_all_sorted = indices[np.argsort(scores)]  # from lowest to highest score
    idx_normal_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # from lowest to highest score

    if dataset_name in ('mnist', 'fmnist', 'cifar10'):

        if dataset_name in ('mnist', 'fmnist'):
            X_all_low = dataset.test_set.data[idx_all_sorted[:32], ...].unsqueeze(1)
            X_all_high = dataset.test_set.data[idx_all_sorted[-32:], ...].unsqueeze(1)
            X_normal_low = dataset.test_set.data[idx_normal_sorted[:32], ...].unsqueeze(1)
            X_normal_high = dataset.test_set.data[idx_normal_sorted[-32:], ...].unsqueeze(1)

        if dataset_name == 'cifar10':
            X_all_low = torch.tensor(np.transpose(dataset.test_set.data[idx_all_sorted[:32], ...], (0, 3, 1, 2)))
            X_all_high = torch.tensor(np.transpose(dataset.test_set.data[idx_all_sorted[-32:], ...], (0, 3, 1, 2)))
            X_normal_low = torch.tensor(np.transpose(dataset.test_set.data[idx_normal_sorted[:32], ...], (0, 3, 1, 2)))
            X_normal_high = torch.tensor(
                np.transpose(dataset.test_set.data[idx_normal_sorted[-32:], ...], (0, 3, 1, 2)))

        plot_images_grid(X_all_low, export_img=xp_path + '/all_low', padding=2)
        plot_images_grid(X_all_high, export_img=xp_path + '/all_high', padding=2)
        plot_images_grid(X_normal_low, export_img=xp_path + '/normals_low', padding=2)
        plot_images_grid(X_normal_high, export_img=xp_path + '/normals_high', padding=2)



if __name__ == '__main__':
    run()
