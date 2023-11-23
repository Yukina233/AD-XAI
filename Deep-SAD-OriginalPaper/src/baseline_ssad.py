import argparse
import os
import time

import click
import torch
import logging
import random
import numpy as np
import cvxopt as co

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from baselines.ssad import SSAD
from datasets.main import load_dataset


################################################################################
# Settings
################################################################################
def run(config=None):
    path_project = '/home/yukina/Missile_Fault_Detection/project/Deep-SAD-OriginalPaper'
    parser = argparse.ArgumentParser(description="Run Deep SAD experiments with specified parameters.")

    # Add arguments to the parser
    parser.add_argument('--dataset_name', type=str, default='cifar10',
                        help='Name of the dataset to load.{mnist, fmnist, cifar10, arrhythmia, cardio, satellite,'
                             ' satimage-2, shuttle, thyroid}')
    parser.add_argument('--xp_path', type=str,
                        default=path_project + '/log/test/log_' + time.strftime('%Y-%m-%d_%H-%M-%S'),
                        help='Export path for logging the experiment.')
    parser.add_argument('--data_path', type=str, default=path_project + '/data', help='Root path of data.')

    parser.add_argument('--load_config', type=str, default=None, help='Config JSON-file path (default: None).')
    parser.add_argument('--load_model', type=str, default=None, help='Model file path (default: None).')
    parser.add_argument('--ratio_known_normal', type=float, default=0,
                        help='Ratio of known (labeled) normal training examples in all train examples.')
    parser.add_argument('--ratio_known_outlier', type=float, default=0,
                        help='Ratio of known (labeled) anomalous training examples in all train examples.')
    parser.add_argument('--ratio_pollution', type=float, default=0.0,
                        help='Pollution ratio of unlabeled training data with unknown (unlabeled) anomalies.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')

    parser.add_argument('--seed', type=int, default=42, help='Set seed. If -1, use randomization.')
    parser.add_argument('--kernel', type=str, default='rbf', help='Kernel for SSAD. Default: rbf.')
    parser.add_argument('--kappa', type=float, default=1.0, help='Kappa hyperparameter for SSAD. Default: 1.0.')
    parser.add_argument('--hybrid', type=bool, default=False,
                        help='Whether to use hybrid SSAD. Default: False.')
    parser.add_argument('--load_ae', type=str, default=None,
                        help='Path to autoencoder weights file. Default: None.')
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
    xp_path = args.xp_path
    if not os.path.exists(xp_path):
        os.makedirs(xp_path)
    else:
        # 如果已经进行了实验，则不继续
        return
    data_path = args.data_path

    load_config = args.load_config
    load_model = args.load_model

    ratio_known_normal = args.ratio_known_normal
    ratio_known_outlier = args.ratio_known_outlier
    ratio_pollution = args.ratio_pollution
    device = args.device

    seed = args.seed
    kernel = args.kernel
    kappa = args.kappa
    hybrid = args.hybrid
    load_ae = args.load_ae

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
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)

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

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print SSAD configuration
    logger.info('SSAD kernel: %s' % cfg.settings['kernel'])
    logger.info('Kappa-paramerter: %.2f' % cfg.settings['kappa'])
    logger.info('Hybrid model: %s' % cfg.settings['hybrid'])

    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        co.setseed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        torch.cuda.manual_seed(cfg.settings['seed'])
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Use 'cpu' as device for SSAD
    device = 'cpu'
    torch.multiprocessing.set_sharing_strategy('file_system')  # fix multiprocessing issue for ubuntu
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    dataset = load_dataset(dataset_name, data_path, normal_class, known_outlier_class, n_known_outlier_classes,
                           ratio_known_normal, ratio_known_outlier, ratio_pollution,
                           random_state=np.random.RandomState(cfg.settings['seed']))
    # Log random sample of known anomaly classes if more than 1 class
    if n_known_outlier_classes > 1:
        logger.info('Known anomaly classes: %s' % (dataset.known_outlier_classes,))

    # Initialize SSAD model
    ssad = SSAD(kernel=cfg.settings['kernel'], kappa=cfg.settings['kappa'], hybrid=cfg.settings['hybrid'])

    # If specified, load model parameters from already trained model
    if load_model:
        ssad.load_model(import_path=load_model, device=device)
        logger.info('Loading model from %s.' % load_model)

    # If specified, load model autoencoder weights for a hybrid approach
    if hybrid and load_ae is not None:
        ssad.load_ae(dataset_name, model_path=load_ae)
        logger.info('Loaded pretrained autoencoder for features from %s.' % load_ae)

    # Train model on dataset
    ssad.train(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    # Test model
    ssad.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

    # Save results and configuration
    ssad.save_results(export_json=xp_path + '/results.json')
    cfg.save_config(export_json=xp_path + '/config.json')

    # Plot most anomalous and most normal test samples
    indices, labels, scores = zip(*ssad.results['test_scores'])
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
