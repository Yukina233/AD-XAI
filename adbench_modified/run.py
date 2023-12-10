import logging

import numpy as np
import pandas as pd
import itertools
from itertools import product

import torch
from torch import nn
from tqdm import tqdm
import time
import gc
import os
from keras import backend as K

from adbench_modified.datasets.data_generator import DataGenerator
from adbench_modified.myutils import Utils

path_project = '/home/yukina/Missile_Fault_Detection/project/test'


class RunPipeline:
    def __init__(self, suffix: str = None, mode: str = 'rla', parallel: str = None,
                 generate_duplicates=True, n_samples_threshold=1000, seed=3,
                 realistic_synthetic_mode: str = None,
                 noise_type=None,
                 path_result='./results'):
        """
        :param suffix: saved file suffix (including the model performance result and model weights)
        :param mode: rla or nla —— ratio of labeled anomalies or number of labeled anomalies
        :param parallel: unsupervise, semi-supervise or supervise, choosing to parallelly run the code
        :param generate_duplicates: whether to generate duplicated samples when sample size is too small
        :param n_samples_threshold: threshold for generating the above duplicates, if generate_duplicates is False, then datasets with sample size smaller than n_samples_threshold will be dropped
        :param realistic_synthetic_mode: local, global, dependency or cluster —— whether to generate the realistic synthetic anomalies to test different algorithms
        :param noise_type: duplicated_anomalies, irrelevant_features or label_contamination —— whether to test the model robustness
        """

        self.seed = seed
        # utils function
        self.utils = Utils()

        self.mode = mode
        self.parallel = parallel

        # global parameters
        self.generate_duplicates = generate_duplicates
        self.n_samples_threshold = n_samples_threshold

        self.realistic_synthetic_mode = realistic_synthetic_mode
        self.noise_type = noise_type

        self.path_result = path_result

        # the suffix of all saved files
        self.suffix = suffix + '_' + 'type(' + str(realistic_synthetic_mode) + ')_' + 'noise(' + str(noise_type) + ')_' \
                      + self.parallel

        # data generator instantiation
        self.data_generator = DataGenerator(generate_duplicates=self.generate_duplicates,
                                            n_samples_threshold=self.n_samples_threshold)

        # ratio of labeled anomalies
        if self.noise_type is not None:
            self.rla_list = [0.00]
        else:
            self.rla_list = [0.00, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]

        # number of labeled anomalies
        self.nla_list = [0, 1, 5, 10, 25, 50, 75, 100]
        # seed list
        self.seed_list = list(np.arange(self.seed) + 1)

        if self.noise_type is None:
            pass

        elif self.noise_type == 'duplicated_anomalies':
            self.noise_params_list = [1, 2, 3, 4, 5, 6]

        elif self.noise_type == 'irrelevant_features':
            self.noise_params_list = [0.00, 0.01, 0.05, 0.10, 0.25, 0.50]

        elif self.noise_type == 'label_contamination':
            self.noise_params_list = [0.00, 0.01, 0.05, 0.10, 0.25, 0.50]

        else:
            raise NotImplementedError

        # model_dict (model_name: clf)
        self.model_dict = {}

        # unsupervised algorithms
        if self.parallel == 'unsupervise':
            from adbench_modified.baseline.PyOD import PYOD
            from adbench_modified.baseline.DAGMM.run import DAGMM
            from adbench_modified.baseline.DeepSAD.src.run import DeepSAD

            # from pyod
            for _ in ['IForest', 'OCSVM', 'CBLOF', 'COF', 'COPOD', 'ECOD', 'FeatureBagging', 'HBOS', 'KNN', 'LODA',
                      'LOF', 'LSCP', 'MCD', 'PCA', 'SOD', 'SOGAAL', 'MOGAAL', 'DeepSVDD']:
                self.model_dict[_] = PYOD

            # DAGMM
            self.model_dict['DAGMM'] = DAGMM
            # 测试DeepSAD方法
            self.model_dict['DeepSAD'] = DeepSAD

        # semi-supervised algorithms
        elif self.parallel == 'semi-supervise':
            from adbench_modified.baseline.PyOD import PYOD
            from adbench_modified.baseline.GANomaly.run import GANomaly
            from adbench_modified.baseline.DeepSAD.src.run import DeepSAD
            from adbench_modified.baseline.REPEN.run import REPEN
            from adbench_modified.baseline.DevNet.run import DevNet
            from adbench_modified.baseline.PReNet.run import PReNet
            from adbench_modified.baseline.FEAWAD.run import FEAWAD

            self.model_dict = {'GANomaly': GANomaly,
                               'DeepSAD': DeepSAD,
                               'REPEN': REPEN,
                               'DevNet': DevNet,
                               'PReNet': PReNet,
                               'FEAWAD': FEAWAD,
                               'XGBOD': PYOD}

        # fully-supervised algorithms
        elif self.parallel == 'supervise':
            from adbench_modified.baseline.Supervised import supervised
            from adbench_modified.baseline.FTTransformer.run import FTTransformer

            # from sklearn
            for _ in ['LR', 'NB', 'SVM', 'MLP', 'RF', 'LGB', 'XGB', 'CatB']:
                self.model_dict[_] = supervised
            # ResNet and FTTransformer for tabular data
            for _ in ['ResNet', 'FTTransformer']:
                self.model_dict[_] = FTTransformer

        else:
            raise NotImplementedError

        # We remove the following model for considering the computational cost
        for _ in ['SOGAAL', 'MOGAAL', 'LSCP', 'MCD', 'FeatureBagging']:
            if _ in self.model_dict.keys():
                self.model_dict.pop(_)

    # dataset filter for delelting those datasets that do not satisfy the experimental requirement
    def dataset_filter(self):
        # dataset list in the current folder
        dataset_list_org = list(itertools.chain(*self.data_generator.generate_dataset_list()))

        dataset_list, dataset_size = [], []
        for dataset in dataset_list_org:
            # only use CV datasets
            if self.notin_CV(dataset):
                continue

            add = True
            for seed in self.seed_list:
                self.data_generator.seed = seed
                self.data_generator.dataset = dataset
                data = self.data_generator.generator(la=1.00, at_least_one_labeled=True)

                if not self.generate_duplicates and len(data['y_train']) + len(
                        data['y_test']) < self.n_samples_threshold:
                    add = False

                else:
                    if self.mode == 'nla' and sum(data['y_train']) >= self.nla_list[-1]:
                        pass

                    elif self.mode == 'rla' and sum(data['y_train']) > 0:
                        pass

                    else:
                        add = False

            # remove high-dimensional CV and NLP datasets if generating synthetic anomalies or robustness test
            if self.realistic_synthetic_mode is not None or self.noise_type is not None:
                if self.isin_NLPCV(dataset):
                    add = False

            if add:
                dataset_list.append(dataset)
                dataset_size.append(len(data['y_train']) + len(data['y_test']))
            else:
                print(f"remove the dataset {dataset}")

        # sort datasets by their sample size
        dataset_list = [dataset_list[_] for _ in np.argsort(np.array(dataset_size))]

        return dataset_list

    def notin_CV(self, dataset):
        if dataset is None:
            return True
        else:
            CV_list = ['MVTec-AD']

            return not any([_ in dataset for _ in CV_list])

    # whether the dataset in the NLP / CV dataset
    # currently we have 5 NLP datasets and 5 CV datasets
    def isin_NLPCV(self, dataset):
        if dataset is None:
            return False
        else:
            NLPCV_list = ['agnews', 'amazon', 'imdb', 'yelp', '20news',
                          'MNIST-C', 'FashionMNIST', 'CIFAR10', 'SVHN', 'MVTec-AD']

            return any([_ in dataset for _ in NLPCV_list])

    # model fitting function
    def model_fit(self):
        try:
            # model initialization, if model weights are saved, the save_suffix should be specified
            if self.model_name in ['DevNet', 'FEAWAD', 'REPEN']:
                self.clf = self.clf(seed=self.seed, model_name=self.model_name, save_suffix=self.suffix)
            else:
                self.clf = self.clf(seed=self.seed, model_name=self.model_name)

        except Exception as error:
            print(f'Error in model initialization. Model:{self.model_name}, Error: {error}')
            pass

        try:
            # fitting
            start_time = time.time()
            self.clf = self.clf.fit(X_train=self.data['X_train'], y_train=self.data['y_train'])
            end_time = time.time()
            time_fit = end_time - start_time

            # predicting score (inference)
            start_time = time.time()
            if self.model_name == 'DAGMM':
                score_test = self.clf.predict_score(self.data['X_train'], self.data['X_test'])
            else:
                score_test = self.clf.predict_score(self.data['X_test'])
            end_time = time.time()
            time_inference = end_time - start_time

            # performance
            result = self.utils.metric(y_true=self.data['y_test'], y_score=score_test, pos_label=1)

            K.clear_session()
            print(f"Model: {self.model_name}, AUC-ROC: {result['aucroc']}, AUC-PR: {result['aucpr']}")

            del self.clf
            gc.collect()

        except Exception as error:
            print(f'Error in model fitting. Model:{self.model_name}, Error: {error}')
            time_fit, time_inference = None, None
            result = {'aucroc': np.nan, 'aucpr': np.nan}
            pass

        return time_fit, time_inference, result

    # run the experiments in ADBench_modified
    def run(self, dataset=None, clf=None):
        if dataset is None:
            #  filteting dataset that does not meet the experimental requirements
            dataset_list = self.dataset_filter()
            X, y = None, None
        else:
            isinstance(dataset, dict)
            dataset_list = [None]
            X = dataset['X']
            y = dataset['y']

        # experimental parameters
        if self.mode == 'nla':
            if self.noise_type is not None:
                experiment_params = list(product(dataset_list, self.nla_list, self.noise_params_list, self.seed_list))
            else:
                experiment_params = list(product(dataset_list, self.nla_list, self.seed_list))
        else:
            if self.noise_type is not None:
                experiment_params = list(product(dataset_list, self.rla_list, self.noise_params_list, self.seed_list))
            else:
                experiment_params = list(product(dataset_list, self.rla_list, self.seed_list))

        print(f'{len(dataset_list)} datasets, {len(self.model_dict.keys())} models')

        # save the results
        print(f"Experiment results are saved at: {self.path_result}")
        os.makedirs(self.path_result, exist_ok=True)
        columns = list(self.model_dict.keys()) if clf is None else ['Customized']
        df_AUCROC = pd.DataFrame(data=None, index=experiment_params, columns=columns)
        df_AUCPR = pd.DataFrame(data=None, index=experiment_params, columns=columns)
        df_time_fit = pd.DataFrame(data=None, index=experiment_params, columns=columns)
        df_time_inference = pd.DataFrame(data=None, index=experiment_params, columns=columns)

        results = []
        for i, params in tqdm(enumerate(experiment_params)):
            if self.noise_type is not None:
                dataset, la, noise_param, self.seed = params
            else:
                dataset, la, self.seed = params

            if self.parallel == 'unsupervise' and la != 0.0 and self.noise_type is None:
                # if self.parallel == 'unsupervise' and self.noise_type is None:
                continue

            # We only run one time on CV / NLP datasets for considering computational cost
            # The final results are the average performance on different classes
            if self.isin_NLPCV(dataset) and self.seed > 1:
                continue

            # generate data
            self.data_generator.seed = self.seed
            self.data_generator.dataset = dataset

            try:
                if self.noise_type == 'duplicated_anomalies':
                    self.data = self.data_generator.generator(la=la, at_least_one_labeled=True, X=X, y=y,
                                                              realistic_synthetic_mode=self.realistic_synthetic_mode,
                                                              noise_type=self.noise_type, duplicate_times=noise_param)
                elif self.noise_type == 'irrelevant_features':
                    self.data = self.data_generator.generator(la=la, at_least_one_labeled=True, X=X, y=y,
                                                              realistic_synthetic_mode=self.realistic_synthetic_mode,
                                                              noise_type=self.noise_type, noise_ratio=noise_param)
                elif self.noise_type == 'label_contamination':
                    self.data = self.data_generator.generator(la=la, at_least_one_labeled=True, X=X, y=y,
                                                              realistic_synthetic_mode=self.realistic_synthetic_mode,
                                                              noise_type=self.noise_type, noise_ratio=noise_param)
                else:
                    self.data = self.data_generator.generator(la=la, at_least_one_labeled=True, X=X, y=y,
                                                              realistic_synthetic_mode=self.realistic_synthetic_mode)

            except Exception as error:
                print(f'Error when generating data: {error}')
                pass
                continue

            if clf is None:
                for model_name in tqdm(self.model_dict.keys()):
                    self.model_name = model_name
                    self.clf = self.model_dict[self.model_name]

                    # fit and test model
                    time_fit, time_inference, metrics = self.model_fit()
                    results.append([params, model_name, metrics, time_fit, time_inference])
                    print(f'Current experiment parameters: {params}, model: {model_name}, metrics: {metrics}, '
                          f'fitting time: {time_fit}, inference time: {time_inference}')

                    # store and save the result (AUC-ROC, AUC-PR and runtime / inference time)
                    df_AUCROC[model_name].iloc[i] = metrics['aucroc']
                    df_AUCPR[model_name].iloc[i] = metrics['aucpr']
                    df_time_fit[model_name].iloc[i] = time_fit
                    df_time_inference[model_name].iloc[i] = time_inference

                    df_AUCROC.to_csv(os.path.join(self.path_result, 'AUCROC_' + self.suffix + '.csv'), index=True)
                    df_AUCPR.to_csv(os.path.join(self.path_result, 'AUCPR_' + self.suffix + '.csv'), index=True)
                    df_time_fit.to_csv(os.path.join(self.path_result, 'Time(fit)_' + self.suffix + '.csv'), index=True)
                    df_time_inference.to_csv(os.path.join(self.path_result, 'Time(inference)_' + self.suffix + '.csv'),
                                             index=True)

            else:
                self.clf = clf
                self.model_name = 'Customized'
                # fit and test model
                time_fit, time_inference, metrics = self.model_fit()
                results.append([params, self.model_name, metrics, time_fit, time_inference])
                print(f'Current experiment parameters: {params}, model: {self.model_name}, metrics: {metrics}, '
                      f'fitting time: {time_fit}, inference time: {time_inference}')

                # store and save the result (AUC-ROC, AUC-PR and runtime / inference time)
                df_AUCROC[self.model_name].iloc[i] = metrics['aucroc']
                df_AUCPR[self.model_name].iloc[i] = metrics['aucpr']
                df_time_fit[self.model_name].iloc[i] = time_fit
                df_time_inference[self.model_name].iloc[i] = time_inference

                df_AUCROC.to_csv(os.path.join(self.path_result, 'AUCROC_' + self.suffix + '.csv'), index=True)
                df_AUCPR.to_csv(os.path.join(self.path_result, 'AUCPR_' + self.suffix + '.csv'), index=True)
                df_time_fit.to_csv(os.path.join(self.path_result, 'Time(fit)_' + self.suffix + '.csv'), index=True)
                df_time_inference.to_csv(os.path.join(self.path_result, 'Time(inference)_' + self.suffix + '.csv'),
                                         index=True)

        return results

    def run_universum(self, target_dataset_name='MVTec-AD_bottle', path_datasets=None, clf=None, use_preprocess=False,
                      universum_params=None):

        # 加载所有的数据集
        datasets = {}
        for filename in os.listdir(path_datasets):
            if filename.endswith('.npz'):
                data_name = filename.split('.')[0]
                data = np.load(os.path.join(path_datasets, filename), allow_pickle=True)
                datasets[data_name] = data

        # 选择一个目标数据集
        target_dataset = datasets[target_dataset_name]

        # 划分训练集和测试集
        # isinstance(target_dataset, dict)
        dataset_list = [None]
        X = target_dataset['X']
        y = target_dataset['y']

        # experimental parameters
        if self.mode == 'nla':
            if self.noise_type is not None:
                experiment_params = list(product(dataset_list, self.nla_list, self.noise_params_list, self.seed_list))
            else:
                experiment_params = list(product(dataset_list, self.nla_list, self.seed_list))
        else:
            if self.noise_type is not None:
                experiment_params = list(product(dataset_list, self.rla_list, self.noise_params_list, self.seed_list))
            else:
                experiment_params = list(product(dataset_list, self.rla_list, self.seed_list))

        print(f'{len(dataset_list)} datasets, {len(self.model_dict.keys())} models')

        # save the results
        print(f"Experiment results are saved at: {self.path_result}")
        os.makedirs(self.path_result, exist_ok=True)
        columns = list(self.model_dict.keys()) if clf is None else ['Customized']
        df_AUCROC = pd.DataFrame(data=None, index=experiment_params, columns=columns)
        df_AUCPR = pd.DataFrame(data=None, index=experiment_params, columns=columns)
        df_time_fit = pd.DataFrame(data=None, index=experiment_params, columns=columns)
        df_time_inference = pd.DataFrame(data=None, index=experiment_params, columns=columns)

        results = []
        for i, params in tqdm(enumerate(experiment_params)):
            if self.noise_type is not None:
                dataset, la, noise_param, self.seed = params
            else:
                dataset, la, self.seed = params

            if self.parallel == 'unsupervise' and la != 0.0 and self.noise_type is None:
                # if self.parallel == 'unsupervise' and self.noise_type is None:
                continue

            # generate data
            self.data_generator.seed = self.seed
            self.data_generator.dataset = dataset

            try:
                if self.noise_type == 'duplicated_anomalies':
                    self.data = self.data_generator.generator(la=la, at_least_one_labeled=True, X=X, y=y,
                                                              realistic_synthetic_mode=self.realistic_synthetic_mode,
                                                              noise_type=self.noise_type, duplicate_times=noise_param)
                elif self.noise_type == 'irrelevant_features':
                    self.data = self.data_generator.generator(la=la, at_least_one_labeled=True, X=X, y=y,
                                                              realistic_synthetic_mode=self.realistic_synthetic_mode,
                                                              noise_type=self.noise_type, noise_ratio=noise_param)
                elif self.noise_type == 'label_contamination':
                    self.data = self.data_generator.generator(la=la, at_least_one_labeled=True, X=X, y=y,
                                                              realistic_synthetic_mode=self.realistic_synthetic_mode,
                                                              noise_type=self.noise_type, noise_ratio=noise_param)
                else:
                    self.data = self.data_generator.generator(la=la, at_least_one_labeled=True, X=X, y=y,
                                                              realistic_synthetic_mode=self.realistic_synthetic_mode)

            except Exception as error:
                print(f'Error when generating data: {error}')
                pass
                continue

            # 假设npz文件中有 'train' 和 'test' 键
            # 如果数据集结构不同，需要调整代码以匹配实际结构
            X_train = self.data['X_train']
            y_train = self.data['y_train']

            # 对于训练集中的每个样本，从其他数据集中随机抽取一个样本
            augmented_train_samples = {
                'X': [],
                'y': []
            }
            for sample in X_train:
                for j in range(universum_params['aux_size']):
                    # 从除了目标数据集之外的其他数据集中随机选择一个数据集
                    other_datasets = [name for name in datasets if name != target_dataset_name]
                    random_dataset_name = np.random.choice(other_datasets)
                    random_dataset = datasets[random_dataset_name]

                    # 首先找出所有 y == 0 的索引
                    indices_of_y_equals_0 = np.where(random_dataset['y'] == 0)[0]
                    # 然后只从这些索引对应的 X 中抽取一个样本
                    sample_indices = np.random.choice(indices_of_y_equals_0, 1)
                    random_sample = random_dataset['X'][sample_indices][0]

                    # 将随机抽取的样本与目标样本结合
                    from auxiliary_data_AD.universum_generate import get_universum
                    augmented_sample = get_universum(sample, random_sample, universum_params['aug_type'],
                                                     universum_params['lamda'])
                    augmented_train_samples['X'].append(augmented_sample)
                    augmented_train_samples['y'].append(1)

            # 现在 'augmented_train_samples' 包含了原始训练样本和随机抽取的样本对
            # 接下来可以根据实际需求使用这些样本对进行训练模型等操作
            self.data['X_train'] = np.concatenate((self.data['X_train'], np.stack(augmented_train_samples['X'])))
            self.data['y_train'] = np.concatenate((self.data['y_train'], np.stack(augmented_train_samples['y'])))

            if use_preprocess:
                pretrain_encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
                pretrain_encoder = nn.Sequential(*list(pretrain_encoder.children())[:-1])
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                pretrain_encoder.to(device)
                pretrain_encoder.eval()
                with torch.no_grad():
                    X_t = self.data['X_train']
                    X_t = torch.tensor(X_t).to(device)
                    X_v = self.data['X_test']
                    X_v = torch.tensor(X_v).to(device)
                    if X_t.shape[1] == 1:
                        X_t = X_t.repeat(1, 3, 1, 1)
                    if X_v.shape[1] == 1:
                        X_v = X_v.repeat(1, 3, 1, 1)
                    X_t_e = pretrain_encoder(X_t).squeeze().cpu().numpy()
                    X_v_e = pretrain_encoder(X_v).squeeze().cpu().numpy()

                self.data['X_train'] = X_t_e
                self.data['X_test'] = X_v_e

                del pretrain_encoder

            if clf is None:
                print(f'Clf is None')
                assert False

            self.clf = clf
            self.model_name = 'Customized'

            # fit and test model
            time_fit, time_inference, metrics = self.model_fit()
            results.append([params, self.model_name, metrics, time_fit, time_inference])
            print(f'Current experiment parameters: {params}, model: {self.model_name}, metrics: {metrics}, '
                  f'fitting time: {time_fit}, inference time: {time_inference}')

            # store and save the result (AUC-ROC, AUC-PR and runtime / inference time)
            df_AUCROC[self.model_name].iloc[i] = metrics['aucroc']
            df_AUCPR[self.model_name].iloc[i] = metrics['aucpr']
            df_time_fit[self.model_name].iloc[i] = time_fit
            df_time_inference[self.model_name].iloc[i] = time_inference

            df_AUCROC.to_csv(os.path.join(self.path_result, 'AUCROC_' + self.suffix + '.csv'), index=True)
            df_AUCPR.to_csv(os.path.join(self.path_result, 'AUCPR_' + self.suffix + '.csv'), index=True)
            df_time_fit.to_csv(os.path.join(self.path_result, 'Time(fit)_' + self.suffix + '.csv'), index=True)
            df_time_inference.to_csv(os.path.join(self.path_result, 'Time(inference)_' + self.suffix + '.csv'),
                                     index=True)

        return results


# For test use
if __name__ == '__main__':
    seed = 1
    aug_type = 'cutmix'
    lamda = 0.95
    aux_size = 1
    use_preprocess = True

    path_project = '/home/yukina/Missile_Fault_Detection/project'

    category = 'MVTec-AD_grid'
    dataset_path = os.path.join(path_project, f'data/mvtec_ad/{category}.npz')

    path_save = os.path.join(path_project, 'auxiliary_data_AD/log/test',
                             f'DeepSAD_{aug_type},lamda={lamda},aux_size={aux_size}', category)
    os.makedirs(path_save, exist_ok=True)  # 创建结果文件夹

    # 实例化并运行pipeline
    from adbench_modified.baseline.DeepSAD.src.run import DeepSAD

    pipeline = RunPipeline(suffix='DeepSAD', parallel='unsupervise', n_samples_threshold=200, seed=seed,
                           realistic_synthetic_mode=None,
                           noise_type=None, path_result=path_save)
    results = pipeline.run_universum(clf=DeepSAD, target_dataset_name=category,
                                     path_datasets=path_project + '/data/mvtec_ad',
                                     use_preprocess=use_preprocess,
                                     universum_params={'aug_type': aug_type, 'lamda': lamda, 'aux_size': aux_size})
