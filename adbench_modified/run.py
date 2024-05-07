import logging

import numpy as np
import pandas as pd
import itertools
from itertools import product

import torch
from sklearn.metrics import precision_recall_curve
from torch import nn
from tqdm import tqdm
import time
import gc
import os
from keras import backend as K

from adbench_modified.datasets.data_generator import DataGenerator
from adbench_modified.myutils import Utils
from auxiliary_data_AD.universum_generate import get_universum

path_project = '/home/yukina/Missile_Fault_Detection/project'


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
        self.seed_list = [1, 2, 3]

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
    def model_fit(self, load_model=None):
        try:
            # model initialization, if model weights are saved, the save_suffix should be specified
            if self.model_name in ['DevNet', 'FEAWAD', 'REPEN']:
                self.clf = self.clf(seed=self.seed, model_name=self.model_name, save_suffix=self.suffix)
            else:
                if load_model is not None:
                    self.clf = self.clf(seed=self.seed, model_name=self.model_name, load_model=load_model)
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

        except Exception as error:
            print(f'Error in model fitting. Model:{self.model_name}, Error: {error}')
            time_fit, time_inference = None, None
            pass

        return time_fit

    def model_test(self):
        try:
            # predicting score (inference)
            start_time = time.time()
            if self.model_name == 'DAGMM':
                score_test, outputs = self.clf.predict_score(self.data['X_train'], self.data['X_test'])
            else:
                score_test, outputs = self.clf.predict_score(self.data['X_test'])

            # 计算故障检测率和虚警率
            score_train, outputs = self.clf.predict_score(self.data['X_train'])
            thresholds = np.percentile(score_train, 100 * (1 - sum(self.data['y_train']) / len(self.data['y_train'])))
            # thresholds = thresholds * 1.6
            id_anomaly_pred = np.where(score_test > thresholds)[0]
            id_normal_pred = np.where(score_test <= thresholds)[0]

            tp = np.size(np.where(self.data['y_test'][id_anomaly_pred] == 1)[0], 0)
            fp = np.size(np.where(self.data['y_test'][id_anomaly_pred] == 0)[0], 0)
            fn = np.size(np.where(self.data['y_test'][id_normal_pred] == 1)[0], 0)

            FDR = tp / (tp + fn)
            if tp + fp == 0:
                FAR = 0
            else:
                FAR = fp / (tp + fp)

            end_time = time.time()
            time_inference = end_time - start_time

            # performance
            result = self.utils.metric(y_true=self.data['y_test'], y_score=score_test, pos_label=1)
            result['FDR'] = FDR
            result['FAR'] = FAR

            precision, recall, _ = precision_recall_curve(self.data['y_test'], score_test)
            precision_threshold = 0.999
            recall_at_threshold = recall[np.where(precision >= precision_threshold)[0][0]]
            recall_threshold = 0.999
            precision_at_threshold = precision[np.where(recall >= recall_threshold)[0][-1]]

            result['FDR_at_threshold'] = recall_at_threshold
            result['FAR_at_threshold'] = 1 - precision_at_threshold

            K.clear_session()
            print(f"Model: {self.model_name}, AUC-ROC: {result['aucroc']}, AUC-PR: {result['aucpr']}")

        except Exception as error:
            print(f'Error in model testing. Model:{self.model_name}, Error: {error}')
            time_inference = None
            result = {'aucroc': np.nan, 'aucpr': np.nan}
            pass

        return time_inference, result

    # run the experiments in ADBench_modified
    def run(self, dataset=None, clf=None):
        if dataset is None:
            #  filteting dataset that does not meet the experimental requirements
            dataset_list = self.dataset_filter()
            X, y, X_train, y_train, X_test, y_test = None, None, None, None, None, None
        else:
            isinstance(dataset, dict)
            dataset_list = [None]
            X = dataset['X']
            y = dataset['y']
            X_train = dataset['X_train']
            y_train = dataset['y_train']
            X_test = dataset['X_test']
            y_test = dataset['y_test']

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
        df_FDR = pd.DataFrame(data=None, index=experiment_params, columns=columns)
        df_FAR = pd.DataFrame(data=None, index=experiment_params, columns=columns)
        df_FDR_at_threshold = pd.DataFrame(data=None, index=experiment_params, columns=columns)
        df_FAR_at_threshold = pd.DataFrame(data=None, index=experiment_params, columns=columns)
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
                                                              X_train=X_train, y_train=y_train, X_test=X_test,
                                                              y_test=y_test,
                                                              realistic_synthetic_mode=self.realistic_synthetic_mode,
                                                              noise_type=self.noise_type, duplicate_times=noise_param)
                elif self.noise_type == 'irrelevant_features':
                    self.data = self.data_generator.generator(la=la, at_least_one_labeled=True, X=X, y=y,
                                                              X_train=X_train, y_train=y_train, X_test=X_test,
                                                              y_test=y_test,
                                                              realistic_synthetic_mode=self.realistic_synthetic_mode,
                                                              noise_type=self.noise_type, noise_ratio=noise_param)
                elif self.noise_type == 'label_contamination':
                    self.data = self.data_generator.generator(la=la, at_least_one_labeled=True, X=X, y=y,
                                                              X_train=X_train, y_train=y_train, X_test=X_test,
                                                              y_test=y_test,
                                                              realistic_synthetic_mode=self.realistic_synthetic_mode,
                                                              noise_type=self.noise_type, noise_ratio=noise_param)
                else:
                    self.data = self.data_generator.generator(la=la, at_least_one_labeled=True, X=X, y=y,
                                                              X_train=X_train, y_train=y_train, X_test=X_test,
                                                              y_test=y_test,
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
                train_once = True
                if train_once is True:
                    if not os.path.exists(os.path.join(path_project, f'adversarial_ensemble_AD/models',
                                                       f'DeepSAD_seed={self.seed}.pth')):
                        time_fit = self.model_fit()
                        self.clf.deepSAD.save_model(export_model=os.path.join(path_project, 'adversarial_ensemble_AD/models',
                                                                      f'DeepSAD_seed={self.seed}.pth'), save_ae=True)
                    else:
                        time_fit = self.model_fit(load_model=os.path.join(path_project, 'adversarial_ensemble_AD/models',
                                                                    f'DeepSAD_seed={self.seed}.pth'))
                else:
                    time_fit = self.model_fit()

                time_inference, metrics = self.model_test()
                results.append([params, self.model_name, metrics, time_fit, time_inference])
                print(f'Current experiment parameters: {params}, model: {self.model_name}, metrics: {metrics}, '
                      f'fitting time: {time_fit}, inference time: {time_inference}')

                # store and save the result (AUC-ROC, AUC-PR and runtime / inference time)
                df_AUCROC[self.model_name].iloc[i] = metrics['aucroc']
                df_AUCPR[self.model_name].iloc[i] = metrics['aucpr']
                df_FDR[self.model_name].iloc[i] = metrics['FDR']
                df_FAR[self.model_name].iloc[i] = metrics['FAR']
                df_FDR_at_threshold[self.model_name].iloc[i] = metrics['FDR_at_threshold']
                df_FAR_at_threshold[self.model_name].iloc[i] = metrics['FAR_at_threshold']
                df_time_fit[self.model_name].iloc[i] = time_fit
                df_time_inference[self.model_name].iloc[i] = time_inference

                df_AUCROC.to_csv(os.path.join(self.path_result, 'AUCROC_' + self.suffix + '.csv'), index=True)
                df_AUCPR.to_csv(os.path.join(self.path_result, 'AUCPR_' + self.suffix + '.csv'), index=True)
                df_FDR.to_csv(os.path.join(self.path_result, 'FDR_' + self.suffix + '.csv'), index=True)
                df_FAR.to_csv(os.path.join(self.path_result, 'FAR_' + self.suffix + '.csv'), index=True)
                df_FDR_at_threshold.to_csv(os.path.join(self.path_result, 'FDR_at_threshold_' + self.suffix + '.csv'),
                                             index=True)
                df_FAR_at_threshold.to_csv(os.path.join(self.path_result, 'FAR_at_threshold_' + self.suffix + '.csv'),
                                           index=True)
                df_time_fit.to_csv(os.path.join(self.path_result, 'Time(fit)_' + self.suffix + '.csv'), index=True)
                df_time_inference.to_csv(os.path.join(self.path_result, 'Time(inference)_' + self.suffix + '.csv'),
                                         index=True)

                # save the Deep SAD model
                torch.save(self.clf.deepSAD.ae_net,
                           os.path.join(self.path_result, 'model_ae_net' + '.pth'))
                torch.save(self.clf.deepSAD.net,
                           os.path.join(self.path_result, 'model_net' + '.pth'))
                np.save(os.path.join(self.path_result, 'c' + '.npy'), self.clf.deepSAD.c)

                print_score = True
                if print_score is True:
                    result_data = {'scores': metrics['scores'], 'labels': metrics['labels']}
                    df_results = pd.DataFrame(data=result_data)
                    df_results.to_csv(os.path.join(self.path_result, 'results.csv'), index=False)

                del self.clf
                gc.collect()
        return result_data

    def run_universum(self, target_dataset_name='MVTec-AD_bottle', path_datasets=None, clf=None, use_preprocess=False,
                      universum_params=None, preprocess_params=None):

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
        # X_train = None
        # y_train = None
        # X_test = None
        # y_test = None
        X_train = target_dataset['X_train']
        y_train = target_dataset['y_train']
        X_test = target_dataset['X_test']
        y_test = target_dataset['y_test']

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
        for i, params in tqdm(enumerate(experiment_params), desc='Seed'):
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
                                                              X_train=X_train, y_train=y_train, X_test=X_test,
                                                              y_test=y_test,
                                                              realistic_synthetic_mode=self.realistic_synthetic_mode,
                                                              noise_type=self.noise_type, duplicate_times=noise_param)
                elif self.noise_type == 'irrelevant_features':
                    self.data = self.data_generator.generator(la=la, at_least_one_labeled=True, X=X, y=y,
                                                              X_train=X_train, y_train=y_train, X_test=X_test,
                                                              y_test=y_test,
                                                              realistic_synthetic_mode=self.realistic_synthetic_mode,
                                                              noise_type=self.noise_type, noise_ratio=noise_param)
                elif self.noise_type == 'label_contamination':
                    self.data = self.data_generator.generator(la=la, at_least_one_labeled=True, X=X, y=y,
                                                              X_train=X_train, y_train=y_train, X_test=X_test,
                                                              y_test=y_test,
                                                              realistic_synthetic_mode=self.realistic_synthetic_mode,
                                                              noise_type=self.noise_type, noise_ratio=noise_param)
                else:
                    self.data = self.data_generator.generator(la=la, at_least_one_labeled=True, X=X, y=y,
                                                              X_train=X_train, y_train=y_train, X_test=X_test,
                                                              y_test=y_test,
                                                              realistic_synthetic_mode=self.realistic_synthetic_mode)

            except Exception as error:
                print(f'Error when generating data: {error}')
                pass
                continue

            output_path = os.path.join(path_project, 'auxiliary_data_AD/aug_data',
                                       f'n_samples_threshold={self.n_samples_threshold},aug_type={universum_params["aug_type"]},aux_size={universum_params["aux_size"]},lamda={universum_params["lamda"]},seed={self.seed}')
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            npz_file = os.path.join(output_path, target_dataset_name + '.npz')
            if os.path.exists(npz_file):
                npz_data = np.load(npz_file, allow_pickle=True)
                self.data['X_train'] = npz_data['X_train']
                self.data['y_train'] = npz_data['y_train']
            else:
                # 假设npz文件中有 'train' 和 'test' 键
                # 如果数据集结构不同，需要调整代码以匹配实际结构
                X_train = self.data['X_train']
                y_train = self.data['y_train']

                # 预先分配内存
                total_samples = len(X_train) * universum_params['aux_size']
                augmented_train_samples = {
                    'X': [None] * total_samples,
                    'y': [1] * total_samples
                }

                other_datasets = [name for name in datasets if name != target_dataset_name]
                # 为每个非目标数据集预先计算 y == 0 的索引
                indices_of_y_equals_0 = {
                    name: np.where(datasets[name]['y'] == 0)[0]
                    for name in other_datasets
                }
                # 对于每个样本，从其他数据集中随机抽取一个样本
                sample_counter = 0
                for sample in tqdm(X_train, desc='Generate auxiliary data'):
                    for _ in range(universum_params['aux_size']):
                        # 随机选择一个数据集
                        random_dataset_name = np.random.choice(other_datasets)
                        # 从预先计算好的索引中随机选择一个
                        sample_indices = np.random.choice(indices_of_y_equals_0[random_dataset_name], 1)
                        random_sample = datasets[random_dataset_name]['X'][sample_indices][0]

                        # 结合样本
                        augmented_sample = get_universum(sample, random_sample, universum_params['aug_type'],
                                                         universum_params['lamda'])

                        # 填充预先分配的数组
                        augmented_train_samples['X'][sample_counter] = augmented_sample
                        # 'y'的值已经在预分配时设置为1，所以这里不需要再次设置
                        sample_counter += 1

                # 现在 'augmented_train_samples' 包含了原始训练样本和随机抽取的样本对
                self.data['X_train'] = np.concatenate((self.data['X_train'], np.array(augmented_train_samples['X'])))
                self.data['y_train'] = np.concatenate((self.data['y_train'], np.array(augmented_train_samples['y'])))
                np.savez_compressed(os.path.join(output_path, target_dataset_name + '.npz'),
                                    X_train=self.data['X_train'],
                                    y_train=self.data['y_train'],
                                    X_test=self.data['X_test'],
                                    y_test=self.data['y_test'])

            if use_preprocess:
                print(f'Start preprocess...')
                from auxiliary_data_AD.preprocessing import embedding_data
                X_t = self.data['X_train']
                y_t = self.data['y_train']
                X_v = self.data['X_test']
                y_v = self.data['y_test']

                X_t_e, _ = embedding_data(encoder_name=preprocess_params['encoder_name'],
                                          layers=preprocess_params['layers'],
                                          interpolate=preprocess_params['interpolate'], X=X_t,
                                          y=y_t)
                X_v_e, _ = embedding_data(encoder_name=preprocess_params['encoder_name'],
                                          layers=preprocess_params['layers'],
                                          interpolate=preprocess_params['interpolate'], X=X_v,
                                          y=y_v)
                self.data['X_train'] = X_t_e
                self.data['X_test'] = X_v_e
                print(f'Finish preprocess.')

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
    aug_type = 'citmix'
    lamda = 0.85
    aux_size = 1
    use_preprocess = False

    path_project = '/home/yukina/Missile_Fault_Detection/project'

    category = 'MVTec-AD_metal_nut'
    dataset_path = os.path.join(path_project, f'data/mvtec_ad/{category}.npz')

    path_save = os.path.join(path_project, 'auxiliary_data_AD/log/test',
                             f'DeepSAD_{aug_type},lamda={lamda},aux_size={aux_size}', category)
    os.makedirs(path_save, exist_ok=True)  # 创建结果文件夹

    # 实例化并运行pipeline
    from adbench_modified.baseline.DeepSAD.src.run import DeepSAD

    pipeline = RunPipeline(suffix='DeepSAD', parallel='unsupervise', n_samples_threshold=200, seed=seed,
                           realistic_synthetic_mode=None,
                           noise_type=None, path_result=path_save)
    result_data = pipeline.run_universum(clf=DeepSAD, target_dataset_name=category,
                                         path_datasets=path_project + '/data/mvtec_ad',
                                         use_preprocess=use_preprocess,
                                         universum_params={'aug_type': aug_type, 'lamda': lamda, 'aux_size': aux_size})
