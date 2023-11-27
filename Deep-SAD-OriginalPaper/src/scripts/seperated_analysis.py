import json
import os
import sys

import matplotlib.colors
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, precision_score, recall_score, \
    f1_score, average_precision_score
from torchvision.datasets import CIFAR10, MNIST
import matplotlib.pyplot as plt

from adbench_modified.baseline.DeepSAD.src.utils import plot_images_grid

sys.path.append('..')
from DeepSAD import DeepSAD


def plot_roc(fpr, tpr, roc_auc):
    # 画出ROC曲线
    plt.cla()
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(file_path + '/test ROC curve')


def plot_pr(recall, precision, pr_auc):
    # 画出PR曲线
    plt.figure()
    plt.plot(recall, precision, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.savefig(file_path + '/test PR curve')


def plot_distribution(trainset=1, threshold=0, scores_list=None, label_list=None, outlier=None):
    # 画出正常和异常数据得分的分布图
    plt.cla()
    plt.figure()
    if 1 in label_list:
        plt.hist([item[1] for item in enumerate(scores_list) if label_list[item[0]] == 1], bins=100, alpha=0.5,
                 label='Anomalies',
                 color=matplotlib.colors.TABLEAU_COLORS['tab:orange'])
    plt.hist([item[1] for item in enumerate(scores_list) if label_list[item[0]] == 0], bins=100, alpha=0.5,
             label='Normal',
             color=matplotlib.colors.TABLEAU_COLORS['tab:blue'])
    if not trainset == 2:
        plt.axvline(threshold, color='r', linestyle='dashed', linewidth=1, label='Threshold')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    if trainset == 1:
        set_name = 'train set'
    elif trainset == 0:
        set_name = 'test set'
        plt.title('Distribution of train set')
    else:
        set_name = 'train set at start'
        plt.title('Distribution of train set at start')
    plt.title('Distribution of ' + set_name)
    plt.savefig(file_path + '/Distribution of ' + set_name)


def plot_removal_distribution(config=None, scores_list=None, label_list=None):
    # 画出正常和异常数据得分的分布图
    plt.cla()
    plt.figure()
    threshold = config['remove_threshold']
    if 1 in label_list:
        plt.hist([item[1] for item in enumerate(scores_list) if label_list[item[0]] == 1], bins=100, alpha=0.5,
                 label='Anomalies',
                 color=matplotlib.colors.TABLEAU_COLORS['tab:orange'])
    plt.hist([item[1] for item in enumerate(scores_list)
              if label_list[item[0]] == 0], bins=100, alpha=0.5,
             label='Normal',
             color=matplotlib.colors.TABLEAU_COLORS['tab:blue'])
    # plt.hist([item[1] for item in enumerate(scores_list)
    #           if label_list[item[0]] == 0 and scores_list[item[0]] < threshold], bins=100, alpha=0.5,
    #          label='Normal to remove',
    #          color=matplotlib.colors.TABLEAU_COLORS['tab:red'])
    plt.axvline(threshold, color='r', linestyle='dashed', linewidth=1, label=f'Threshold={threshold}')
    plt.xlabel('Distance to anomaly centers')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.title('Distribution of train data to remove')
    plt.savefig(file_path + '/Distribution of train data to remove')


def show_test_samples_around_threshold(threshold=0, config=None, indices_list=None, labels_list=None, scores_list=None):
    # 输出阈值附近的样本标号
    normal_indices = indices_list[scores_list <= threshold]
    ordered_normal_indices = normal_indices[np.argsort(scores_list[normal_indices])]
    anomaly_indices = indices_list[scores_list > threshold]
    ordered_anomaly_indices = anomaly_indices[np.argsort(scores_list[anomaly_indices])]

    false_normal_indices = ordered_normal_indices[np.where(labels_list[ordered_normal_indices] == 1)]
    false_anomaly_indices = ordered_anomaly_indices[np.where(labels_list[ordered_anomaly_indices] == 0)]

    transform = transforms.ToTensor()
    target_transform = transforms.Lambda(lambda x: int(x != config['normal_class']))
    if config['dataset_name'] == 'cifar10':
        test_set = CIFAR10(root=config['data_path'], train=False, transform=transform,
                           target_transform=target_transform,
                           download=True)
        false_normal = torch.tensor(
            np.transpose(test_set.data[false_normal_indices[-32:], ...], (0, 3, 1, 2)))
        false_anomaly = torch.tensor(
            np.transpose(test_set.data[false_anomaly_indices[:32], ...], (0, 3, 1, 2)))

    if config['dataset_name'] == 'mnist':
        test_set = MNIST(root=config['data_path'], train=False, transform=transform, target_transform=target_transform,
                         download=True)
        false_normal = torch.tensor(test_set.data[false_normal_indices[-32:], ...].unsqueeze(1))
        false_anomaly = torch.tensor(test_set.data[false_anomaly_indices[:32], ...].unsqueeze(1))

    # 输出y_true = 1的正常样本
    plt.clf()
    plot_images_grid(false_normal, export_img=file_path + '/test_false_normal', padding=2)

    # 输出y_true = 0的异常样本
    plt.clf()
    plot_images_grid(false_anomaly, export_img=file_path + '/test_false_anomaly', padding=2)


def show_train_samples(config=None, exp_path=None, indices_list=None, labels_list=None, scores_list=None):
    train_idx_all_sorted = indices_list[np.argsort(scores_list)]  # from lowest to highest score
    train_idx_normal_sorted = indices_list[labels_list == 0][
        np.argsort(scores_list[labels_list == 0])]  # from lowest to highest score

    if config['dataset_name'] in ('mnist', 'cifar10'):
        transform = transforms.ToTensor()
        target_transform = transforms.Lambda(lambda x: int(x != config['normal_class']))
        if config['dataset_name'] == 'cifar10':
            train_set = CIFAR10(root=config['data_path'], train=True, transform=transform,
                                target_transform=target_transform,
                                download=True)
            train_X_all_low = torch.tensor(np.transpose(train_set.data[train_idx_all_sorted[:32], ...], (0, 3, 1, 2)))
            train_X_all_high = torch.tensor(np.transpose(train_set.data[train_idx_all_sorted[-32:], ...], (0, 3, 1, 2)))
            train_X_normal_low = torch.tensor(
                np.transpose(train_set.data[train_idx_normal_sorted[:32], ...], (0, 3, 1, 2)))
            train_X_normal_high = torch.tensor(
                np.transpose(train_set.data[train_idx_normal_sorted[-32:], ...], (0, 3, 1, 2)))

        if config['dataset_name'] == 'mnist':
            train_set = MNIST(root=config['data_path'], train=True, transform=transform,
                              target_transform=target_transform,
                              download=True)
            train_X_all_low = train_set.data[train_idx_all_sorted[:32], ...].unsqueeze(1)
            train_X_all_high = train_set.data[train_idx_all_sorted[-32:], ...].unsqueeze(1)

        plot_images_grid(train_X_all_low, export_img=exp_path + '/train_all_low', padding=2)
        plot_images_grid(train_X_all_high, export_img=exp_path + '/train_all_high', padding=2)
        plot_images_grid(train_X_normal_low, export_img=exp_path + '/train_normal_low', padding=2)
        plot_images_grid(train_X_normal_high, export_img=exp_path + '/train_normal_high', padding=2)


def reconstruction_C(exp_path):
    # load Deep SAD model (center c, network weights, and possibly autoencoder weights)
    deepSAD = DeepSAD()
    with open(os.path.join(exp_path, 'config.json'), 'r') as file:
        try:
            config_data = json.load(file)
            net_name = config_data.get('net_name')
        except json.JSONDecodeError:
            print(f"JSON解析错误：{exp_path + 'config.json'}")
        except KeyError:
            print(f"没有找到net_name项：{exp_path + 'config.json'}")
    deepSAD.set_network(net_name)
    deepSAD.load_model(model_path=os.path.join(exp_path, 'model.tar'), load_ae=True)
    print('Loading model from %s.' % exp_path + 'model.tar')

    c = torch.Tensor(np.array(deepSAD.c)).unsqueeze(0)
    deepSAD.ae_net.eval()
    reconstruction = deepSAD.ae_net.decoder(c).detach().numpy()
    reconstruction_image = reconstruction.squeeze()  # 去掉不必要的维度，如果有的话
    if reconstruction_image.ndim == 3 and reconstruction_image.shape[0] in {1, 3}:
        # 如果是单通道图像，去掉通道维度；如果是三通道，将通道移到最后
        reconstruction_image = np.transpose(reconstruction_image,
                                            (1, 2, 0) if reconstruction_image.shape[0] == 3 else (1, 0))

    # 输出图像
    plt.imshow(reconstruction_image, cmap='gray' if reconstruction_image.ndim == 2 else None)
    plt.axis('off')  # 不显示坐标轴
    plt.savefig(exp_path + '/Reconstruction C', bbox_inches='tight')

    # 读取原有的数据
    with open(os.path.join(exp_path, 'ae_results.json'), 'r') as file:
        data = json.load(file)

    # 在字典中添加新的键值对
    data['c'] = deepSAD.c

    # 将新的数据写回文件
    with open(os.path.join(exp_path, 'ae_results.json'), 'w') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    print('c = ', c)


if __name__ == '__main__':
    # 假设您的JSON数据保存在名为'data.json'的文件中
    project_path = '/home/yukina/Missile_Fault_Detection/project/Deep-SAD-OriginalPaper/log/'
    file_path = project_path + 'remove/seperate_normal, remove_threshold=15/cifar10/ae_epochs=100/ratio=0.2/dataset=cifar10,normal=3,outlier=5,ratioNormal=0.2,ratioOutlier=0.2,seed=0/'  # 文件路径

    # 从文件中读取JSON数据
    with open(file_path + 'config.json', 'r') as file:
        config = json.load(file)

    with open(file_path + 'results.json', 'r') as file:
        results = json.load(file)

    test_scores = results['test_scores']

    # 分离出真实标签和预测得分
    train_indices_0, train_labels_0, train_semi_targets_0, train_y_scores_0 = zip(*results['train_scores_0'])
    train_indices_0, train_labels_0, train_semi_targets_0, train_y_scores_0 = np.array(train_indices_0), np.array(
        train_labels_0), np.array(train_semi_targets_0), np.array(train_y_scores_0)

    train_indices, train_labels, train_semi_targets, train_y_scores = zip(*results['train_scores'])
    train_indices, train_labels, train_semi_targets, train_y_scores = np.array(train_indices), np.array(
        train_labels), np.array(train_semi_targets), np.array(train_y_scores)

    test_indices = [score[0] for score in test_scores]  # 样本索引
    test_labels = [score[1] for score in test_scores]  # 真实标签
    test_y_scores = [score[2] for score in test_scores]  # 预测得分
    test_indices, test_labels, test_y_scores = np.array(test_indices), np.array(test_labels), np.array(test_y_scores)

    origin_train_indices, origin_train_labels, origin_train_semi_targets, train_dis_to_anomaly = zip(
        *results['train_normal_dis_to_anomaly'])
    origin_train_indices, origin_train_labels, origin_train_semi_targets, train_dis_to_anomaly = np.array(
        origin_train_indices), np.array(origin_train_labels), np.array(origin_train_semi_targets), np.array(
        train_dis_to_anomaly)
    # 计算ROC曲线
    fpr_list, tpr_list, thresholds = roc_curve(test_labels, test_y_scores)
    roc_auc = auc(fpr_list, tpr_list)
    precision_list, recall_list, _ = precision_recall_curve(test_labels, test_y_scores)
    pr_auc = auc(recall_list, precision_list)

    # 根据训练数据计算阈值
    # 将95的非异常数据分类为正常, train_semi_targets = -1的数据为标记为异常的数据
    thresholds_95 = np.percentile([item[1] for item in enumerate(train_y_scores) if train_semi_targets[item[0]] != -1],
                                  95)

    print("test_AUROC: ", roc_auc)
    print("test_AUPR: ", pr_auc)
    print("Threshold:", thresholds_95)
    print("test_accuracy: ", accuracy_score(test_labels, test_y_scores > thresholds_95))
    print("test_precision: ", precision_score(test_labels, test_y_scores > thresholds_95))
    print("test_recall: ", recall_score(test_labels, test_y_scores > thresholds_95))
    print("test_f1: ", f1_score(test_labels, test_y_scores > thresholds_95))

    plot_roc(fpr_list, tpr_list, roc_auc)
    plot_pr(recall_list, precision_list, pr_auc)
    plot_distribution(trainset=2, threshold=config['remove_threshold'], scores_list=train_y_scores_0,
                      label_list=train_labels_0)
    plot_distribution(trainset=1, threshold=thresholds_95, scores_list=train_y_scores, label_list=train_labels)
    plot_distribution(trainset=0, threshold=thresholds_95, scores_list=test_y_scores, label_list=test_labels)
    plot_removal_distribution(config=config, scores_list=train_dis_to_anomaly, label_list=origin_train_labels)
    show_test_samples_around_threshold(threshold=thresholds_95, config=config, indices_list=test_indices,
                                       labels_list=test_labels, scores_list=test_y_scores)
    show_train_samples(config=config, exp_path=file_path, indices_list=train_indices, labels_list=train_labels,
                       scores_list=train_y_scores)
    reconstruction_C(exp_path=file_path)

    print('Analysis finished.')
