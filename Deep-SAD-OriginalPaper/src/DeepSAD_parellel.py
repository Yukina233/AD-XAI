import multiprocessing
from new_main import run


def main():
    path_project = '/home/yukina/Missile_Fault_Detection/project/Deep-SAD-OriginalPaper/log/remove/seperate_normal, '
    seed_num = 3
    config = {
        'dataset_name': 'cifar10',
        'net_name': 'cifar10_LeNet',
        'xp_path': None,
        'seed': None,
        'normal_class': None,
        'known_outlier_class': None,
        'n_known_outlier_classes': 1,
        'ratio_known_normal': 0.2,
        'ratio_known_outlier': 0.2,
        'n_epochs': 50,
        'ae_n_epochs': 100,
        'remove_threshold': 0,
    }
    experiment_configs = []
    # 10 * 9 * 3 = 270 configs
    if config['normal_class'] is None and config['known_outlier_class'] is None:
        for i in range(0, 10):
            for j in range(0, 10):
                if i != j:  # Only proceed if normal_class is different from known_outlier_class
                    for k in range(0, seed_num):
                        config['normal_class'] = i
                        config['known_outlier_class'] = j
                        config['seed'] = k
                        # Generate xp_path with the required format
                        config[
                            'xp_path'] = path_project + \
                                         f"remove_threshold={config['remove_threshold']}/{config['dataset_name']}/ae_epochs={config['ae_n_epochs']}/ratio={config['ratio_known_outlier']}/" \
                                         f"dataset={config['dataset_name']},normal={config['normal_class']}," \
                                         f"outlier={config['known_outlier_class']},ratioNormal={config['ratio_known_normal']}," \
                                         f"ratioOutlier={config['ratio_known_outlier']},seed={config['seed']}"
                        experiment_configs.append(config.copy())
    else:
        # 指定normal_class和known_outlier_class
        for k in range(0, seed_num):
            config['seed'] = k
            # Generate xp_path with the required format
            config[
                'xp_path'] = path_project + \
                             f"remove_threshold={config['remove_threshold']}/{config['dataset_name']}" \
                             f"/ae_epochs={config['ae_n_epochs']}/ratio={config['ratio_known_outlier']}/" \
                             f"dataset={config['dataset_name']},normal={config['normal_class']}," \
                             f"outlier={config['known_outlier_class']},ratioNormal={config['ratio_known_normal']}," \
                             f"ratioOutlier={config['ratio_known_outlier']},seed={config['seed']}"
            experiment_configs.append(config.copy())

    # 创建一个工作池，大小为可用CPU核心数
    with multiprocessing.Pool(processes=4) as pool:
        # 将 run_experiment 函数映射到配置上
        pool.map(run, experiment_configs)

        # 关闭工作池并等待工作完成
        pool.close()
        pool.join()


if __name__ == '__main__':
    main()
