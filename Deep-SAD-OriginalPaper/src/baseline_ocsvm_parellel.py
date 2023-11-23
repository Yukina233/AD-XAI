import multiprocessing
from baseline_ocsvm import run

def main():
    path_project = '/home/yukina/Missile_Fault_Detection/project/Deep-SAD-OriginalPaper/log/baseline/ocsvm/'
    seed_num = 3
    config = {
        'dataset_name': 'cifar10',
        'xp_path': None,
        'seed': None,
        'normal_class': None,
        'known_outlier_class': None,
        'n_known_outlier_classes': 0,
        'ratio_known_normal': 0,
        'ratio_known_outlier': 0,
        'hybrid': False,
    }
    experiment_configs = []
    # 10 * 9 * 3 = 270 configs
    for i in range(0, 10):
        j_iterator = 0
        for j in range(0, 10):
            if j_iterator == 1:
                break
            if i != j:  # Only proceed if normal_class is different from known_outlier_class
                for k in range(0, seed_num):
                    config['normal_class'] = i
                    config['known_outlier_class'] = j
                    config['seed'] = k
                    # Generate xp_path with the required format
                    config[
                        'xp_path'] = path_project + \
                                     f"hybrid={config['hybrid']}/ratio={config['ratio_known_outlier']}/"\
                                     f"dataset={config['dataset_name']},normal={config['normal_class']}," \
                                     f"outlier={config['known_outlier_class']},ratioNormal={config['ratio_known_normal']}," \
                                     f"ratioOutlier={config['ratio_known_outlier']},seed={config['seed']}"
                    experiment_configs.append(config.copy())
                j_iterator += 1

    # 创建一个工作池，大小为可用CPU核心数
    with multiprocessing.Pool(processes=6) as pool:
        # 将 run_experiment 函数映射到配置上
        pool.map(run, experiment_configs)

        # 关闭工作池并等待工作完成
        pool.close()
        pool.join()

if __name__ == '__main__':
    main()