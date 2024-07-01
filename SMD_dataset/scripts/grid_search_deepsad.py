import itertools
import subprocess
import os

# 定义超参数的搜索空间
epoch_grid = [50, 100, 150, 200]
lr_grid = [0.005]
ae_lr_grid = [0.001]

# 生成所有的超参数组合
param_grid = list(itertools.product(epoch_grid, lr_grid, ae_lr_grid))

# 项目路径
path_project = '/home/yukina/Missile_Fault_Detection/project'

# 现有脚本的路径
script_path = os.path.join(path_project, 'SMD_dataset/SMD_DeepSAD_dataset.py')

# 遍历所有的超参数组合
for epoch, lr, ae_lr in param_grid:
    # 构建命令行参数
    command = [
        'python', script_path,
        '--DeepSAD_config',
        f'{{"n_epochs": {epoch}, "ae_n_epochs": {epoch}, "lr": {lr}, "ae_lr": {ae_lr}, "net_name": "SMD_GRU"}}'
    ]

    # 打印当前超参数组合
    print(f"Running with epoch={epoch}, lr={lr} and ae_lr={ae_lr}")

    # 调用现有脚本
    subprocess.run(command)
