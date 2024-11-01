import subprocess
from multiprocessing import Pool

# 定义不同参数组合
params_list = []
for i in range(0, 10):
    # if i == 5:
    #     continue
    params_list.append(
        {"n_epochs": '10',
        "train_set_name": f'MNIST_nonorm_{i}',
        }
    )



# 定义单个运行函数
def run_script(params):
    command = [
        "python", "MNIST_ensemble_DeepSAD_train.py",
        "--n_epochs", params["n_epochs"],
        "--train_set_name", params["train_set_name"]
    ]
    subprocess.run(command)


# 使用 Pool 并行运行
if __name__ == '__main__':
    with Pool() as pool:
        pool.map(run_script, params_list)
