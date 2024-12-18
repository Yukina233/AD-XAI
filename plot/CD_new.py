import os

from aeon.visualisation import plot_critical_difference
from aeon.benchmarking.results_loaders import get_estimator_results_as_array

import pandas as pd

path_project = '/home/yukina/Missile_Fault_Detection/project'

# 假设从一个 CSV 文件中加载数据
# 文件格式为：classifier_name, dataset_name, accuracy
# 用 pandas 读取数据
df = pd.read_csv(os.path.join(path_project, f'plot/csv/results_AUCROC_GHL.csv'))

# 使用 pivot 将数据转换为目标格式
result = df.pivot(index="dataset_name", columns="classifier_name", values="accuracy")

methods = result.columns
# 可选：调整行列名称（更具可读性）
result.index.name = "Dataset"
result.columns.name = "Classifier"

# 可选：将结果保存为 CSV 文件
auroc = result.to_numpy()

plot, _, p = plot_critical_difference(auroc, methods, alpha=0.05, return_p_values=True)
plot.set_size_inches(9, 3)
plot.show()
plot.savefig(os.path.join(path_project, f'plot/results/cd-AUROC.jpg'), dpi=330, format="jpg")


df = pd.read_csv(os.path.join(path_project, f'plot/csv/results_AUCPR_GHL.csv'))

# 使用 pivot 将数据转换为目标格式
result = df.pivot(index="dataset_name", columns="classifier_name", values="accuracy")

methods = result.columns
# 可选：调整行列名称（更具可读性）
result.index.name = "Dataset"
result.columns.name = "Classifier"

# 可选：将结果保存为 CSV 文件
auroc = result.to_numpy()

plot, _ = plot_critical_difference(auroc, methods, alpha=0.05)
plot.set_size_inches(9, 3)
plot.show()
plot.savefig(os.path.join(path_project, f'plot/results/cd-AUPR.jpg'), dpi=330, format="jpg")