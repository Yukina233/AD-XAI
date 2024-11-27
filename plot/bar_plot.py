import os

import matplotlib.pyplot as plt
import numpy as np

path_project = '/home/yukina/Missile_Fault_Detection/project'
output_path = os.path.join(path_project, 'plot/results')
os.makedirs(output_path, exist_ok=True)

# 数据
datasets = ("AUROC", "AUPR")
means = {
    'DeepSAD': (52.37, 0.68),
    'Ensemble DeepSAD': (71.84, 1.23),
    'BAEM without Anomaly Score': (77.28, 1.47),
    'BAEM without Uncertainty': (72.01, 1.36),
    'BAEM': (81.09, 1.62)
}

# 错误数据
errors = {
    'DeepSAD': (7.56, 0.12),
    'Ensemble DeepSAD': (4.92, 0.26),
    'BAEM without Anomaly Score': (3.26, 0.16),
    'BAEM without Uncertainty': (8.24, 0.16),
    'BAEM': (0.06, 0.01)
}

# 颜色
# colors = ['gainsboro', 'lightgray', 'darkgray', 'gray', 'dimgray']
colors = ['indianred', 'sandybrown', 'darkseagreen', 'steelblue', 'slategrey']

x = np.array([0, 1.2])  # 添加间隔，设置x轴的位置，2对应AUPR的位置
width = 0.15  # 条形宽度
spacing = 0.04  # 条之间的间隔
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
fig.set_size_inches(12, 5)

# 设置字体大小
label_fontsize = 14  # 标签字体大小
title_fontsize = 16  # 标题字体大小
tick_fontsize = 14   # 刻度字体大小
legend_fontsize = 14  # 图例字体大小

for (attribute, measurement), color in zip(means.items(), colors):
    offset = (width + spacing) * multiplier  # 考虑间隔的偏移量
    error = errors[attribute]
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=color, yerr=error, capsize=5)
    ax.bar_label(rects, padding=5)
    multiplier += 1

# 设置标签、标题和自定义x轴刻度标签等
ax.set_ylabel('Area Under Curve(%)', fontsize=label_fontsize)  # Y轴标签字体大小
ax.set_title('Metro Dataset', fontsize=title_fontsize)  # 设置标题和字体大小
ax.set_xticks(x + (width + spacing) * 2, datasets)  # 设置x轴刻度和标签
ax.tick_params(axis='x', labelsize=tick_fontsize)  # 设置x轴刻度字体大小
ax.tick_params(axis='y', labelsize=tick_fontsize)  # 设置y轴刻度字体大小
ax.legend(loc='upper right', ncols=1, fontsize=legend_fontsize)  # 图例字体大小
ax.set_ylim(0, 105)  # 调整y轴范围以适应误差线

plt.savefig(os.path.join(output_path, 'bar_plot_1.jpg'), dpi=330, format='jpg')
plt.show()
plt.cla()

# 数据
datasets = ("AUROC", "AUPR")
means = {
    'DeepSAD': (74.59, 5.77),
    'Ensemble DeepSAD': (86.69, 12.18),
    'BAEM without Anomaly Score': (91.85, 10.11),
    'BAEM without Uncertainty': (83.26, 9.71),
    'BAEM': (93.52, 12.46)
}

# 错误数据
errors = {
    'DeepSAD': (7.49, 2.95),
    'Ensemble DeepSAD': (10.68, 7.50),
    'BAEM without Anomaly Score': (4.65, 8.68),
    'BAEM without Uncertainty': (8.20, 5.60),
    'BAEM': (1.94, 5.06)
}

# 颜色
# colors = ['gainsboro', 'lightgray', 'darkgray', 'gray', 'dimgray']
colors = ['indianred', 'sandybrown', 'darkseagreen', 'steelblue', 'slategrey']


x = np.array([0, 1.2])  # 添加间隔，设置x轴的位置，2对应AUPR的位置
width = 0.15  # 条形宽度
spacing = 0.04  # 条之间的间隔
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
fig.set_size_inches(12, 5)

for (attribute, measurement), color in zip(means.items(), colors):
    offset = (width + spacing) * multiplier  # 考虑间隔的偏移量
    error = errors[attribute]
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=color, yerr=error, capsize=5)
    ax.bar_label(rects, padding=5)
    multiplier += 1


# 设置标签、标题和自定义x轴刻度标签等
ax.set_ylabel('Area Under Curve(%)', fontsize=label_fontsize)  # Y轴标签字体大小
ax.set_title('GHL Dataset', fontsize=title_fontsize)  # 设置标题和字体大小
ax.set_xticks(x + (width + spacing) * 2, datasets)  # 设置x轴刻度和标签
ax.tick_params(axis='x', labelsize=tick_fontsize)  # 设置x轴刻度字体大小
ax.tick_params(axis='y', labelsize=tick_fontsize)  # 设置y轴刻度字体大小
ax.legend(loc='upper right', ncols=1, fontsize=legend_fontsize)  # 图例字体大小
ax.set_ylim(0, 105)  # 调整y轴范围以适应误差线

plt.savefig(os.path.join(output_path, 'bar_plot_2.jpg'), dpi=330, format='jpg')
plt.show()
plt.cla()