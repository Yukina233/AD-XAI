import os

import matplotlib.pyplot as plt
import numpy as np

path_project = '/home/yukina/Missile_Fault_Detection/project'
output_path = os.path.join(path_project, 'plot/results')
os.makedirs(output_path, exist_ok=True)

# 数据
import numpy as np
import matplotlib.pyplot as plt
import os

# 数据
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
colors = ['indianred', 'sandybrown', 'darkseagreen', 'steelblue', 'slategrey']

# 只提取 AUROC 数据
x = np.array([0])  # AUROC 对应的 x 坐标
width = 0.15  # 条形宽度
spacing = 0.04  # 条之间的间隔
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
fig.set_size_inches(8, 6)

# 设置字体大小
label_fontsize = 14  # 标签字体大小
title_fontsize = 16  # 标题字体大小
tick_fontsize = 14  # 刻度字体大小
legend_fontsize = 14  # 图例字体大小

# 绘制 AUROC 图表
for (attribute, measurement), color in zip(means.items(), colors):
    # 只绘制 AUROC (索引 0)
    measurement_auroc = measurement[0]
    error_auroc = errors[attribute][0]

    offset = (width + spacing) * multiplier  # 考虑间隔的偏移量
    rects = ax.bar(x + offset, measurement_auroc, width, label=attribute, color=color, yerr=error_auroc, capsize=5)
    ax.bar_label(rects, padding=5)
    multiplier += 1

# 设置标签、标题和自定义x轴刻度标签等
ax.set_ylabel('Area Under Curve(%)', fontsize=label_fontsize)  # Y轴标签字体大小
ax.set_title('Metro Dataset - AUROC', fontsize=title_fontsize)  # 设置标题和字体大小
ax.set_xticks(x + (width + spacing) * 2)  # 只显示 AUROC 对应的标签
ax.set_xticklabels(['AUROC'], fontsize=tick_fontsize)  # x轴只显示 AUROC
ax.tick_params(axis='x', labelsize=tick_fontsize)  # 设置x轴刻度字体大小
ax.tick_params(axis='y', labelsize=tick_fontsize)  # 设置y轴刻度字体大小
ax.legend(loc='lower right', ncols=1, fontsize=legend_fontsize)  # 图例字体大小
ax.set_ylim(40, 105)  # 调整y轴范围以适应误差线

# 保存图像
plt.savefig(os.path.join(output_path, 'bar_plot_1.jpg'), dpi=330, format='jpg')
plt.show()
plt.cla()

# 数据
means = {
    '(a) DeepSAD': (74.59, 5.77),
    '(b) Ensemble DeepSAD': (86.69, 12.18),
    '(c) BAEM without Anomaly Score': (91.85, 10.11),
    '(d) BAEM without Uncertainty': (83.26, 9.71),
    '(e) BAEM': (93.52, 12.46)
}

# 错误数据
errors = {
    '(a) DeepSAD': (7.49, 2.95),
    '(b) Ensemble DeepSAD': (10.68, 7.50),
    '(c) BAEM without Anomaly Score': (4.65, 8.68),
    '(d) BAEM without Uncertainty': (8.20, 5.60),
    '(e) BAEM': (1.94, 5.06)
}

# 颜色
colors = ['indianred', 'sandybrown', 'darkseagreen', 'steelblue', 'slategrey']

# 只提取 AUROC 数据
x = np.array([0])  # 只绘制 AUROC 对应的 x 坐标
width = 0.15  # 条形宽度
spacing = 0.04  # 条之间的间隔
multiplier = 0

# 创建图表
fig, ax = plt.subplots(layout='constrained')
fig.set_size_inches(8, 6)

# 绘制 AUROC 图表
for (attribute, measurement), color in zip(means.items(), colors):
    # 只提取 AUROC 的值 (索引 0)
    measurement_auroc = measurement[0]
    error_auroc = errors[attribute][0]

    offset = (width + spacing) * multiplier  # 考虑间隔的偏移量
    rects = ax.bar(x + offset, measurement_auroc, width, label=attribute, color=color, yerr=error_auroc, capsize=5)
    ax.bar_label(rects, padding=5)
    multiplier += 1

# 设置标签、标题和自定义x轴刻度标签等
ax.set_ylabel('AUROC(%)', fontsize=14)  # Y轴标签字体大小
ax.set_title('GHL Dataset', fontsize=16)  # 设置标题和字体大小
ax.set_xticks(x + (width + spacing) * 2)  # 设置x轴刻度位置
ax.set_xticklabels(['ablation settings'], fontsize=14)  # 设置x轴标签，只显示 AUROC
ax.tick_params(axis='x', labelsize=14)  # 设置x轴刻度字体大小
ax.tick_params(axis='y', labelsize=14)  # 设置y轴刻度字体大小
ax.legend(loc='lower right', ncols=1, fontsize=14)  # 图例字体大小
ax.set_ylim(40, 105)  # 调整y轴范围

# 保存图像
plt.savefig(os.path.join(output_path, 'bar_plot_2.jpg'), dpi=330, format='jpg')
plt.show()
plt.cla()
