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

import numpy as np
import matplotlib.pyplot as plt
import os

import numpy as np
import matplotlib.pyplot as plt
import os

# 数据集和其他设置
datasets = ("Metro", "GHL")
means = {
    'DeepSAD': (52.37, 74.59),
    'Ensemble DeepSAD': (71.84, 86.69),
    'BAEM (w/o Anomaly Score Loss)': (77.28, 91.85),
    'BAEM (w/o Uncertainty Loss)': (72.01, 83.26),
    'BAEM (ours)': (81.09, 93.52)
}

# 错误数据
errors = {
    'DeepSAD': (7.56, 7.49),
    'Ensemble DeepSAD': (4.92, 10.68),
    'BAEM (w/o Anomaly Score Loss)': (3.26, 4.65),
    'BAEM (w/o Uncertainty Loss)': (8.24, 8.20),
    'BAEM (ours)': (0.06, 1.94)
}

# 颜色
colors = ['#ae7181', '#e6daa6', 'darkseagreen', '#5a86ad', 'slategrey']
# 花纹模式
hatches = ['', '|', '+', 'x', '.']  # 每个条形的不同花纹
# 花纹颜色（边框颜色）
hatch_colors = ['white', 'white', 'white', 'white', 'white']  # 这里的颜色控制花纹的颜色

x = np.array([0, 1.2])  # 设置x轴的位置
width = 0.15  # 条形宽度
spacing = 0.04  # 条之间的间隔
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
fig.set_size_inches(12, 9)

# 设置字体大小
label_fontsize = 14
title_fontsize = 16
tick_fontsize = 14
legend_fontsize = 14

for (attribute, measurement), color, hatch, hatch_color in zip(means.items(), colors, hatches, hatch_colors):
    offset = (width + spacing) * multiplier  # 考虑间隔的偏移量
    error = errors[attribute]
    # 设置edgecolor来改变花纹的颜色
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=color, yerr=error, capsize=5, hatch=hatch, edgecolor=hatch_color)
    ax.bar_label(rects, padding=5)
    multiplier += 1

# 设置标签、标题和自定义x轴刻度标签
ax.set_ylabel('AUROC(%)', fontsize=label_fontsize)
# ax.set_title('Metro Dataset', fontsize=title_fontsize)
ax.set_xticks(x + (width + spacing) * 2)  # 设置x轴刻度
ax.set_xticklabels(datasets)  # 设置x轴标签
ax.tick_params(axis='x', labelsize=tick_fontsize)  # 设置x轴刻度字体大小
ax.tick_params(axis='y', labelsize=tick_fontsize)  # 设置y轴刻度字体大小

# 图例在外部下方，ncol=2表示每行显示2个图例项
ax.legend(loc='upper center', ncol=3, fontsize=legend_fontsize, bbox_to_anchor=(0.5, -0.05))

# 设置y轴范围
ax.set_ylim(40, 105)

# 保存并显示图形
plt.savefig(os.path.join(output_path, 'bar_plot_1_with_hatches_and_colors.jpg'), dpi=330, format='jpg')
plt.show()
plt.cla()

# # 数据
# datasets = ("AUROC", "AUPR")
# means = {
#     'DeepSAD': (74.59, 5.77),
#     'Ensemble DeepSAD': (86.69, 12.18),
#     'BAEM without Anomaly Score': (91.85, 10.11),
#     'BAEM without Uncertainty': (83.26, 9.71),
#     'BAEM': (93.52, 12.46)
# }
#
# # 错误数据
# errors = {
#     'DeepSAD': (7.49, 2.95),
#     'Ensemble DeepSAD': (10.68, 7.50),
#     'BAEM without Anomaly Score': (4.65, 8.68),
#     'BAEM without Uncertainty': (8.20, 5.60),
#     'BAEM': (1.94, 5.06)
# }
#
# # 颜色
# # colors = ['gainsboro', 'lightgray', 'darkgray', 'gray', 'dimgray']
# colors = ['indianred', 'sandybrown', 'darkseagreen', 'steelblue', 'slategrey']
#
#
# x = np.array([0, 1.2])  # 添加间隔，设置x轴的位置，2对应AUPR的位置
# width = 0.15  # 条形宽度
# spacing = 0.04  # 条之间的间隔
# multiplier = 0
#
# fig, ax = plt.subplots(layout='constrained')
# fig.set_size_inches(12, 5)
#
# for (attribute, measurement), color in zip(means.items(), colors):
#     offset = (width + spacing) * multiplier  # 考虑间隔的偏移量
#     error = errors[attribute]
#     rects = ax.bar(x + offset, measurement, width, label=attribute, color=color, yerr=error, capsize=5)
#     ax.bar_label(rects, padding=5)
#     multiplier += 1
#
#
# # 设置标签、标题和自定义x轴刻度标签等
# ax.set_ylabel('Area Under Curve(%)', fontsize=label_fontsize)  # Y轴标签字体大小
# ax.set_title('GHL Dataset', fontsize=title_fontsize)  # 设置标题和字体大小
# ax.set_xticks(x + (width + spacing) * 2, datasets)  # 设置x轴刻度和标签
# ax.tick_params(axis='x', labelsize=tick_fontsize)  # 设置x轴刻度字体大小
# ax.tick_params(axis='y', labelsize=tick_fontsize)  # 设置y轴刻度字体大小
# ax.legend(loc='upper right', ncols=1, fontsize=legend_fontsize)  # 图例字体大小
# ax.set_ylim(0, 105)  # 调整y轴范围以适应误差线
#
# plt.savefig(os.path.join(output_path, 'bar_plot_2.jpg'), dpi=330, format='jpg')
# plt.show()
# plt.cla()