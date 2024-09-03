import matplotlib.pyplot as plt
import numpy as np

# 数据
datasets = ("Metro", "GHL")
means = {
    'DeepSAD': (52.37, 74.59),
    'Ensemble DeepSAD': (71.84, 86.39),
    'EGD-SAD without Mean Loss': (77.28, 91.85),
    'EGD-SAD without Consistency Loss': (72.01, 89.83),
    'EGD-SAD': (81.09, 93.52)
}

# 错误数据
errors = {
    'DeepSAD': (7.56, 7.49),
    'Ensemble DeepSAD': (4.92, 10.68),
    'EGD-SAD without Mean Loss': (3.26, 4.65),
    'EGD-SAD without Consistency Loss': (8.24, 6.14),
    'EGD-SAD': (0.06, 1.94)
}

# 颜色
colors = ['gainsboro', 'lightgray', 'darkgray', 'gray', 'dimgray']

x = np.array([0, 1])  # 标签位置
width = 0.15  # 条形宽度
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
fig.set_size_inches(8, 6)
for (attribute, measurement), color in zip(means.items(), colors):
    offset = width * multiplier
    error = errors[attribute]
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=color, yerr=error, capsize=5)
    ax.bar_label(rects, padding=5)
    multiplier += 1

# 设置标签、标题和自定义x轴刻度标签等
ax.set_xlabel('Dataset')
ax.set_ylabel('AUROC(%)')
ax.set_xticks(x + width * 2, datasets)
ax.legend(loc='lower left', ncols=1)
ax.set_ylim(0, 105)  # 调整y轴范围以适应误差线

plt.show()
plt.cla()

# 数据
datasets = ("Metro", "GHL")
means = {
    'DeepSAD': (0.68, 5.77),
    'Ensemble DeepSAD': (1.23, 12.18),
    'EGD-SAD without Mean Loss': (1.47, 10.11),
    'EGD-SAD without Consistency Loss': (1.36, 9.71),
    'EGD-SAD': (1.62, 12.46)
}

# 错误数据
errors = {
    'DeepSAD': (0.12, 2.95),
    'Ensemble DeepSAD': (0.26, 7.50),
    'EGD-SAD without Mean Loss': (0.16, 8.68),
    'EGD-SAD without Consistency Loss': (0.16, 5.60),
    'EGD-SAD': (0.01, 5.06)
}

# 颜色
colors = ['gainsboro', 'lightgray', 'darkgray', 'gray', 'dimgray']

x = np.array([0, 1])  # 标签位置
width = 0.15  # 条形宽度
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
fig.set_size_inches(8, 6)
for (attribute, measurement), color in zip(means.items(), colors):
    offset = width * multiplier
    error = errors[attribute]
    rects = ax.bar(x + offset, measurement, width, label=attribute, color=color, yerr=error, capsize=5)
    ax.bar_label(rects, padding=5)
    multiplier += 1

# 设置标签、标题和自定义x轴刻度标签等
ax.set_xlabel('Dataset')
ax.set_ylabel('AUPR(%)')
ax.set_xticks(x + width * 2, datasets)
ax.legend(loc='upper left', ncols=1)
ax.set_ylim(0, 105)  # 调整y轴范围以适应误差线

plt.show()
plt.cla()