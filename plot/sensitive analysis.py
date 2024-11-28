import os

import matplotlib.pyplot as plt


path_project = '/home/yukina/Missile_Fault_Detection/project'

output_dir = os.path.join(path_project, 'plot/results')
plt.cla()

x_values = [1, 5, 10, 15, 20, 25, 30]
y_values1 = [85.15, 86.12, 86.7, 85.36, 84.94, 86.35, 86.17]
y_values2 = [74.37, 76.43, 77.28, 75.74, 75.3, 77.18, 76.03]
y_values3 = [72.97, 73.72, 72.61, 74.62, 74.24, 73.12, 74.75]
y_values4 = [93.53, 93.57, 93.36, 93.96, 93.81, 93.39, 93.91]

# 创建折线图
plt.figure(figsize=(4, 3))
plt.plot(range(len(x_values)), y_values1, label="SWAT Dataset", marker='o', color='#1e488f', linestyle='-', linewidth=2)
# plt.plot(range(len(x_values)), y_values2, label="SWAT Dataset-AUPR", marker='o', color='#1e488f', linestyle='--', linewidth=2)
plt.plot(range(len(x_values)), y_values3, label="TLM Dataset", marker='s', color='green', linestyle='-', linewidth=2)
# plt.plot(range(len(x_values)), y_values4, label="TLM Dataset-AUPR", marker='s', color='green', linestyle='--', linewidth=2)
plt.xticks(ticks=range(len(x_values)), labels=x_values)
plt.xlabel(r"$\alpha$")
plt.ylabel('AUROC(%)')
plt.ylim(70, 100)
plt.tight_layout()
# 显示图表
plt.savefig(os.path.join(output_dir, 'alpha.jpg'), dpi=330, format='jpg')
plt.cla()

x_values = [0, 0.1, 1, 5, 10, 20, 25]
y_values1 = [58.98, 55.84, 60.63, 71.67, 69.46, 75.85, 73.59]

# 创建折线图
plt.figure(figsize=(14, 5))
plt.rcParams['font.family'] = 'Times New Roman'
ax = plt.gca()  # 获取当前的坐标轴
plt.plot(range(len(x_values)), y_values1, label="SWAT Dataset", marker='o', color='#1e488f', linestyle='-', linewidth=2)
# plt.plot(range(len(x_values)), y_values2, label="SWAT Dataset-AUPR", marker='o', color='#1e488f', linestyle='--', linewidth=2)
# plt.plot(range(len(x_values)), y_values4, label="TLM Dataset-AUPR", marker='s', color='green', linestyle='--', linewidth=2)
plt.xticks(ticks=range(len(x_values)), labels=x_values)
plt.xlabel(r"Weight of Uncertainty Term $\beta$", fontsize=16, weight='bold')
plt.ylabel('AUROC(%)', fontsize=16, weight='bold')
plt.ylim(50, 100)
ax.spines['top'].set_visible(False)   # 隐藏上边框
ax.spines['right'].set_visible(False) # 隐藏右边框
plt.tight_layout()


# 显示图表
plt.savefig(os.path.join(output_dir, 'beta.jpg'), dpi=330, format='jpg')
plt.cla()

x_values = [5, 7, 9, 11, 13, 15]
y_values1 = [84.1, 86.7, 84.99, 85.86, 85.8, 85.51]
y_values2 = [74.8, 77.28, 74.52, 74.48, 75.58, 75.01]
y_values3 = [75.16, 76.1, 71.33, 70.43, 71.49, 73.76]
y_values4 = [94.07, 94.26, 93.11, 92.87, 93.1, 93.55]

# 创建折线图
plt.figure(figsize=(4, 3))

plt.plot(range(len(x_values)), y_values1, label="SWAT Dataset", marker='o', color='#1e488f', linestyle='-', linewidth=2)
# plt.plot(range(len(x_values)), y_values2, label="SWAT Dataset-AUPR", marker='o', color='#1e488f', linestyle='--', linewidth=2)
plt.plot(range(len(x_values)), y_values3, label="TLM Dataset", marker='s', color='green', linestyle='-', linewidth=2)
# plt.plot(range(len(x_values)), y_values4, label="TLM Dataset-AUPR", marker='s', color='green', linestyle='--', linewidth=2)

plt.xticks(ticks=range(len(x_values)), labels=x_values)
plt.xlabel(r"$K$")
plt.ylabel('AUROC(%)')
plt.ylim(60, 100)
plt.tight_layout()
# 显示图表
plt.show()
plt.cla()

x_values = [25, 50, 100, 150, 250, 300, 350]
y_values1 = [85.29, 86.74, 86.04, 86.22, 86.7, 85.26, 85.26]
y_values2 = [76.05, 77.07, 75.11, 75.16, 77.28, 75.78, 74.48]
y_values3 = [73.13, 76.1, 75.64, 75.03, 74.62, 72.79, 72.35]
y_values4 = [93.53, 94.26, 94.23, 94.12, 93.98, 93.41, 93.43]

# 创建折线图
plt.figure(figsize=(4, 3))
plt.plot(range(len(x_values)), y_values1, label="SWAT Dataset", marker='o', color='#1e488f', linestyle='-', linewidth=2)
# plt.plot(range(len(x_values)), y_values2, label="SWAT Dataset-AUPR", marker='o', color='#1e488f', linestyle='--', linewidth=2)
plt.plot(range(len(x_values)), y_values3, label="TLM Dataset", marker='s', color='green', linestyle='-', linewidth=2)
# plt.plot(range(len(x_values)), y_values4, label="TLM Dataset-AUPR", marker='s', color='green', linestyle='--', linewidth=2)
plt.xticks(ticks=range(len(x_values)), labels=x_values)
plt.xlabel(r"$H$")
plt.ylabel('AUROC(%)')
plt.ylim(60, 100)
plt.tight_layout()
# 显示图表
plt.show()
plt.cla()

x_values = ['2e-5', '2e-4', '5e-4', '2e-3', '5e-3', '2e-2']
y_values1 = [85.62, 85.53, 86.51, 86.7, 85.53, 86.38]
y_values2 = [75.01, 76.62, 76.44, 77.28, 76.3, 76.6]
y_values3 = [75.43, 73.09, 72.99, 73.6, 76.1, 75.27]
y_values4 = [94.2, 93.49, 93.43, 93.78, 94.26, 94.06]

# 创建折线图
plt.figure(figsize=(4, 3))
plt.plot(range(len(x_values)), y_values1, label="SWAT Dataset", marker='o', color='#1e488f', linestyle='-', linewidth=2)
# plt.plot(range(len(x_values)), y_values2, label="SWAT Dataset-AUPR", marker='o', color='#1e488f', linestyle='--', linewidth=2)
plt.plot(range(len(x_values)), y_values3, label="TLM Dataset", marker='s', color='green', linestyle='-', linewidth=2)
# plt.plot(range(len(x_values)), y_values4, label="TLM Dataset-AUPR", marker='s', color='green', linestyle='--', linewidth=2)
plt.xticks(ticks=range(len(x_values)), labels=x_values)
plt.xlabel(r"$\eta$")
plt.ylabel('AUROC(%)')
plt.ylim(60, 100)
plt.tight_layout()
# 显示图表
plt.show()
plt.cla()


# 数据
x_values = ['2e-5', '2e-4', '5e-4', '2e-3', '5e-3', '2e-2']
y_values1 = [85.62, 85.53, 86.51, 86.7, 85.53, 86.38]  # AUROC
y_values2 = [75.01, 76.62, 76.44, 77.28, 76.3, 76.6]   # AUPR

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(20, 12))

# 绘制第一组数据 (AUROC, 实线)
line1, = ax.plot(
    range(len(x_values)), y_values1,
    label="SWAT Dataset", marker='o', color='#1e488f', linestyle='-', linewidth=2
)
line2, = ax.plot(
    range(len(x_values)), y_values2,
    label="TLM Dataset", marker='s', color='green', linestyle='-', linewidth=2
)

# 设置x轴的标签为等距离的x_values
ax.set_xticks(range(len(x_values)))
ax.set_xticklabels(x_values)

# 显示网格
ax.grid(True)

# 单独输出图例
fig.legend(handles=[line1, line2], loc='center right', ncol=1)

# 显示图形
plt.show()
plt.cla()

# 数据
x_values = ['2e-5', '2e-4', '5e-4', '2e-3', '5e-3', '2e-2']
y_values1 = [85.62, 85.53, 86.51, 86.7, 85.53, 86.38]  # AUROC
y_values2 = [75.01, 76.62, 76.44, 77.28, 76.3, 76.6]   # AUPR

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(20, 12))

# 绘制第一组数据 (AUROC, 实线)
line1, = ax.plot(
    range(len(x_values)), y_values1,
    label="AUROC", color='black', linestyle='-', linewidth=2
)
line2, = ax.plot(
    range(len(x_values)), y_values2,
    label="AUPR", color='black', linestyle='--', linewidth=2
)

# 设置x轴的标签为等距离的x_values
ax.set_xticks(range(len(x_values)))
ax.set_xticklabels(x_values)

# 显示网格
ax.grid(True)

# 单独输出图例
fig.legend(handles=[line1, line2], loc='center right', ncol=1)

# 显示图形
plt.show()