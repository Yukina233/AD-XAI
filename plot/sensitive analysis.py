import matplotlib.pyplot as plt


x_values = [1, 5, 10, 15, 20, 25, 30]
y_values1 = [85.15, 86.12, 86.7, 85.36, 84.94, 86.35, 86.17]
y_values2 = [74.37, 76.43, 77.28, 75.74, 75.3, 77.18, 76.03]
# 创建折线图
plt.figure(figsize=(4, 3))
# 绘制第一组数据，带有数据点和不同的颜色
plt.plot(range(len(x_values)), y_values1, label="AUCROC", marker='o', color='#1e488f', linestyle='-', linewidth=2)
# 绘制第二组数据，带有数据点和不同的颜色
plt.plot(range(len(x_values)), y_values2, label="AUCPR", marker='s', color='green', linestyle='-', linewidth=2)
plt.xticks(ticks=range(len(x_values)), labels=x_values)
plt.xlabel(r"$\alpha$")
plt.ylim(60, 100)
plt.tight_layout()
# 显示图表
plt.show()
plt.cla()

x_values = [5, 7, 9, 11, 13, 15]
y_values1 = [84.1, 86.7, 84.99, 85.86, 85.8, 85.51]
y_values2 = [74.8, 77.28, 74.52, 74.48, 75.58, 75.01]

# 创建折线图
plt.figure(figsize=(4, 3))
# 绘制第一组数据，带有数据点和不同的颜色
plt.plot(range(len(x_values)), y_values1, label="AUCROC", marker='o', color='#1e488f', linestyle='-', linewidth=2)
# 绘制第二组数据，带有数据点和不同的颜色
plt.plot(range(len(x_values)), y_values2, label="AUCPR", marker='s', color='green', linestyle='-', linewidth=2)
plt.xticks(ticks=range(len(x_values)), labels=x_values)
plt.xlabel(r"$K$")
plt.ylim(60, 100)
plt.tight_layout()
# 显示图表
plt.show()
plt.cla()

x_values = [25, 50, 100, 150, 255, 300, 350]
y_values1 = [85.29, 86.74, 86.04, 86.22, 86.7, 85.26, 85.26]
y_values2 = [76.05, 77.07, 75.11, 75.16, 77.28, 75.78, 74.48]

# 创建折线图
plt.figure(figsize=(4, 3))
# 绘制第一组数据，带有数据点和不同的颜色
plt.plot(range(len(x_values)), y_values1, label="AUCROC", marker='o', color='#1e488f', linestyle='-', linewidth=2)
# 绘制第二组数据，带有数据点和不同的颜色
plt.plot(range(len(x_values)), y_values2, label="AUCPR", marker='s', color='green', linestyle='-', linewidth=2)
plt.xticks(ticks=range(len(x_values)), labels=x_values)
plt.xlabel(r"$H$")
plt.ylim(60, 100)
plt.tight_layout()
# 显示图表
plt.show()
plt.cla()

x_values = ['2e-5', '2e-4', '5e-4', '2e-3', '5e-3', '2e-2']
y_values1 = [85.62, 85.53, 86.51, 86.7, 85.53, 86.38]
y_values2 = [75.01, 76.62, 76.44, 77.28, 76.3, 76.6]

# 创建折线图
plt.figure(figsize=(4, 3))
# 绘制第一组数据，带有数据点和不同的颜色
plt.plot(range(len(x_values)), y_values1, label="AUCROC", marker='o', color='#1e488f', linestyle='-', linewidth=2)
# 绘制第二组数据，带有数据点和不同的颜色
plt.plot(range(len(x_values)), y_values2, label="AUCPR", marker='s', color='green', linestyle='-', linewidth=2)
plt.xticks(ticks=range(len(x_values)), labels=x_values)
plt.xlabel(r"$\eta$")
plt.ylim(60, 100)
plt.tight_layout()
# 显示图表
plt.show()
plt.cla()


x_values = ['2e-5', '2e-4', '5e-4', '2e-3', '5e-3', '2e-2']
y_values1 = [85.62, 85.53, 86.51, 86.7, 85.53, 86.38]
y_values2 = [75.01, 76.62, 76.44, 77.28, 76.3, 76.6]


# 创建折线图
fig, ax = plt.subplots(figsize=(16, 12))
# 绘制第一组数据
line1, = ax.plot(range(len(x_values)), y_values1, label="AUCROC", marker='o', color='#1e488f', linestyle='-', linewidth=2)

# 绘制第二组数据
line2, = ax.plot(range(len(x_values)), y_values2, label="AUCPR", marker='s', color='green', linestyle='-', linewidth=2)

# 设置x轴的标签为等距离的x_values
ax.set_xticks(range(len(x_values)))
ax.set_xticklabels(x_values)

# 显示网格
ax.grid(True)

# 不在图表内显示图例
# plt.legend()

# 单独输出图例
fig.legend(handles=[line1, line2], labels=["AUCROC", "AUCPR"], loc='center right', ncol=1)

plt.show()