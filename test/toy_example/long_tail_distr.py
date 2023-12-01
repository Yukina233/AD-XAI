import numpy as np
import matplotlib.pyplot as plt
from math import erf
from mpmath import erfinv
from scipy.stats import pareto, norm


def generate_gaussian_data(mean=0, std_dev=1, num_samples=100, seed=0):
    np.random.seed(seed)
    # 生成一定数量的样本
    samples = np.random.normal(mean, std_dev, num_samples)

    # 定义我们感兴趣的范围
    lower_bound = mean - 3 * std_dev
    upper_bound = mean + 3 * std_dev

    # 筛选出在这个范围内的点
    filtered_samples = samples[(samples >= lower_bound) & (samples <= upper_bound)]
    print(f"Number of samples in range [{lower_bound}, {upper_bound}]: {len(filtered_samples)}")
    return samples


def generate_pareto_data(alpha=1.0, num_samples=100, seed=0):
    np.random.seed(seed)
    # 生成一定数量的样本
    samples = pareto.rvs(alpha, size=num_samples)

    # 定义我们感兴趣的范围
    lower_bound = 0
    upper_bound = 10

    # 筛选出在这个范围内的点
    filtered_samples = samples[(samples >= lower_bound) & (samples <= upper_bound)]
    print(f"Number of samples in range [{lower_bound}, {upper_bound}]: {len(filtered_samples)}")

    return samples


if __name__ == '__main__':
    # data = generate_gaussian_data(mean=1, std_dev=2, num_samples=1000, seed=0)
    data = generate_pareto_data(alpha=3.0, num_samples=1000, seed=0)
    hist_bins = int(np.sqrt(len(data)))  # 一个简单的直方图bin数量选择
    plt.hist(data, bins=hist_bins)
    plt.show()

    # initialize with max likelyhood estimation
    mu_initial = np.mean(data)
    sigma_initial = np.std(data, ddof=1)
    log_likelihood = np.sum(norm.logpdf(data, mu_initial, sigma_initial))
    print(f"初始对数似然: {log_likelihood}")

    # 可视化数据及其拟合的高斯分布
    plt.hist(data, bins=hist_bins, density=True, alpha=0.6, color='g')

    # 在同一图上画出高斯分布
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu_initial, sigma_initial)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu_initial, sigma_initial)
    plt.title(title)
    plt.show()

    # 去掉低密度点后拟合
    # 假设我们只想拟合高于百分比的点
    # 删除低于百分比的点
    removed_percentile = 0.05
    mu_mle = mu_initial
    sigma_mle = sigma_initial
    for i in range(1, 3):
        # E step: 计算每个点的概率, 删去低于百分比的点
        threshold_low = norm.ppf(removed_percentile / 2, mu_mle, sigma_mle)
        threshold_high = norm.ppf(1 - removed_percentile / 2, mu_mle, sigma_mle)
        filtered_data = data[(threshold_low <= data) & (data <= threshold_high)]
        # M step: 重新计算参数
        mu_mle = np.mean(filtered_data)
        sigma_mle = np.std(filtered_data, ddof=1)
        log_likelihood = np.sum(norm.logpdf(data, mu_mle, sigma_mle))
        print(f"EM step={i}, 对数似然: {log_likelihood}")
        # 可视化数据及其拟合的高斯分布
        hist_bins = int(np.sqrt(len(data)))  # 一个简单的直方图bin数量选择
        counts, bin_edges = np.histogram(data, bins=hist_bins, density=True)
        # bin_edges 数组长度比 counts 长1，因为它定义了每个bin的边界
        # 我们可以计算每个bin的中心，作为x轴的表示
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 如果你想绘制这个直方图
        used_bins = (threshold_low <= bin_centers) & (bin_centers <= threshold_high)
        unused_bins = (threshold_low > bin_centers) | (bin_centers > threshold_high)
        plt.bar(bin_centers[used_bins], counts[used_bins], width=(bin_edges[1] - bin_edges[0]), color='g')
        plt.bar(bin_centers[unused_bins], counts[unused_bins], width=(bin_edges[1] - bin_edges[0]), color='r')
        # 在同一图上画出高斯分布
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu_mle, sigma_mle)
        plt.plot(x, p, 'k', linewidth=2)
        title = "EM step = %d, Fit results: mu = %.2f,  std = %.2f" % (i, mu_mle, sigma_mle)
        plt.title(title)
        plt.show()
