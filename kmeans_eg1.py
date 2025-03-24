import numpy as np
import matplotlib.pyplot as plt
# 读取数据
data = np.genfromtxt('./xigua_data/4.0.csv', delimiter=',')

def kmeans(data, k, max_iterations=4):
    # 随机初始化中心点
    centroids = data[np.random.choice(range(len(data)), k, replace=False)]
    all_labels = []  # 存储每次迭代的标签
    all_centroids = []  # 存储每次迭代的中心点
    for i in range(max_iterations):
        # 计算每个点到中心点的距离
        distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        # 分配每个点到最近的中心点
        labels = np.argmin(distances, axis=0)
        # 更新中心点
        new_centroids = np.array([data[labels == j].mean(axis=0) for j in range(k)])
        # 存储当前迭代的结果
        all_labels.append(labels)
        all_centroids.append(centroids)
        # 如果中心点不再变化，提前退出
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return all_labels, all_centroids

k = 3              # 聚类簇数
max_iterations = 4 # 迭代次数

all_labels, all_centroids = kmeans(data, k, max_iterations)
plt.figure(figsize=(12, 10))  # 设置画布大小
for i in range(max_iterations):
    plt.subplot(2, 2, i + 1)  # 将画布分为2x2的四个子图
    labels = all_labels[i]
    centroids = all_centroids[i]
    # 绘制数据点
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50)
    # 绘制中心点
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3)
    # 为每个类绘制虚框
    for j in range(k):
        class_points = data[labels == j]
        if len(class_points) > 0:
            x_min, y_min = np.min(class_points, axis=0)
            x_max, y_max = np.max(class_points, axis=0)
            plt.plot([x_min, x_max, x_max, x_min, x_min],
                     [y_min, y_min, y_max, y_max, y_min],
                     'k--', linewidth=1)  # 黑色虚线框
    plt.title(f'Iteration {i + 1}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
plt.tight_layout()  # 调整子图间距
plt.show()