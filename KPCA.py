import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons

# 生成数据集
X, y = make_moons(n_samples=100, random_state=123)

# 定义径向基函数核
def rbf_kernel(x, y, gamma=1.0):
    return np.exp(-gamma * np.linalg.norm(x - y)**2)

# 计算核矩阵
def compute_kernel_matrix(X, gamma=1.0):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = rbf_kernel(X[i], X[j], gamma)
    return K

# 中心化核矩阵
def center_kernel_matrix(K):
    n_samples = K.shape[0]
    one_n = np.ones((n_samples, n_samples)) / n_samples
    K_centered = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    return K_centered

# 计算特征值和特征向量
def compute_eigenvectors(K_centered, n_components):
    eigvals, eigvecs = np.linalg.eigh(K_centered)
    idx = np.argsort(eigvals)[::-1][:n_components]
    return eigvecs[:, idx]

# 执行KPCA并绘制降维可视化结果
K = compute_kernel_matrix(X)
K_centered = center_kernel_matrix(K)
n_components = 2
eigenvectors = compute_eigenvectors(K_centered, n_components)
X_pca = K_centered.dot(eigenvectors)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('Kernelized PCA - 2D Projection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()