import numpy as np
from sklearn.decomposition import PCA

omega_1 = np.array([[-5, -5], [-5, -4], [-4, -5], [-5, -6], [-6, -5]])
omega_2 = np.array([[5, 5], [5, 6], [6, 5], [5, 4], [4, 5]])
X = np.vstack((omega_1, omega_2)) # 合并两类样本

pca = PCA(n_components=1) # 使用PCA进行一维特征提取
X_pca = pca.fit_transform(X)

print("原始数据：",X)
print("一维特征提取结果：",X_pca)
print("主成分（特征向量）：",pca.components_)
