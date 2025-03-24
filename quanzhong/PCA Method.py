import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

file_path = "origin_summary_zhejiang_withid.xlsx"
df = pd.read_excel(file_path)

expected_columns = ['province', 'year'] + [f'x{i}' for i in range(1,17)]
actual_columns = df.columns.tolist()

# 提取特征数据（x1-x16）
feature_columns = [f'x{i}' for i in range(1,17)]
X = df[feature_columns]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用PCA计算权重（不需要目标变量）
pca = PCA(n_components=0.95)
pca.fit(X_scaled)

# 计算PCA权重（基于载荷和方差贡献率）
loadings = np.abs(pca.components_)
variance_ratio = pca.explained_variance_ratio_
pca_weights = (loadings.T @ variance_ratio).reshape(-1)
pca_weights /= pca_weights.sum()

# 将权重添加到原始数据框
for i, col in enumerate(feature_columns):
    df[f'w_{col}'] = pca_weights[i]

# 结果保存（可选）
output_cols = ['province', 'year'] + feature_columns + [f'w_{col}' for col in feature_columns]
df[output_cols].to_excel("result_with_weights.xlsx", index=False)

print("数据处理完成，前5行预览：")
print(df[output_cols].head())