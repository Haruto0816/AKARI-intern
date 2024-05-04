import numpy as np

# データセット
data = np.array([[8, 1, 6], [8, 2, 6], [7, 8, 9]])

# np.cov を使用した共分散行列の計算
cov_matrix = np.cov(data, rowvar=False)  # rowvar=False は各列が変数であることを指定

print("共分散行列 (np.cov):")
print(cov_matrix)

# np.outer を使用した手動計算
mean_data = np.mean(data, axis=0)
cov_manual = np.zeros((3, 3))
for d in data:
    cov_manual += np.outer(d - mean_data, d - mean_data)
cov_manual /= (data.shape[0] - 1)

print("共分散行列 (手動計算):")
print(cov_manual)
