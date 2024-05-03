import numpy as np

# 例: 予測された点群と観測された点群
predicted_points = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
observed_points = np.array([
    [1, 3, 5],
    [2, 4, 6],
    [8, 7, 9]
])

# 各点対のユークリッド距離を計算
distances = np.linalg.norm(predicted_points - observed_points, axis=1)

print("距離:", distances)
# RMSEを計算
rmse = np.sqrt(np.mean(distances ** 2))
print("RMSE:", rmse)