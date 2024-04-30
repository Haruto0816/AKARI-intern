import numpy as np

# ファイルパス
file_path = 'workspace1.txt'

# データを読み込む関数
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # 各行を空白で分割し、浮動小数点数のリストに変換
            point_data = list(map(float, line.split()))
            data.append(point_data)
    return np.array(data)

# データを読み込む
points = load_data(file_path)
print(points)
