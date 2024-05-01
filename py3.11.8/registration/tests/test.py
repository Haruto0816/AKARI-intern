import numpy as np

# データを読み込む関数
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # 各行を空白で分割し、浮動小数点数のリストに変換
            parts = line.split()
            point_data = list(map(float, parts[:3]))
            data.append(point_data)
    return np.array(data)

a = load_data("workspace1.txt")
print(a[0])