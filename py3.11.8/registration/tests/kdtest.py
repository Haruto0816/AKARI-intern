import numpy as np

class KDNode:
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right

def build_kdtree(points, depth=0):
    if not points.size:  # 空の配列をチェック
        return None

    k = points.shape[1]  # 3次元点群
    axis = depth % k

    # NumPyを用いた軸に沿ったソート
    indices = np.argsort(points[:, axis])
    points = points[indices]
    median = len(points) // 2

    # 再帰的に左右の子を構築
    return KDNode(
        point=points[median],
        left=build_kdtree(points[:median], depth + 1),
        right=build_kdtree(points[median + 1:], depth + 1)
    )

# 例として3次元のランダムな点群を生成
points = np.random.rand(10, 3)

# KD-treeを構築
tree = build_kdtree(points)
