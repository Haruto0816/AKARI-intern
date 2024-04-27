import numpy as np

class KDNode:
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right

def build_kdtree(points, depth=0):
    if not points.size:
        return None
    
    k = points.shape[1]
    axis = depth % k

    indices = np.argsort(points[:, axis])
    points = points[indices]
    median = len(points) // 2

    return KDNode(
        point=points[median],
        left=build_kdtree(points[:median], depth + 1),
        right=build_kdtree(points[median + 1:], depth + 1)
    )

def print_kdtree(node, depth=0):
    if node is not None:
        # ノードの点と深さに応じたインデントを表示
        print(' ' * (depth * 4) + f"Depth {depth}: {node.point}")
        # 左の子を再帰的に表示
        print_kdtree(node.left, depth + 1)
        # 右の子を再帰的に表示
        print_kdtree(node.right, depth + 1)

# 3次元の点群データ
points = np.random.rand(10, 3)  # 10個のランダムな3次元点を生成
print(points)
# KD-treeを構築
tree = build_kdtree(points)

# KD-treeを表示
print("KD-Tree structure:")
print_kdtree(tree)
