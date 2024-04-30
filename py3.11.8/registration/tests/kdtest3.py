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

def find_nearest(node, point, depth=0, best=None):
    if node is None:
        return best

    k = len(point)
    axis = depth % k

    next_branch = None
    opposite_branch = None

    if point[axis] < node.point[axis]:
        next_branch = node.left
        opposite_branch = node.right
    else:
        next_branch = node.right
        opposite_branch = node.left

    best = find_nearest(next_branch, point, depth + 1, best)

    if best is None or (np.linalg.norm(point - node.point) < np.linalg.norm(point - best)):
        best = node.point

    if opposite_branch is not None:
        if np.abs(point[axis] - node.point[axis]) < np.linalg.norm(point - best):
            best = find_nearest(opposite_branch, point, depth + 1, best)

    return best



# 3次元の点群データ
points = np.random.rand(10, 3)  # 10個のランダムな3次元点を生成
print(points)

# KD-treeを構築
tree = build_kdtree(points)

# KD-treeを表示
print("KD-Tree structure:")
print_kdtree(tree)

query_point = np.random.rand(3) # 探索したいクエリ点

# 最近傍点を探索
nearest_point = find_nearest(tree, query_point)
print("Query Point:", query_point)
print("Nearest Point:", nearest_point)
