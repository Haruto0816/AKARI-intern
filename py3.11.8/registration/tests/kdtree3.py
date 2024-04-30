import numpy as np

class KDNode:
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right

# データを読み込む関数
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # 各行を空白で分割し、浮動小数点数のリストに変換
            point_data = list(map(float, line.split()))
            data.append(point_data)
    return np.array(data)

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


# ソース点群
source_path = 'workspace1.txt'
source_points = load_data(source_path)

# ターゲット点群
target_path = 'workspace2.txt'
target_points = load_data(target_path)

# KD-treeを構築
tree = build_kdtree(target_points)

# 対応点ペアを生成
indices = np.random.choice(source_points.shape[0], 3, replace = False)
selected_points = source_points[indices]

sets = np.zeros(( len(selected_points), 2, len(selected_points[0]) ))
for index, query_point in enumerate(selected_points):
    nearest_point = find_nearest(tree, query_point)
    sets[index] = np.array([query_point, nearest_point])


for i, v in enumerate(sets):
    print(sets[i])
# 最近傍点を探索
#print("Query Point:", query_point)
#print("Nearest Point:", nearest_point)
