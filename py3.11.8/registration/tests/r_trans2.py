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
            parts = line.split()
            point_data = list(map(float, parts[:3])) # x,y,zのみを抽出
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

def centroid(points):
    return np.mean(points, axis=0)

# ソース点群
source_path = 'workspace1.txt'
source_points = load_data(source_path)

# ターゲット点群
target_path = 'workspace2.txt'
target_points = load_data(target_path)

# ターゲット点群のKD-treeを構築
tree = build_kdtree(target_points)

# 対応点ペアを生成
num = 3 # 選ぶ数
indices = np.random.choice(source_points.shape[0], num, replace = False)
query_points = source_points[indices]


sets = np.zeros(( len(query_points), 2, len(query_points[0]) )) # 対応点のセット(クエリ点, 最近傍点)
sum_query = np.zeros(( query_points.shape[1])) #クエリ点の合計
sum_nearest = np.zeros(( target_points.shape[1])) #探索した最近傍点の合計
# nearest_points = np.zeros((query_point.shape[0], query_point.shape[1]))

for index, query_point in enumerate(query_points):
    nearest_point = find_nearest(tree, query_point)
    sets[index] = np.array([query_point, nearest_point])
    print(query_point)
    print(nearest_point)
    print("\n")

    sum_query += query_point
    sum_nearest += nearest_point

print(sum_query)
print(sum_nearest)

# 重心の計算
centroid_query = sum_query / num
centroid_nearest = sum_nearest / num
#print( centroid_query )
#print( centroid_nearest )

# 重心を原点に合わせる(元の座標 - 重心)
query_points = sets[:,0] - centroid_query
nearest_points = sets[:,1] - centroid_nearest

#for i, v in enumerate(sets):
#    m_query += sets[i][0]
#    m_nearest += sets[i][1]

#m_query = m_query / (m_query.shape[0]) # 重心を計算
#m_nearest = m_nearest / (m_nearest.shape[0]) #重心を計算


# for i, v in enumerate(sets):
    # print(sets[i,0])

    # print(query_points[i])


# 最近傍点を探索
#print("Query Point:", query_point)
#print("Nearest Point:", nearest_point)
