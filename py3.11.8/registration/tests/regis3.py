import numpy as np
import numpy.linalg as LA

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

# 四元数から回転行列への変換
def quaternion2rotation( q ):
    rot = np.array([
        [
            q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2,
            2.0 * (q[1]*q[2] - q[0]*q[3]),
            2.0 * (q[1]*q[3] + q[0]*q[2])
        ],
        
        [
            2.0 * (q[1]*q[2] + q[0]*q[3]),
            q[0]**2 + q[2]**2 - q[1]**2 - q[3]**2,
            2.0 * (q[2]*q[3] - q[0]*q[1])
        ],
        
        [
            2.0 * (q[1]*q[3] - q[0]*q[2]),
            2.0 * (q[2]*q[3] + q[0]*q[1]),
            q[0]**2 + q[3]**2 - q[1]**2 - q[2]**2
        ]
    ])

    return rot

q = np.array([1., 0., 0., 0., 0., 0., 0.])
rot = quaternion2rotation(q)
# print(rot)


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
# query_points = source_points


sets = np.zeros(( len(query_points), 2, len(query_points[0]) )) # 対応点のセット(クエリ点, 最近傍点)
sum_query = np.zeros(( query_points.shape[1])) #クエリ点の合計
sum_nearest = np.zeros(( target_points.shape[1])) #探索した最近傍点の合計
# nearest_points = np.zeros((query_point.shape[0], query_point.shape[1]))


nearest_points = np.zeros(( query_points.shape[0], query_points.shape[1]))

for k in range(10):

    
    for index, query_point in enumerate(query_points):
        nearest_point = find_nearest(tree, query_point)
        sets[index] = np.array([query_point, nearest_point])
        # print(query_point)
        # print(nearest_point)
        # print("\n")
        nearest_points[index, :] = nearest_point

        # sum_query += query_point #np.meanを使うため、使用しない
        # sum_nearest += nearest_point


    # 重心の計算

    mu_s = np.mean( query_points, axis=0 ) #クエリ点（ソース点群）の重心
    mu_y = np.mean( nearest_points, axis=0 ) #最近傍点の重心
    # print(mu_s)
    # print(mu_y)

    # 共分散行列
    covar = np.zeros( (3, 3) )
    n_points = query_points.shape[0]
    for i in range(n_points):
        p = query_points[i] - mu_s
        y = nearest_points[i] - mu_y
        covar += np.outer(p, y)
    covar /= n_points
    # print(covar)



    A = covar - covar.T
    delta = np.array( [A[1, 2], A[2, 0], A[0, 1]] )
    tr_covar = np.trace(covar)
    i3d = np.identity(3)

    N_py = np.zeros( (4, 4) )
    N_py[0, 0] = tr_covar
    N_py[0, 1:4] = delta
    N_py[1:4, 0] = delta
    N_py[1:4, 1:4] = covar + covar.T - tr_covar*i3d
    # print(N_py)

    # 固有値・固有ベクトル
    w, v = LA.eig(N_py)
    rot = quaternion2rotation(v[:, np.argmax(w)])
    # print("固有値\n", w)
    # print("固有ベクトル\n", v)
    # print("最大固有値に対応する固有ベクトル\n", v[:, np.argmax(w)])
    # print("回転行列\n", rot)
    # 平行移動成分
    trans = mu_y - np.dot(rot, mu_s)

    # 4x4同次変換行列
    transform = np.identity(4)
    transform[0:3, 0:3] = rot.copy()
    transform[0:3, 3] = trans.copy()
    # print("剛体変換行列\n", transform)

    # source_points2 = np.hstack( ( source_points, np.ones( ( source_points.shape[0], 1 ) ) ) )
    # transformed_points = np.dot( source_points2, transform )[:, :3]
    query_points2 = np.hstack( ( query_points, np.ones( ( query_points.shape[0], 1 ) ) ) )
    transformed_points = np.dot( query_points2, transform )[:, :3]
    distances = np.linalg.norm( nearest_points - transformed_points )
    rmse = np.sqrt(np.mean(distances**2))
    # print(distances)
    print("RMSE:", rmse)

    query_points = transformed_points
