import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt


class KDNode:
    def __init__(self, point, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            point_data = list(map(float, parts[:3]))
            data.append(point_data)
    return np.array(data)

def build_kdtree(points, depth=0):
    if not points.size:
        return None
    k = points.shape[1]
    axis = depth % k
    sorted_indices = np.argsort(points[:, axis])
    median = len(points) // 2
    return KDNode(
        point=points[sorted_indices[median]],
        left=build_kdtree(points[sorted_indices[:median]], depth + 1),
        right=build_kdtree(points[sorted_indices[median + 1:]], depth + 1)
    )

def find_nearest(node, point, depth=0, best=None):
    if node is None:
        return best
    k = len(point)
    axis = depth % k
    next_branch = node.left if point[axis] < node.point[axis] else node.right
    opposite_branch = node.right if point[axis] < node.point[axis] else node.left
    best = find_nearest(next_branch, point, depth + 1, best)
    if best is None or np.linalg.norm(point - node.point) < np.linalg.norm(point - best):
        best = node.point
    if opposite_branch is not None and np.abs(point[axis] - node.point[axis]) < np.linalg.norm(point - best):
        best = find_nearest(opposite_branch, point, depth + 1, best)
    return best

def quaternion2rotation(q):
    return np.array([
        [q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2, 2 * (q[1]*q[2] - q[0]*q[3]), 2 * (q[1]*q[3] + q[0]*q[2])],
        [2 * (q[1]*q[2] + q[0]*q[3]), q[0]**2 + q[2]**2 - q[1]**2 - q[3]**2, 2 * (q[2]*q[3] - q[0]*q[1])],
        [2 * (q[1]*q[3] - q[0]*q[2]), 2 * (q[2]*q[3] + q[0]*q[1]), q[0]**2 + q[3]**2 - q[1]**2 - q[2]**2]
    ])

source_points = load_data('workspace1.txt')
target_points = load_data('workspace2.txt')
tree = build_kdtree(target_points)

num = 60000
indices = np.random.choice(source_points.shape[0], num, replace=False)
query_points = source_points[indices]

RMSE = []

for k in range(20):
    nearest_points = np.array([find_nearest(tree, qp) for qp in query_points])

    mu_s = np.mean(query_points, axis=0)
    mu_y = np.mean(nearest_points, axis=0)

    #covar = np.sum([np.outer((qp - mu_s), (n_p - mu_y)) for qp, n_p in zip(query_points, nearest_points)], axis=0) / num
    # 共分散行列
    covar = np.zeros( (3, 3) )
    n_points = query_points.shape[0]
    for i in range(n_points):
        p = query_points[i] - mu_s
        y = nearest_points[i] - mu_y
        covar += np.outer(p, y)
    covar /= n_points

    # delta = np.array([covar[1, 2] - covar[2, 1], covar[2, 0] - covar[0, 2], covar[0, 1] - covar[1, 0]])
    # tr_covar = np.trace(covar)
    # Q = np.zeros((4, 4))
    # Q[0, 0] = tr_covar
    # Q[0, 1:4] = Q[1:4, 0] = delta
    # Q[1:4, 1:4] = covar + covar.T - np.eye(3) * tr_covar
    # _, v = np.linalg.eig(Q)
    # rot = quaternion2rotation(v[:, np.argmax(np.real(_))])

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

    # trans = mu_y - rot @ mu_s
    # transform = np.eye(4)
    # transform[:3, :3] = rot
    # transform[:3, 3] = trans

    # 平行移動成分
    trans = mu_y - np.dot(rot, mu_s)

    # 4x4同次変換行列
    transform = np.identity(4)
    transform[0:3, 0:3] = rot.copy()
    transform[0:3, 3] = trans.copy()

    query_points = (transform @ np.hstack((query_points, np.ones((query_points.shape[0], 1)))).T).T[:, :3]
    # rmse = np.sqrt(np.mean(np.sum((nearest_points - query_points)**2, axis=1)))

    # query_points2 = np.hstack( ( query_points, np.ones( ( query_points.shape[0], 1 ) ) ) )
    # transformed_points = np.dot( query_points2, transform )[:, :3]
    # query_points = transformed_points

    distances = np.linalg.norm( nearest_points - query_points, axis=1 )
    rmse = np.sqrt(np.mean(distances**2))
    RMSE.append((k, rmse))
    print(f"RMSE: {rmse}")

x, y = zip(*RMSE)
plt.plot(x, y)
plt.savefig('myplot.png')