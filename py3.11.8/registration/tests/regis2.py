import numpy as np
import numpy.linalg as LA

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

num = 10000
indices = np.random.choice(source_points.shape[0], num, replace=False)
query_points = source_points[indices]

for k in range(10):
    nearest_points = np.array([find_nearest(tree, qp) for qp in query_points])

    mu_s = np.mean(query_points, axis=0)
    mu_y = np.mean(nearest_points, axis=0)
    #covar = np.sum([np.outer((qp - mu_s), (np - mu_y)) for qp, np in zip(query_points, nearest_points)], axis=0) / num
    covar = np.sum([np.outer((qp - mu_s), (n_p - mu_y)) for qp, n_p in zip(query_points, nearest_points)], axis=0) / num

    delta = np.array([covar[1, 2] - covar[2, 1], covar[2, 0] - covar[0, 2], covar[0, 1] - covar[1, 0]])
    tr_covar = np.trace(covar)
    Q = np.zeros((4, 4))
    Q[0, 0] = tr_covar
    Q[0, 1:4] = Q[1:4, 0] = delta
    Q[1:4, 1:4] = covar + covar.T - np.eye(3) * tr_covar
    _, v = np.linalg.eig(Q)
    rot = quaternion2rotation(v[:, np.argmax(np.real(_))])

    trans = mu_y - rot @ mu_s
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = trans

    query_points = (transform @ np.hstack((query_points, np.ones((query_points.shape[0], 1)))).T).T[:, :3]
    rmse = np.sqrt(np.mean(np.sum((nearest_points - query_points)**2, axis=1)))
    print(f"RMSE: {rmse}")
