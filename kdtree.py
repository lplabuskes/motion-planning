from typing import Tuple, List, Any
import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, point: np.array, depth: int, metadata: Any=None, left: 'Node'=None, right: 'Node'=None) -> None:
        self.point = point
        self.metadata = metadata
        self.k = self.point.shape[0]
        self.depth = depth
        self.left = left
        self.right = right

    def min(self, dim: int) -> float:
        min_left = np.inf if self.left is None else self.left.min(dim)
        min_right = np.inf if self.right is None else self.right.min(dim)
        return np.min([min_left, min_right, self.point[dim]])
    
    def max(self, dim: int) -> float:
        max_left = -np.inf if self.left is None else self.left.max(dim)
        max_right = -np.inf if self.right is None else self.right.max(dim)
        return np.max([max_left, max_right, self.point[dim]])
          
    def all_points(self) -> Tuple[List[np.array], List[Any]]:
        left = ([], []) if self.left is None else self.left.all_points()
        right = ([], []) if self.right is None else self.right.all_points()
        points = left[0]+right[0]+[self.point]
        meta = left[1]+right[1]+[self.metadata]
        return points, meta

    def num_points(self) -> int:
        left_points = 0 if self.left is None else self.left.num_points()
        right_points = 0 if self.right is None else self.right.num_points()
        self_points = 0 if self.point is None else 1
        return left_points + right_points + self_points
    
    def num_nodes(self) -> int:
        left_nodes = 0 if self.left is None else self.left.num_nodes()
        right_nodes = 0 if self.right is None else self.right.num_nodes()
        return 1 + left_nodes + right_nodes
    
    def validate(self) -> bool:
        if self.left is None and self.right is None:
            return True
        elif self.left is None:
            return self.right.validate() and self.point[self.depth%self.k] <= self.right.min(self.depth%self.k)
        elif self.right is None:
            return self.left.validate() and self.point[self.depth%self.k] >= self.left.max(self.depth%self.k)
        else:
            children_valid = self.left.validate() and self.right.validate()
            self_valid = self.left.max(self.depth%self.k) <= self.point[self.depth%self.k] <= self.right.min(self.depth%self.k)
            return children_valid and self_valid
        
    def rectangle_search(self, query: np.array) -> Tuple[List[np.array], List[Any]]:
        # Rectangles are represented by a 4d point (xmin, xmax, ymin, ymax)
        node_dim = self.depth%self.k
        query_dim = node_dim^1  # flip so you compare xmin to xmax, ymin to ymax

        # Query min greater than splitting value for max, prune this node and left branch
        if query_dim%2 == 0 and query[query_dim] > self.point[node_dim]:
            return ([], []) if self.right is None else self.right.rectangle_search(query)
        
        # Query max less than splitting value for min, prune this node and right branch
        if query_dim%2 == 1 and query[query_dim] < self.point[node_dim]:
            return ([], []) if self.left is None else self.left.rectangle_search(query)
        
        # Otherwise, can't definitively prune
        left_results = ([], []) if self.left is None else self.left.rectangle_search(query)
        right_results = ([], []) if self.right is None else self.right.rectangle_search(query)
        points = left_results[0]+right_results[0]
        meta = left_results[1]+right_results[1]
        
        add_point = True
        for i in range(0, self.k, 2):
            if query[i] > self.point[i+1]:
                add_point = False
                break
            if self.point[i] > query[i+1]:
                add_point = False
                break
        if add_point:
            points.append(self.point)
            meta.append(self.metadata)

        return points, meta

def update_knn_list(new_entry: Tuple[np.array, float, Any], tracking: List[Tuple[np.array, float, Any]], k: int) -> List[Tuple[np.array, float, Any]]:
    if len(tracking) >= k and new_entry[1] >= tracking[-1][1]:
        return tracking
    if len(tracking) == 0:
        return [new_entry]
    for idx, entry in enumerate(tracking):
        if new_entry[1] < entry[1]:
            tracking.insert(idx, new_entry)
            break
    if len(tracking) > k:
        tracking.pop()
    return tracking

class KDTree:
    def __init__(self, points: np.array, metadata: List[Any]=None) -> None:
        assert points.ndim == 2
        assert metadata is None or (isinstance(metadata, List) and len(metadata)==points.shape[0])
        self.root = self._build_tree(points, metadata)
 
    def _build_tree(self, points: np.array, metadata: List[Any], depth: int=0) -> Node:
        if points.size == 0:
            return None
        k = points.shape[1]
        dim = depth % k
        sort_idxs = points[:,dim].argsort()
        sorted_points = points[sort_idxs]
        sorted_meta = [None]*points.shape[0] if metadata is None else [metadata[i] for i in sort_idxs]
        median_index = sorted_points.shape[0] // 2
        return Node(
            sorted_points[median_index],
            depth,
            sorted_meta[median_index],
            self._build_tree(sorted_points[:median_index, :], sorted_meta[:median_index], depth + 1),
            self._build_tree(sorted_points[median_index+1:, :], sorted_meta[median_index+1:], depth + 1)
        )

    def min(self, dim: int=0) -> float:
        assert 0 <= dim < self.root.point.shape[0]
        return self.root.min(dim)
    
    def max(self, dim: int=0) -> float:
        assert 0 <= dim < self.root.point.shape[0]
        return self.root.max(dim)

    def put(self, point: np.array, metadata: Any=None) -> None:
        assert point.shape == self.root.point.shape

        k = point.shape[0]
        node = self.root
        while node is not None:  # This condition should never happen but better safe than sorry
            if point[node.depth%k] <= node.point[node.depth%k]:
                if node.left is None:
                    node.left = Node(point, node.depth+1, metadata)
                    break
                else:
                    node = node.left
            else:
                if node.right is None:
                    node.right = Node(point, node.depth+1, metadata)
                    break
                else:
                    node = node.right

    def all_points(self) -> Tuple[List[np.array], List[Any]]:
        return self.root.all_points()

    def num_points(self) -> int:
        return self.root.num_points()
    
    def num_nodes(self) -> int:
        return self.root.num_nodes()

    def validate(self) -> bool:
        return self.root.validate()

    def nearest_neighbor(self, point: np.array) -> Tuple[np.array, float, Any]:
        best, dist2, meta = self._nn_helper(point, self.root)
        return best, np.sqrt(dist2), meta

    def _nn_helper(self, point: np.array, node: Node, best: np.array=None, best_dist2: float=np.inf, best_meta: Any=None) -> Tuple[np.array, float, Any]:
        if node is None:
            return best, best_dist2, best_meta
        dist2 = ((point-node.point)**2).sum()
        if best is None or dist2 < best_dist2:
            best = node.point
            best_dist2 = dist2
            best_meta = node.metadata

        dim = node.depth % node.k

        if point[dim] <= node.point[dim]:
            best, best_dist2, best_meta = self._nn_helper(point, node.left, best, best_dist2, best_meta)
            if best_dist2 > (point[dim]-node.point[dim])**2:  # can nn be on the other side of the splitting plane
                best, best_dist2, best_meta = self._nn_helper(point, node.right, best, best_dist2, best_meta)
        else:
            best, best_dist2, best_meta = self._nn_helper(point, node.right, best, best_dist2, best_meta)
            if best_dist2 > (point[dim]-node.point[dim])**2:  # can nn be on the other side of the splitting plane
                best, best_dist2, best_meta = self._nn_helper(point, node.left, best, best_dist2, best_meta)

        return best, best_dist2, best_meta

    def k_nearest_neighbor(self, point: np.array, k: int) -> List[Tuple[np.array, float, Any]]:
        results = self._knn_helper(point, self.root, k)
        for idx, entry in enumerate(results):
            results[idx] = (entry[0], np.sqrt(entry[1]), entry[2])
        return results

    def _knn_helper(self, point: np.array, node: Node, k: int, tracking: List[Tuple[np.array, float, Any]]=[]) -> List[Tuple[np.array, float, Any]]:
        if node is None:
            return tracking
        dist2 = ((point-node.point)**2).sum()
        entry = (node.point, dist2, node.metadata)
        tracking = update_knn_list(entry, tracking, k)

        dim = node.depth % node.k

        if point[dim] <= node.point[dim]:
            tracking = self._knn_helper(point, node.left, k, tracking)
            if tracking[-1][1] > (point[dim]-node.point[dim])**2:  # can knn be on the other side of the splitting plane
                tracking = self._knn_helper(point, node.right, k, tracking)
        else:
            tracking = self._knn_helper(point, node.right, k, tracking)
            if tracking[-1][1] > (point[dim]-node.point[dim])**2:  # can knn be on the other side of the splitting plane
                tracking = self._knn_helper(point, node.left, k, tracking)

        return tracking

    def rectangle_search(self, query: np.array) -> Tuple[List[np.array], List[Any]]:
        return self.root.rectangle_search(query)

    def rebalance(self) -> None:
        pts, meta = self.root.all_points()
        pts = np.stack(pts, axis=0)
        self.root = self._build_tree(pts, meta)

def render_query(query: np.array, tree: KDTree) -> None:
    fig, ax = plt.subplots()
    if tree is None:
        rpatch = plt.Rectangle((query[0], query[2]), query[1]-query[0], query[3]-query[2], color="tab:green", fill=False)
        ax.add_patch(rpatch)
    else:
        all_rects, _ = tree.all_points()
        for r in all_rects:
            rpatch = plt.Rectangle((r[0], r[2]), r[1]-r[0], r[3]-r[2], color="k", alpha=0.5, fill=True)
            ax.add_patch(rpatch)
        overlapping, _ = tree.rectangle_search(query)
        for r in overlapping:
            rpatch = plt.Rectangle((r[0], r[2]), r[1]-r[0], r[3]-r[2], color="tab:blue", fill=True)
            ax.add_patch(rpatch)
        qcolor = "tab:green" if len(overlapping)==0 else "tab:red"
        rpatch = plt.Rectangle((query[0], query[2]), query[1]-query[0], query[3]-query[2], color=qcolor, fill=False)
        ax.add_patch(rpatch)
    plt.show()
        

if __name__ == "__main__":
    points = np.random.random((2000,2))
    names = [str(row) for row in points]
    tree = KDTree(points, names)

    search_point = np.random.random((2))
    knn_results = tree.k_nearest_neighbor(search_point, 10)
    nn_results = tree.nearest_neighbor(search_point)

    dist = knn_results[-1][1]
    angles = np.linspace(0, 2*np.pi)
    circ_points = dist*np.stack([np.cos(angles), np.sin(angles)], axis=-1) + search_point

    plt.scatter(points[:,0], points[:,1], alpha=0.25, c="k")
    plt.scatter(search_point[0], search_point[1], c="tab:green")
    # plt.scatter(nn[0], nn[1], c="tab:red")
    plt.plot(circ_points[:,0], circ_points[:,1], c="tab:blue")
    plt.show()
