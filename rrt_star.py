from typing import List
import numpy as np
import matplotlib.pyplot as plt

from kdtree import KDTree
from environment import Map, make_env

class RRTStarNode:
    def __init__(self, point: np.array, parent: 'RRTStarNode'=None) -> None:
        self.point = point
        self.parent = parent
        self.cost = 0
        if self.parent is not None:
            self.cost = self.parent.cost + np.linalg.norm(point-parent.point, 2)
        self.children: List['RRTStarNode']=[]

    def add_child(self, node: 'RRTStarNode') -> None:
        self.children.append(node)

    def remove_child(self, node: 'RRTStarNode') -> None:
        self.children = [child for child in self.children if child!=node]

    def propagate_cost(self, delta: float) -> None:
        self.cost += delta
        for child in self.children:
            child.propagate_cost(delta)

    def render(self, ax: plt.Axes) -> None:
        ax.scatter(self.point[0], self.point[1], c="tab:blue")
        for child in self.children:
            ax.plot([self.point[0], child.point[0]], [self.point[1], child.point[1]], c="tab:blue")
            child.render(ax)

class RRTStar:
    def __init__(self, env: Map, root: RRTStarNode=None) -> None:
        self.env = env

        if root is not None:
            self.root = root
        else:
            candidate_root = np.random.random(2)
            while not self.env.point_free(candidate_root):
                candidate_root = np.random.random(2)
            self.root = RRTStarNode(candidate_root)
        
        self.num_points = 1
        self.point_search_tree = KDTree(self.root.point[np.newaxis,...], [self.root])

    def render_path(self, ax: plt.Axes) -> None:
        pt, dist, nearest_node = self.point_search_tree.nearest_neighbor(self.env.goal)
        if dist > self.env.tol:
            return
        while nearest_node.parent is not None:
            ax.plot([nearest_node.point[0], nearest_node.parent.point[0]], [nearest_node.point[1], nearest_node.parent.point[1]], c="tab:red")
            nearest_node = nearest_node.parent

    def render(self, ax: plt.Axes) -> None:
        self.env.render(ax)
        self.root.render(ax)
        self.render_path(ax)

    def rewire(self, node: RRTStarNode, new_parent: RRTStarNode, dist: float) -> None:
        node.parent.remove_child(node)
        new_parent.add_child(node)
        node.parent = new_parent
        cost_delta = new_parent.cost + dist - node.cost
        node.propagate_cost(cost_delta)

    def grow(self) -> bool:
        random_point = np.random.random(2)
        while not self.env.point_free(random_point):
            random_point = np.random.random(2)

        pt, dist, nearest_node = self.point_search_tree.nearest_neighbor(random_point)
        direction = (random_point-pt) / dist
        new_point = pt + min(dist, 0.1)*direction
        
        if not self.env.path_free(pt, new_point):
            return False
        
        k = np.e * (1 + (1/self.root.point.size)) * np.log(self.num_points)
        k = max(1, k)
        points_near = self.point_search_tree.k_nearest_neighbor(new_point, np.ceil(k))

        # Connect node to neighbor which minimizes cost
        min_cost = np.inf
        parent = None
        for coords, dist, node in points_near:
            cost = dist + node.cost
            if cost < min_cost and self.env.path_free(coords, new_point):
                min_cost = cost
                parent = node
        new_node = RRTStarNode(new_point, parent)
        parent.add_child(new_node)
        self.point_search_tree.put(new_point, new_node)
        self.num_points += 1

        # Among neighbors, rewire if it would reduce cost
        for coords, dist, node in points_near:
            if new_node.cost + dist < node.cost and self.env.path_free(coords, new_point):
                self.rewire(node, new_node, dist)

        return True
    
if __name__ == "__main__":
    env = make_env()
    env.set_goal(np.random.random(2))
    env.set_tol(0.05)

    rrt_star = RRTStar(env)
    for _ in range(1000):
        success = rrt_star.grow()
        while not success:
            success = rrt_star.grow()
        if _%100 == 0:
            rrt_star.point_search_tree.rebalance()
            fig, ax = plt.subplots()
            rrt_star.render(ax)
            plt.show()