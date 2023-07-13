from typing import List
import numpy as np
import matplotlib.pyplot as plt

from kdtree import KDTree
from environment import Map, make_env

class RRTNode:
    def __init__(self, point: np.array, parent: 'RRTNode'=None) -> None:
        self.point = point
        self.parent = parent
        self.children: List['RRTNode']=[]

    def add_child(self, node: 'RRTNode') -> None:
        self.children.append(node)

    def render(self, ax: plt.Axes) -> None:
        ax.scatter(self.point[0], self.point[1], c="tab:blue")
        for child in self.children:
            ax.plot([self.point[0], child.point[0]], [self.point[1], child.point[1]], c="tab:blue")
            child.render(ax)

class RRT:
    def __init__(self, env: Map, root: RRTNode=None) -> None:
        self.env = env

        if root is not None:
            self.root = root
        else:
            candidate_root = np.random.random(2)
            while not self.env.point_free(candidate_root):
                candidate_root = np.random.random(2)
            self.root = RRTNode(candidate_root)
        
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

    def grow(self) -> bool:
        random_point = np.random.random(2)
        pt, dist, nearest_node = self.point_search_tree.nearest_neighbor(random_point)
        direction = (random_point-pt) / dist
        new_point = pt + min(dist, 0.1)*direction
        new_point = self.env.boundary_point(pt, new_point, s_pct=0.95)
        if ((new_point-pt)**2).sum() < 0.0001:
            return False
        new_node = RRTNode(new_point, nearest_node)
        nearest_node.add_child(new_node)
        self.point_search_tree.put(new_point, new_node)
        return True

if __name__ == "__main__":
    env = make_env()
    env.set_goal(np.random.random(2))
    env.set_tol(0.05)

    rrt = RRT(env)
    for _ in range(1000):
        success = rrt.grow()
        while not success:
            success = rrt.grow()
        if _%100 == 0:
            rrt.point_search_tree.rebalance()
            fig, ax = plt.subplots()
            rrt.render(ax)
            plt.show()
    