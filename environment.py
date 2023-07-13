from abc import ABC, abstractmethod
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np

from kdtree import KDTree


class Shape(ABC):
    @abstractmethod
    def render(self, ax: plt.Axes) -> None:
        # Used by the GUI to render the object
        raise NotImplementedError()
    
    @abstractmethod
    def contains(self, p: np.array) -> bool:
        # Return if a point p falls within the shape
        raise NotImplementedError()
    
    @abstractmethod
    def line_intersect(self, a: np.array, b: np.array) -> bool:
        # Determine if the line segment AB intersects with the shape
        raise NotImplementedError()
    
    @abstractmethod
    def boundary_point(self, a: np.array, b: np.array) -> Tuple[np.array, float]:
        # If AB does not intersect with the shape, return B
        # Otherwise, return the point where you hit the shape 
        #   boundary when going from A to B in a straight line
        # For convenience, also return s in [0,1] which is the fraction gone from A to B
        raise NotImplementedError()
    
    @abstractmethod
    def bounding_rectangle(self) -> np.array:
        # The (minimal) axes aligned bounding box which surrounds the shape
        raise NotImplementedError()
    
class Rectangle(Shape):
    def __init__(self, x, y, w, h, theta=0) -> None:
        # (x, y) is the center of the rectangle
        # theta is in radians, CCW positive
        corners = np.array([[-w/2,w/2,w/2,-w/2],[-h/2,-h/2,h/2,h/2]])
        if theta != 0:
            rot = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            corners = rot@corners
        self.theta = theta
        self.corners = (np.array([[x],[y]]) + corners).T
        self.edge_vecs = [self.corners[i+1,:]-self.corners[i,:] for i in range(3)] + [self.corners[0,:]-self.corners[3,:]]
        self._bounding_rectangle = np.array([self.corners[:,0].min(), self.corners[:,0].max(), self.corners[:,1].min(), self.corners[:,1].max()])

    def render(self, ax: plt.Axes) -> None:
        ax.fill(self.corners[:,0], self.corners[:,1], c="tab:gray")

    def contains(self, p: np.array) -> bool:
        if self.theta == 0:
            return self._bounding_rectangle[0] <= p[0] <= self._bounding_rectangle[1] and self._bounding_rectangle[2] <= p[1] <= self._bounding_rectangle[3]
        
        # Because of fixed order of vertices, point must be to the "left" of each edge
        for idx, v in enumerate(self.edge_vecs):
            v_cp = p-self.corners[idx,:]
            if np.cross(v, v_cp) < 0:
                return False
        return True

    def line_intersect(self, a: np.array, b: np.array) -> bool:
        # either point is in the rectangle
        if self.contains(a) or self.contains(b):
            return True
        
        # line intersects with an edge
        line_vec = b-a
        for idx, edge in enumerate(self.edge_vecs):
            # line parallel to edge -> would get caught by contain check
            if edge[0]*line_vec[1] == line_vec[0]*edge[1]:
                continue
            # line intersects with edge
            coef = 1/(edge[0]*line_vec[1]-edge[1]*line_vec[0])
            mat = np.array([[line_vec[1], -line_vec[0]],[-edge[1], edge[0]]])
            vec = (a-self.corners[idx, :])[...,np.newaxis]
            sol = coef*(mat@vec)
            if 0 <= sol[0] <= 1 and -1 <= sol[1] <= 0:
                return True
        
        return False
    
    def boundary_point(self, a: np.array, b: np.array) -> Tuple[np.array, float]:
        if self.contains(a):
            raise RuntimeError("a is inside the rectangle")
        
        line_vec = b-a
        min_s = 1
        for idx, edge in enumerate(self.edge_vecs):
            # line parallel to edge
            if edge[0]*line_vec[1] == line_vec[0]*edge[1]:
                continue
            # line intersects with edge
            coef = 1/(edge[0]*line_vec[1]-edge[1]*line_vec[0])
            mat = np.array([[line_vec[1], -line_vec[0]],[-edge[1], edge[0]]])
            vec = (a-self.corners[idx, :])[...,np.newaxis]
            sol = coef*(mat@vec)
            if 0 <= sol[0] <= 1 and -1 <= sol[1] <= 0:
                min_s = min(min_s, -sol[1][0])
        
        return a + min_s*line_vec, min_s
    
    def bounding_rectangle(self) -> np.array:
        return self._bounding_rectangle

class Circle(Shape):
    def __init__(self, x: float, y: float, r: float) -> None:
        self.center = np.array([x, y])
        self.r = r
        self.r2 = r**2
        self._bounding_rectangle = np.array([x-r, x+r, y-r, y+r])

    def render(self, ax: plt.Axes) -> None:
        circle = plt.Circle((self.center[0], self.center[1]), self.r, color="tab:gray")
        ax.add_patch(circle)

    def contains(self, p: np.array) -> bool:
        return ((p-self.center)**2).sum() <= self.r2

    def line_intersect(self, a: np.array, b: np.array) -> bool:
        if self.contains(a) or self.contains(b):
            return True
        
        diff = a-self.center
        line_vec = b-a

        # line: u+s*v
        # circle: p st ||p-c||**2 < r**2
        # min quadratic in s over [0,1]
        coeff_a = (line_vec**2).sum()
        coeff_b = 2*np.dot(diff, line_vec)
        coeff_c = (diff**2).sum()

        # Only care if global min s in (0, 1) because 0 and 1 and checked at beginning of function
        s = -coeff_b/(2*coeff_a)
        if 0 <= s <= 1:
            return coeff_a*(s**2)+coeff_b*s+coeff_c<=self.r2
        return False

    def boundary_point(self, a: np.array, b: np.array) -> Tuple[np.array, float]:
        if self.contains(a):
            raise RuntimeError("a is in the circle")
        
        diff = a-self.center
        line_vec = b-a

        # line: u+s*v
        # circle: p st ||p-c||**2 < r**2
        # min quadratic in s over [0,1]
        coeff_a = (line_vec**2).sum()
        coeff_b = 2*np.dot(diff, line_vec)
        coeff_c = (diff**2).sum() - self.r2

        roots = np.roots([coeff_a, coeff_b, coeff_c])
        min_s = 1
        for s in roots:
            if not np.isreal(s):
                continue
            if 0<=s<=1:
                min_s=min(min_s,s)
        return a + min_s*line_vec, min_s

    def bounding_rectangle(self) -> np.array:
        return self._bounding_rectangle

class Map:
    def __init__(self, obstacles: List[Shape]=[]) -> None:
        self.obstacles = obstacles
        self.obstacle_tree = None
        if len(obstacles) > 0:
            rects = np.stack([obstacle.bounding_rectangle() for obstacle in obstacles], axis=0)
            self.obstacle_tree = KDTree(rects, obstacles)
        self.goal: np.array = None
        self.tol: float = None

    def render(self, ax: plt.Axes):
        if self.goal is not None and self.tol is not None:
            circle = plt.Circle((self.goal[0], self.goal[1]), self.tol, color="tab:green", alpha=0.25)
            ax.add_patch(circle)
        for obstacle in self.obstacles:
            obstacle.render(ax)

    def add_obstacle(self, obstacle: Shape) -> None:
        self.obstacles.append(obstacle)
        if self.obstacle_tree is None:
            self.obstacle_tree = KDTree(obstacle.bounding_rectangle()[np.newaxis,...], [obstacle])
        else:
            self.obstacle_tree.put(obstacle.bounding_rectangle(), obstacle)

    def set_goal(self, goal: np.array) -> None:
        self.goal = goal

    def set_tol(self, tol: float) -> None:
        self.tol = tol

    def point_free(self, p: np.array) -> bool:
        if self.obstacle_tree is None:
            return True
        p_rect = np.array([p[0], p[0], p[1], p[1]])
        _, potential_hits = self.obstacle_tree.rectangle_search(p_rect)
        if len(potential_hits) == 0:
            return True
        for obs in potential_hits:
            if obs.contains(p):
                return False
        return True

    def path_free(self, a: np.array, b: np.array) -> bool:
        if self.obstacle_tree is None:
            return True
        line_rect = np.array([min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1])])
        _, potential_hits = self.obstacle_tree.rectangle_search(line_rect)

        for shape in potential_hits:
            if shape.line_intersect(a, b):
                return False
        return True

    def boundary_point(self, a: np.array, b: np.array, s_pct: float=1.0) -> np.array:
        assert 0 < s_pct <= 1.0
        line_rect = np.array([min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1])])
        _, potential_hits = self.obstacle_tree.rectangle_search(line_rect)

        min_s = 1
        for shape in potential_hits:
            pt, s = shape.boundary_point(a, b)
            if s < min_s:
                min_s = s
        s = min_s*s_pct
        return s*b + (1-s)*a
        

def make_env() -> Map:
    shapes = []
    for _ in range(10):
        xy = np.random.random(2)
        wh = 0.1 + 0.1*np.random.random(2)
        rect = Rectangle(xy[0], xy[1], wh[0], wh[1])
        shapes.append(rect)
    for _ in range(5):
        xy = np.random.random(2)
        r = 0.05 + 0.05*np.random.random(1)
        circ = Circle(xy[0], xy[1], r[0])
        shapes.append(circ)
    return Map(shapes)

def free_space_example():
    env = make_env()

    fig, ax = plt.subplots()
    env.render(ax)
    ax.set_aspect('equal')
    plt.show()

    free_pts = []
    obs_pts = []
    pts = np.random.random((1000, 2))
    for pt in pts:
        if env.point_free(pt):
            free_pts.append(pt)
        else:
            obs_pts.append(pt)
    free_pts = np.stack(free_pts, axis=0)
    obs_pts = np.stack(obs_pts, axis=0)
    fig, ax = plt.subplots()
    env.render(ax)
    ax.scatter(free_pts[:,0], free_pts[:,1], c="tab:green", alpha=0.25)
    ax.scatter(obs_pts[:,0], obs_pts[:,1], c="tab:red", alpha=0.25)
    ax.set_aspect('equal')
    plt.show()

def projection_example():
    env = make_env()

    fig, ax = plt.subplots()
    env.render(ax)
    ax.set_aspect('equal')
    plt.show()

    lines = []
    for _ in range(1000):
        a = np.random.random(2)
        if not env.point_free(a):
            continue
        ang = 2*np.pi*np.random.random(1)
        b = a + 0.1*np.array([np.cos(ang[0]), np.sin(ang[0])])
        b = env.boundary_point(a, b, s_pct=0.95)
        lines.append(np.stack([a,b], axis=0))
    
    fig, ax = plt.subplots()
    env.render(ax)
    for line in lines:
        ax.plot(line[:,0], line[:,1], c="tab:red", marker='o')
    ax.set_aspect('equal')
    plt.show()

if __name__ == "__main__":
    env = make_env()
    env.set_goal(np.array([0.5,0.5]))
    env.set_tol(0.05)
    
    fig, ax = plt.subplots()
    env.render(ax)
    plt.show()
