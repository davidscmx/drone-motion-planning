from enum import Enum
from queue import PriorityQueue
import numpy as np
import pickle

import matplotlib.pyplot as plt
from skimage.morphology import medial_axis
from skimage.util import invert
from scipy.spatial import Voronoi, voronoi_plot_2d
from bresenham import bresenham
import networkx as nx
from shapely.geometry import Polygon, Point, LineString
import time
from sampling import Sampler
from sklearn.neighbors import KDTree
from pykdtree.kdtree import KDTree as KDTreePy

def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """
    print("Creating grid")
    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))
    print("north_min, north_max", north_min, north_max)
    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))
    print("east_min, east_max", east_min, east_max)
    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))
    print("north_size, east_size", north_size, east_size)
    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Initialize an empty list for Voronoi points
    points = []

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1
            points.append([north - north_min, east - east_min])

    print("Finished creating grid")
    return grid, int(north_min), int(east_min), points


def point(p):
        return np.array([p[0], p[1], 1.]).reshape(1, -1)

def create_voronoi_edges(grid, points):
    """
    Returns a grid representation of a 2D configuration space
    along with Voronoi graph edges given obstacle data and the
    drone's altitude.
    """
    # create a voronoi graph based on location of obstacle centres
    graph = Voronoi(points)

    # Check each edge from graph.ridge_vertices for collision
    edges = []
    for v in graph.ridge_vertices:
        p1 = graph.vertices[v[0]]
        p2 = graph.vertices[v[1]]
        cells = list(bresenham(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
        hit = False

        for c in cells:
            # First check if we're off the map
            if np.amin(c) < 0 or c[0] >= grid.shape[0] or c[1] >= grid.shape[1]:
                hit = True
                break
            # Next check if we're in collision
            if grid[c[0], c[1]] == 1:
                hit = True
                break

        # If the edge does not hit on obstacle
        # add it to the list
        if not hit:
            # array to tuple for future graph creation step)
            p1 = (p1[0], p1[1])
            p2 = (p2[0], p2[1])

            edges.append((p1, p2))
    print("finished voronoi")
    return edges

# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)

    NORTHWEST = (-1, -1, np.sqrt(2))
    NORTHEAST = (-1, 1, np.sqrt(2))
    SOUTHEAST = (1, 1, np.sqrt(2))
    SOUTHWEST = (1, -1, np.sqrt(2))

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle
    if x >= grid.shape[0] - 1 or y >= grid.shape[1] - 1:
        return []

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
        valid_actions.remove(Action.NORTHWEST)
        valid_actions.remove(Action.NORTHEAST)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
        valid_actions.remove(Action.SOUTHWEST)
        valid_actions.remove(Action.SOUTHEAST)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
        try:
            valid_actions.remove(Action.NORTHWEST)
            valid_actions.remove(Action.SOUTHWEST)
        except ValueError:
            pass
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)
        try:
            valid_actions.remove(Action.NORTHEAST)
            valid_actions.remove(Action.SOUTHEAST)
        except ValueError:
            pass
    return valid_actions


def a_star(grid, h, start, goal):

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))
    # end of while loop
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
    return path[::-1], path_cost

def a_star_graph(graph, h, start, goal):
    """Modified A* to work with NetworkX graphs."""

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for next_node in graph[current_node]:
                cost = graph.edges[current_node, next_node]['weight']
                branch_cost = current_cost + cost
                queue_cost = branch_cost + h(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    queue.put((queue_cost, next_node))
                    branch[next_node] = (branch_cost, current_node)

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
    return path[::-1], path_cost

def heuristic(n1, n2):
    return np.linalg.norm(np.array(n2) - np.array(n1))

def closest_point(graph, current_point):
    """
    Compute the closest point in the `graph`
    to the `current_point`.
    """
    closest_point = None
    dist = 100000
    for p in graph.nodes:
        d = np.linalg.norm(np.array(p) - np.array(current_point))
        if d < dist:
            closest_point = p
            dist = d
    return closest_point

def extract_polygons(data):

    polygons = []
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]

        # Extract the 4 corners of each obstacle
        #
        # The order of the points needs to be counterclockwise
        # in order to work with the simple angle test
        # Also, `shapely` draws sequentially from point to point.
        obstacle = [north - d_north, north + d_north, east - d_east, east + d_east]
        corners = [(obstacle[0], obstacle[2]), (obstacle[0], obstacle[3]),
                  (obstacle[1], obstacle[3]), (obstacle[1], obstacle[2])]

        # Compute the height of the polygon
        height = alt + d_alt

        p = Polygon(corners)
        polygons.append((p, height))

    return polygons

def collides(polygons, point):
    for (p, height) in polygons:
        if p.contains(Point(point)) and height >= point[2]:
            return True
    return False

def can_connect(n1, n2, polygons):
    l = LineString([n1, n2])

    for p in polygons:
        if p.crosses(l) and p.height >= min(n1[2], n2[2]):
            return False
    return True

def create_probabilistic_graph(nodes, k, polygons):
    g = nx.Graph()
    tree = KDTree(nodes)
    for n1 in nodes:
        # for each node connect try to connect to k nearest nodes
        dist, idxs = tree.query([n1], k)
        idxs = idxs[0]
        for idx in idxs:

            n2 = nodes[idx]
            if n2 == n1:
                continue

            if can_connect(n1, n2, polygons):
                dist = np.linalg.norm(np.array(n1) - np.array(n2))
                g.add_edge(n1, n2, weight=dist)
    return g

if __name__ == "__main__":

    data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)
    # Grid coordinates with origin at xmin and ymin of cvs
    start_ne = (316, 445, 5)
    goal_ne = (750, 370, 5)

    grid, north_offset, east_offset, points = create_grid(data, 5, 5)

    fig = plt.figure()
    plt.xlabel('EAST')
    plt.ylabel('NORTH')

    plt.imshow(grid, cmap='Greys', origin='lower')

    # North is y axis, East is x axis
    plt.plot(start_ne[1], start_ne[0], 'gx')
    plt.plot(goal_ne[1], goal_ne[0], 'gx')

    sampler = Sampler(data)
    polygons = sampler._polygons

    nodes = sampler.sample(300)

    # Convert to "map" coordinates
    start_ne_with_offset = (int(start_ne[0]+north_offset), int(start_ne[1]+east_offset), int(start_ne[2]))
    goal_ne_with_offset  = (int(goal_ne[0]+north_offset), int(goal_ne[1]+east_offset), int(goal_ne[2]) )

    print('Local Start and Goal: ', start_ne, goal_ne)
    print('Local Start and Goal no offset: ', start_ne_with_offset, goal_ne_with_offset)

    nodes.append(start_ne_with_offset)
    nodes.append(goal_ne_with_offset)

    t0 = time.time()
    g = create_probabilistic_graph(nodes, 10, polygons)
    print(f'graph took {time.time()-t0} seconds to build')

    with open('graph.pkl', 'wb') as f:
        pickle.dump(g, f)
    for i in list(g.nodes):
        print(i)
    # draw nodes
    for n1 in g.nodes:
        plt.scatter(n1[1] - east_offset, n1[0] - north_offset, c='red')

    # draw edges
    for (n1, n2) in g.edges:
        plt.plot([n1[1] - east_offset, n2[1] - east_offset], [n1[0] - north_offset, n2[0] - north_offset], 'black')

    path, cost = a_star_graph(g, heuristic, start_ne_with_offset, goal_ne_with_offset)

    ## Visualize the path
    path_pairs = zip(path[:-1], path[1:])
    for (n1, n2) in path_pairs:
        print(n1,n2)
        plt.plot([n1[1] - east_offset, n2[1] - east_offset], [n1[0] - north_offset, n2[0] - north_offset], 'green')

    plt.show()