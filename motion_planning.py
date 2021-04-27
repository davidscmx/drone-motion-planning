import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np
import csv
import time
import networkx as nx
import pickle
import os

from planning_utils import *

#/home/david/miniconda3/envs/fcnd/lib/python3.6/site-packages/udacidrone
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local,local_to_global


def bres(p1, p2):
    """
    Note this solution requires `x1` < `x2` and `y1` < `y2`.
    """
    x1, y1 = p1
    x2, y2 = p2
    cells = []

    # Here's a quick explanation in math terms of our approach
    # First, set dx = x2 - x1 and dy = y2 - y1
    dx, dy = x2 - x1, y2 - y1
    # Then define a new quantity: d = x dy - y dx.
    # and set d = 0 initially
    d = 0
    # The condition we care about is whether
    # (x + 1) * m < y + 1 or moving things around a bit:
    # (x + 1) dy / dx < y + 1
    # which implies: x dy - y dx < dx - dy
    # or in other words: d < dx - dy is our new condition

    # Initialize i, j indices
    i = x1
    j = y1

    while i < x2 and j < y2:
        cells.append([i, j])
        if d < dx - dy:
            d += dy
            i += 1
        elif d == dx - dy:
            # uncomment these two lines for conservative approach
            cells.append([i+1, j])
            cells.append([i, j+1])

            d += dy
            i += 1
            d -= dx
            j += 1
        else:
            d -= dx
            j += 1

    return np.array(cells)

def prune_with_bresenham(waypoints, grid):

    waypoints = [(int(wp[0]), int(wp[1])) for wp in waypoints]
    start = waypoints[0]
    reference = start
    new_waypoints = []
    new_waypoints.append(start)

    for i in range(1, len(waypoints)-1):
        p1 = reference
        p2 = waypoints[i]
        print(p1,p2)
        cells = bres(p1, p2)

        for c in cells:
            # First check if we're off the map
            if np.amin(c) < 0 or c[0] >= grid.shape[0] or c[1] >= grid.shape[1]:
                break
            # Next check if we're in collision
            if grid[c[0], c[1]] == 1:
                reference = waypoints[i]
                new_waypoints.append(waypoints[i-1])
                break

    new_waypoints.append(waypoints[-1])
    return new_waypoints


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()

class MotionPlanner(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)


    def point(self, p):
        return np.array([p[0], p[1], 1.]).reshape(1, -1)

    def collinearity_check(self, p1, p2, p3, epsilon=1):
        m = np.concatenate((p1, p2, p3), 0)
        det = np.linalg.det(m)
        return abs(det) < epsilon

    def prune_path(self, path):
        """
        If the 3 points are in a line remove the 2nd point.
        The 3rd point now becomes and 2nd pointand the check is
        redone with a new third pointon the next iteration.
        """

        pruned_path = [p for p in path]

        i = 0
        while i < len(pruned_path) - 2:
            p1 = self.point(pruned_path[i])
            p2 = self.point(pruned_path[i+1])
            p3 = self.point(pruned_path[i+2])

            if self.collinearity_check(p1, p2, p3):
                # Something subtle here but we can mutate
                # `pruned_path` freely because the length
                # of the list is check on every iteration.
                pruned_path.remove(pruned_path[i+1])
            else:
                i += 1
        return pruned_path

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5

        self.target_position[2] = TARGET_ALTITUDE

        # 1. read lat0, lon0 from colliders into floating point values
        with open('colliders.csv') as f:
            initial_point_str = f.read().splitlines()[0].replace(",","").split()
            lat0, lon0 = float(initial_point_str[1]), float(initial_point_str[3])

        self.set_home_position(lon0, lat0, 0)
        # convert to current local position using global_to_local()
        self._north, self._east, self._down   = global_to_local(self.global_position, self.global_home).ravel()

        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))

        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

        # Define a grid for a particular altitude and safety margin around obstacles

        grid, north_offset, east_offset, points = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)

        # map coordinates as given in the csv file
        start_ne = (int(self.local_position[0]), int(self.local_position[1]), 5)
        goal_north, goal_east, goal_down = global_to_local((-122.393278, 37.794305, -5), self.global_home).ravel()
        goal_ne = (int(goal_north ), int(goal_east ), int(goal_down))

        use_voronoi = False
        use_grid = True
        use_polygon_graph = False

        if use_grid:
            # convert to grid "coordinate" system i.e. origin is at minimum north, east
            start_ne_int = (int(start_ne[0] - north_offset), int(start_ne[1] - east_offset))
            goal_ne_int = (int(goal_ne[0] - north_offset), int(goal_ne[1] - east_offset))

            path, cost = a_star(grid, heuristic, start_ne_int, goal_ne_int)
            waypoints_before_conversion = [p for p in path]

        elif use_voronoi:
            if not os.path.exists("voronoi_edges.pkl"):
                edges = create_voronoi_edges(grid, points)
                with open('voronoi_edges.pkl', 'wb') as f:
                    pickle.dump(edges, f)
            else:
                print("Loading cached edges")
                with open('voronoi_edges.pkl', 'rb') as f:
                    edges = pickle.load(f)

            G = nx.Graph()
            # Stepping through each edge
            for e in edges:
                p1, p2 = e[0], e[1]
                dist = np.linalg.norm(np.array(p2) - np.array(p1))
                G.add_edge(p1, p2, weight=dist)

            start_ne = (start_ne[0] - north_offset, start_ne[1] - east_offset)
            goal_ne = (goal_ne[0] - north_offset, goal_ne[1] - east_offset)

            start_ne_closest = closest_point(G, start_ne)
            goal_ne_closest = closest_point(G, goal_ne)

            path, cost = a_star_graph(G, heuristic, start_ne_closest, goal_ne_closest)

            # Convert path to waypoints
            waypoints_before_conversion = [p for p in path]

        elif use_polygon_graph:
            print("Running polygon approach")
            if not os.path.exists("./graph.pkl"):
                sampler = Sampler(data)
                polygons = sampler._polygons
                nodes = sampler.sample(400)

                start_ne = (start_ne[0] - north_offset, start_ne[1] - east_offset)
                goal_ne = (goal_ne[0] - north_offset, goal_ne[1] - east_offset)

                nodes.append(start_ne)
                nodes.append(goal_ne)
                t0 = time.time()
                G = create_probabilistic_graph(nodes, 10, polygons)
                print(f'graph took {time.time()-t0} seconds to build')

                with open('graph.pkl', 'wb') as f:
                    pickle.dump(G, f)
            else:
                print("Loading cached graph")
                with open('graph.pkl', 'rb') as f:
                    G = pickle.load(f)

            path, cost = a_star_graph(G, heuristic, start_ne, goal_ne)
            waypoints_before_conversion = [p for p in path]

        waypoints_before_conversion = self.prune_path(waypoints_before_conversion)

        if use_polygon_graph:
            waypoints = [[int(p[0]), int(p[1]), int(p[2]), 0] for p in waypoints_before_conversion]
        else:
            waypoints = [[int(p[0] + north_offset), int(p[1] + east_offset), TARGET_ALTITUDE, 0] for p in waypoints_before_conversion]

        print("waypoints, len(waypoints) ", waypoints, len(waypoints))
        self.waypoints = waypoints
        # Send waypoints to simulator for visualization
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanner(conn)
    time.sleep(1)

    drone.start()
