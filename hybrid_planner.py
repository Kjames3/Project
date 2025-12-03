#!/usr/bin/env python

# Copyright (c) 2024
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Hybrid Route Planner
Combines global static route planning with dynamic occupancy grid from FusionServer.
"""

import numpy as np
import math
from numba import njit

@njit(fastmath=True)
def _check_line_numba(global_map, sr, sc, er, ec, grid_dim, threshold):
    """
    Optimized Bresenham-like line check. 
    Returns True if ANY cell on the line is occupied (> threshold).
    """
    # Calculate distance in grid cells
    dx = er - sr
    dy = ec - sc
    distance = math.sqrt(dx*dx + dy*dy)
    steps = int(distance) + 1
    
    for i in range(steps + 1):
        t = i / float(steps) if steps > 0 else 0.0
        
        # Interpolate indices
        r = int(sr + t * dx)
        c = int(sc + t * dy)
        
        # Boundary check
        if 0 <= r < grid_dim and 0 <= c < grid_dim:
            # Check occupancy
            if global_map[r, c] > threshold:
                return True
                
    return False

class HybridRoutePlanner(object):
    """
    HybridRoutePlanner uses the FusionServer's global map to check for obstacles
    along the planned route.
    """

    def __init__(self, fusion_server):
        """
        :param fusion_server: Instance of FusionServer
        """
        self._fusion_server = fusion_server

    def is_path_blocked(self, start_location, end_location, threshold=0.6):
        """
        Checks if the straight line between start and end is blocked by obstacles
        in the global map.

        :param start_location: carla.Location or tuple (x, y, z)
        :param end_location: carla.Location or tuple (x, y, z)
        :param threshold: Probability threshold for considering a cell occupied
        :return: True if blocked, False otherwise
        """
        global_map = self._fusion_server.get_global_map()
        if global_map is None:
            return False

        # Get map parameters
        grid_size = self._fusion_server.grid_size
        min_x = self._fusion_server.min_x
        min_y = self._fusion_server.min_y
        max_x = self._fusion_server.max_x
        max_y = self._fusion_server.max_y
        grid_dim = self._fusion_server.grid_dim

        # Extract x, y
        sx = start_location.x if hasattr(start_location, 'x') else start_location[0]
        sy = start_location.y if hasattr(start_location, 'y') else start_location[1]
        ex = end_location.x if hasattr(end_location, 'x') else end_location[0]
        ey = end_location.y if hasattr(end_location, 'y') else end_location[1]

        # Check bounds
        if not (min_x <= sx <= max_x and min_y <= sy <= max_y and
                min_x <= ex <= max_x and min_y <= ey <= max_y):
            # Out of map bounds - assume clear or handle otherwise
            return False

        # Convert to Grid Indices
        # Row = (Max_X - X) / res
        # Col = (Y - Min_Y) / res
        
        sr = int((max_x - sx) / grid_size)
        sc = int((sy - min_y) / grid_size)
        er = int((max_x - ex) / grid_size)
        ec = int((ey - min_y) / grid_size)

        # Bresenham's Line Algorithm (Optimized with Numba)
        if _check_line_numba(global_map, sr, sc, er, ec, grid_dim, threshold):
            return True

        # --- Dynamic obstruction check: other agents from FusionServer ---
        # If FusionServer is tracking agent trajectories, treat other agents as obstacles
        # along this segment.
        if hasattr(self._fusion_server, "trajectories") and self._fusion_server.trajectories:
            
            # Helper: distance from a point to the start-end segment in world coords
            def point_to_segment_dist(px, py):
                vx = ex - sx
                vy = ey - sy
                wx = px - sx
                wy = py - sy
                seg_len_sq = vx * vx + vy * vy
                if seg_len_sq == 0.0:
                    return math.hypot(px - sx, py - sy)
                u = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len_sq))
                proj_x = sx + u * vx
                proj_y = sy + u * vy
                return math.hypot(px - proj_x, py - proj_y)

            # Tune these radii as needed
            own_ignore_radius = 2.0   # ignore positions extremely close to our start (likely ourselves)
            agent_radius = 3.0        # treat other agents as discs of this radius

            for agent_id, traj in self._fusion_server.trajectories.items():
                if not traj:
                    continue

                ox, oy = traj[-1]  # latest reported position of that agent

                # Ignore positions essentially on top of our current pose (likely our own ID)
                if math.hypot(ox - sx, oy - sy) < own_ignore_radius:
                    continue

                # If another agent is close to our planned straight-line segment, treat as blocked
                if point_to_segment_dist(ox, oy) < agent_radius:
                    return True

        return False

    def get_frontiers(self):
        """
        Scans the global map to find frontiers (edges between Free and Unknown space).
        
        :return: List of (x, y) tuples in global coordinates representing frontier points.
        """
        global_map = self._fusion_server.get_global_map()
        if global_map is None:
            return []

        # 1. Define Masks
        # Free: < 0.4
        # Unknown: 0.4 <= p <= 0.6
        # Occupied: > 0.6
        
        free_mask = (global_map < 0.4).astype(np.uint8)
        unknown_mask = ((global_map >= 0.4) & (global_map <= 0.6)).astype(np.uint8)
        
        # 2. Find Frontiers
        # Frontier pixels are Free pixels that are adjacent to Unknown pixels.
        # We can dilate Unknown mask and intersect with Free mask.
        
        import cv2
        kernel = np.ones((3, 3), np.uint8)
        dilated_unknown = cv2.dilate(unknown_mask, kernel, iterations=1)
        
        frontier_mask = (free_mask == 1) & (dilated_unknown == 1)
        
        # 3. Extract Coordinates
        # Indices (r, c)
        frontier_indices = np.argwhere(frontier_mask)
        
        if len(frontier_indices) == 0:
            return []
            
        # Downsample if too many?
        # Or cluster them. For now, return all (or a subset).
        # Let's return a subset (stride) to avoid returning thousands of points.
        # stride = 5
        # frontier_indices = frontier_indices[::stride]
        
        # 4. Convert to Global Coordinates
        # Row = (Max_X - X) / res => X = Max_X - Row * res
        # Col = (Y - Min_Y) / res => Y = Min_Y + Col * res
        
        grid_size = self._fusion_server.grid_size
        max_x = self._fusion_server.max_x
        min_y = self._fusion_server.min_y
        
        frontiers = []
        for r, c in frontier_indices:
            x = max_x - r * grid_size
            y = min_y + c * grid_size
            frontiers.append((x, y))
            
        return frontiers
