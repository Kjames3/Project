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

        # Bresenham's Line Algorithm (or simple sampling)
        # We can just sample points along the line
        dist = math.sqrt((sx - ex)**2 + (sy - ey)**2)
        steps = int(dist / grid_size) + 1
        
        for i in range(steps + 1):
            t = i / float(steps) if steps > 0 else 0
            r = int(sr + t * (er - sr))
            c = int(sc + t * (ec - sc))
            
            if 0 <= r < grid_dim and 0 <= c < grid_dim:
                prob = global_map[r, c]
                if prob > threshold:
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
