#!/usr/bin/env python

# Copyright (c) 2024
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Local Mapping Module for Occupancy Grid Generation
"""

import numpy as np
import cv2
from numba import njit

@njit(fastmath=True)
def fast_raycast(grid, center_r, center_c, end_r, end_c):
    """Bresenham's Line Algorithm optimized with Numba"""
    # Loop through all unique endpoints
    for i in range(len(end_r)):
        r0, c0 = center_r, center_c
        r1, c1 = end_r[i], end_c[i]
        
        dx = abs(r1 - r0)
        dy = abs(c1 - c0)
        sx = 1 if r0 < r1 else -1
        sy = 1 if c0 < c1 else -1
        err = dx - dy
        
        while True:
            # Set pixel to 1 (Free)
            # Note: The original code used 255 for free in a uint8 mask.
            # The user's snippet sets grid[r0, c0] = 1.
            # We should ensure 'grid' is passed as the mask (uint8) or the map itself?
            # The original code used a separate 'free_mask' (uint8) and drew lines with 255.
            # Then set self.local_map[free_mask > 0] = 0.0.
            # If we pass 'free_mask' to this function, we should set it to 1 (or 255).
            # Let's stick to the user's snippet 'grid[r0, c0] = 1' and assume we pass a zero-initialized grid.
            # We can then use this grid to update self.local_map.
            grid[r0, c0] = 1 
            
            if r0 == r1 and c0 == c1:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                r0 += sx
            if e2 < dx:
                err += dx
                c0 += sy

class LocalMapper(object):
    """
    LocalMapper processes Semantic LiDAR data to generate a 2D local occupancy grid.
    """

    def __init__(self, vehicle, grid_size=0.5, window_size=40.0):
        """
        :param vehicle: The vehicle actor
        :param grid_size: Resolution of the grid in meters per pixel (default: 0.5m)
        :param window_size: Size of the local window in meters (default: 40m x 40m)
        """
        self._vehicle = vehicle
        self.grid_size = grid_size
        self.window_size = window_size
        
        # Grid dimensions
        self.grid_dim = int(self.window_size / self.grid_size)
        self.center_idx = self.grid_dim // 2
        
        # Occupancy Grid: 0.0 = Free, 1.0 = Occupied, 0.5 = Unknown
        self.local_map = np.full((self.grid_dim, self.grid_dim), 0.5, dtype=np.float32)

    def process_lidar(self, lidar_data, vehicle_transform):
        """
        Updates the local map based on new Semantic LiDAR data.

        :param lidar_data: carla.SemanticLidarMeasurement
        :param vehicle_transform: Current vehicle transform (carla.Transform)
        """
        self.local_map.fill(0.5)

        if lidar_data is None:
            return

        # Parse Semantic LiDAR Data
        # Format: x, y, z, cos_inc_angle, object_idx, object_tag
        # We need to handle the buffer correctly.
        # numpy dtype for structured array
        data = np.frombuffer(lidar_data.raw_data, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32), 
            ('cos', np.float32), ('idx', np.uint32), ('tag', np.uint32)
        ]))

        points = np.column_stack((data['x'], data['y'], data['z']))
        tags = data['tag']

        # Coordinate Conversion: Sensor Frame -> Grid Frame
        # Sensor Frame: x=forward, y=right, z=up (CARLA standard)
        # Grid Frame: (row, col)
        # We map:
        # Row = Center - X / res
        # Col = Center + Y / res
        
        px_r = self.center_idx - (points[:, 0] / self.grid_size).astype(np.int32)
        px_c = self.center_idx + (points[:, 1] / self.grid_size).astype(np.int32)
        
        # Filter points inside grid
        valid_mask = (px_r >= 0) & (px_r < self.grid_dim) & \
                     (px_c >= 0) & (px_c < self.grid_dim)
        
        valid_r = px_r[valid_mask]
        valid_c = px_c[valid_mask]
        valid_r = px_r[valid_mask]
        valid_c = px_c[valid_mask]
        valid_tags = tags[valid_mask]
        
        # DEBUG
        # if len(points) > 0:
        #     print(f"LocalMapper: {len(points)} points. Valid: {len(valid_r)}")
        
        # 1. Ray Casting for Free Space
        # We raycast to ALL valid points (Ground, Obstacles, Dynamic) to clear the path.
        # This assumes that if the laser hit something, the space in between is empty.
        
        free_mask = np.zeros((self.grid_dim, self.grid_dim), dtype=np.uint8)
        
        # Unique endpoints to optimize raycasting
        # Use (row, col) for Numba function
        unique_endpoints = np.unique(np.column_stack((valid_r, valid_c)), axis=0)
        
        if len(unique_endpoints) > 0:
            end_r = unique_endpoints[:, 0]
            end_c = unique_endpoints[:, 1]
            fast_raycast(free_mask, self.center_idx, self.center_idx, end_r, end_c)
            
        # 2. Mark Occupied Cells
        # Filter based on Semantic Tags
        # Keep: Building(3), Fence(5), Pole(6), TrafficLight(7), TrafficSign(8), Wall(4)
        # Discard (don't mark as occupied): Road(1), Sidewalk(2), Ground, Vegetation(9)?, 
        # Vehicles(10, 14+), Pedestrians(4/12?)
        # Note: CARLA tag for Pedestrian is 12. 4 is Wall?
        # Let's check standard tags:
        # 1=Road, 2=Sidewalk, 3=Building, 4=Wall, 5=Fence, 6=Pole, 7=TrafficLight, 8=TrafficSign, 9=Vegetation, 10=Terrain
        # 11=Sky, 12=Pedestrian, 13=Rider, 14=Car, 15=Truck...
        
        # User said: "Retain Building, Fence, Pole, TrafficSign"
        # I will include Wall(4) and TrafficLight(7) as they are static obstacles too.
        # I will exclude Vegetation(9) as it might be drivable or soft? User didn't specify. I'll exclude it to be safe (or include if it's a tree).
        # Let's stick to the user's explicit list + obvious static obstacles.
        
        static_obstacle_tags = [3, 4, 5, 6, 7, 8] 
        
        is_static_obstacle = np.isin(valid_tags, static_obstacle_tags)
        
        obs_r = valid_r[is_static_obstacle]
        obs_c = valid_c[is_static_obstacle]
        
        # 3. Combine into Local Map
        # Default 0.5
        # Set Free (0.0)
        self.local_map[free_mask > 0] = 0.0
        
        # Set Occupied (1.0)
        # Note: Occupied overwrites Free (at the endpoint)
        self.local_map[obs_r, obs_c] = 1.0

    def get_local_map(self):
        return self.local_map
