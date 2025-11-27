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

class LocalMapper(object):
    """
    LocalMapper processes LiDAR data to generate a 2D local occupancy grid.
    """

    def __init__(self, vehicle, grid_size=0.5, window_size=40.0):
        """
        :param vehicle: The vehicle actor (used for transforms if needed, though we mostly use relative data)
        :param grid_size: Resolution of the grid in meters per pixel (default: 0.5m)
        :param window_size: Size of the local window in meters (default: 40m x 40m)
        """
        self._vehicle = vehicle
        self.grid_size = grid_size
        self.window_size = window_size
        
        # Grid dimensions
        self.grid_dim = int(self.window_size / self.grid_size)
        self.center_idx = self.grid_dim // 2
        
        # Occupancy Grid: 0 = Free, 1 = Occupied, 0.5 = Unknown
        # We initialize with 0.5 (Unknown)
        self.local_map = np.full((self.grid_dim, self.grid_dim), 0.5, dtype=np.float32)

    def process_lidar(self, lidar_data, vehicle_transform):
        """
        Updates the local map based on new LiDAR data.

        :param lidar_data: LiDAR points in sensor frame (N, 4) or (N, 3)
        :param vehicle_transform: Current vehicle transform (carla.Transform)
        """
        # Reset map to Unknown for the new step (or decay old values - for now, simple reset/update)
        # In a real SLAM system, we might keep history, but for "Local Map" attached to vehicle, 
        # we often rebuild or scroll. For this phase, let's rebuild from scratch or just update.
        # The prompt says "Convert LiDAR data into a 2D local occupancy grid", implying a snapshot or accumulated local view.
        # Let's reset to 0.5 to represent "current local view".
        self.local_map.fill(0.5)

        if lidar_data is None or len(lidar_data) == 0:
            return

        # 1. Filter points within the window
        # LiDAR data is usually in sensor frame. 
        # We assume sensor is roughly at vehicle origin for 2D mapping, or we apply offset.
        # For simplicity, we assume sensor x,y aligns enough with vehicle x,y or we ignore z-offset for 2D grid.
        # Points are (x, y, z).
        
        # Filter by height (z) to remove ground plane and too high obstacles
        # Sensor is at z=~1.6m. Ground is at z=~-1.6m relative to sensor.
        # We want obstacles between e.g. -1.0 and +1.0 relative to sensor?
        # Or just use all points that are not ground.
        # Let's assume simple obstacle detection: points within a z-range.
        # CARLA LiDAR usually returns points.
        
        points = lidar_data[:, :3] # x, y, z
        
        # Filter out ground (approx z < -1.4 relative to sensor mounted at 1.6m)
        # Adjust threshold as needed.
        obstacle_mask = points[:, 2] > -1.4 
        obstacle_points = points[obstacle_mask]
        
        # 2. Convert to Grid Coordinates
        # x (forward) -> col (increasing)
        # y (right) -> row (increasing) ?? 
        # Standard image coords: (0,0) top-left.
        # Let's define:
        # Vehicle at (center_idx, center_idx)
        # +x (forward) -> Up (-row) or Right (+col)?
        # Let's use standard map convention: x=forward, y=left
        # Image: row=y, col=x? No.
        # Let's map:
        # Grid X (col) = Vehicle Y (left) ... wait
        # Let's stick to:
        # Grid Row (i) = -Vehicle X (forward) + Center (so forward is Up)
        # Grid Col (j) = -Vehicle Y (left) + Center (so left is Left) -> No, usually Y is left in CARLA.
        # Let's align with Pygame visualization later.
        # Pygame: (0,0) top-left.
        # We want Forward to be Up (negative y in pixel coords).
        # So Pixel Y = Center - Vehicle X / resolution
        # Pixel X = Center + Vehicle Y / resolution (if Y is right)
        # CARLA: X=Forward, Y=Right, Z=Up.
        
        # Pixel Row = CenterRow - (Points_X / grid_size)
        # Pixel Col = CenterCol + (Points_Y / grid_size)
        
        px_r = self.center_idx - (points[:, 0] / self.grid_size).astype(np.int32)
        px_c = self.center_idx + (points[:, 1] / self.grid_size).astype(np.int32)
        
        # Filter points inside grid
        valid_mask = (px_r >= 0) & (px_r < self.grid_dim) & \
                     (px_c >= 0) & (px_c < self.grid_dim)
        
        valid_points = points[valid_mask]
        valid_r = px_r[valid_mask]
        valid_c = px_c[valid_mask]
        
        # 3. Ray Casting (Free Space)
        # We want to mark cells between sensor (center) and hit point as Free (0).
        # We can use cv2.line to draw lines of 0s.
        # But we have thousands of points. Drawing thousands of lines might be slow in Python.
        # Optimization: Draw on a temporary "free" mask.
        
        # Create a "free space" image initialized to 0
        free_grid = np.zeros_like(self.local_map, dtype=np.uint8)
        
        # We only need to raycast for a subset of points to cover the area?
        # Or just draw all.
        # Let's try drawing all lines.
        # We need to iterate? cv2.line takes one line.
        # cv2.polylines? No.
        # We can iterate. It might be slow.
        # Alternative: Raytrace only to obstacle points.
        
        # For "Unknown", we initialized with 0.5.
        # "Free" = 0.0
        # "Occupied" = 1.0
        
        # Let's use a simple approach first:
        # 1. Initialize grid with 0.5
        # 2. Draw lines from center to all valid points with value 0.0 (Free)
        # 3. Set valid obstacle points to 1.0 (Occupied)
        
        # Optimization: Downsample points for raycasting if too many?
        # Let's try full resolution first.
        
        center_point = (self.center_idx, self.center_idx)
        
        # We can use a separate mask for "seen" areas.
        # But we need to distinguish Free vs Occupied.
        
        # Let's update the map directly.
        # Since we want "Occupied" to overwrite "Free" if there's a conflict in the same cell?
        # Actually, Occupied > Free.
        # But Raycast clears the path.
        # So:
        # 1. Create Free Grid (0 = Unknown, 1 = Free)
        # 2. Create Occupied Grid (0 = Empty, 1 = Occupied)
        # 3. Combine.
        
        # Step A: Raycast for Free Space
        # We use 255 for Free in a uint8 image for speed, then convert.
        free_mask = np.zeros((self.grid_dim, self.grid_dim), dtype=np.uint8)
        
        # Unique ray endpoints to reduce redundancy
        # Combine r, c into unique pairs
        # This significantly reduces draw calls
        unique_endpoints = np.unique(np.column_stack((valid_c, valid_r)), axis=0)
        
        for uc, ur in unique_endpoints:
            cv2.line(free_mask, center_point, (uc, ur), 255, 1)
            
        # Step B: Mark Occupied Cells
        # Only for points that were actually obstacles (filtered by height)
        # We need to re-filter valid_r/c for obstacle_mask
        
        # Re-calculate indices for obstacle points only
        obs_px_r = self.center_idx - (obstacle_points[:, 0] / self.grid_size).astype(np.int32)
        obs_px_c = self.center_idx + (obstacle_points[:, 1] / self.grid_size).astype(np.int32)
        
        obs_valid_mask = (obs_px_r >= 0) & (obs_px_r < self.grid_dim) & \
                         (obs_px_c >= 0) & (obs_px_c < self.grid_dim)
        
        obs_r = obs_px_r[obs_valid_mask]
        obs_c = obs_px_c[obs_valid_mask]
        
        # Step C: Combine
        # Default is 0.5
        self.local_map[:] = 0.5
        
        # Set Free (0.0) where free_mask is set
        self.local_map[free_mask > 0] = 0.0
        
        # Set Occupied (1.0) where obstacles are
        self.local_map[obs_r, obs_c] = 1.0
        
    def get_local_map(self):
        return self.local_map
