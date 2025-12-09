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
import math

@njit(fastmath=True)
def voxel_filter(points, voxel_size):
    """
    Returns discretized integer voxel indices.
    """
    # 1. Discretize coords to integer voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    return voxel_indices

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

def get_matrix(transform):
    """
    Creates a 4x4 matrix from a CARLA-like Transform object.
    Supports objects with .location (x,y,z) and .rotation (pitch,yaw,roll).
    """
    if hasattr(transform, 'get_matrix'):
        # If it has get_matrix (CARLA object), use it but convert to numpy
        return np.array(transform.get_matrix())
    
    # Manual construction
    x = transform.location.x
    y = transform.location.y
    z = transform.location.z
    
    yaw = math.radians(transform.rotation.yaw)
    pitch = math.radians(transform.rotation.pitch)
    roll = math.radians(transform.rotation.roll)
    
    # Rotation Matrix (Yaw * Pitch * Roll) - Standard CARLA (Unreal) convention
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cr = math.cos(roll)
    sr = math.sin(roll)
    
    matrix = np.identity(4)
    
    # Rotation
    matrix[0, 0] = cp * cy
    matrix[0, 1] = cy * sp * sr - sy * cr
    matrix[0, 2] = -cy * sp * cr - sy * sr
    
    matrix[1, 0] = cp * sy
    matrix[1, 1] = sy * sp * sr + cy * cr
    matrix[1, 2] = -sy * sp * cr + cy * sr
    
    matrix[2, 0] = sp
    matrix[2, 1] = -cp * sr
    matrix[2, 2] = cp * cr
    
    # Translation
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    
    return matrix

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
        
        # --- NEW: SENSOR TO VEHICLE TRANSFORM ---
        # Points are in Sensor Frame. Transform to Vehicle Frame.
        if hasattr(lidar_data, 'transform') and vehicle_transform is not None:
            try:
                # 1. Get Matrices
                sensor_matrix = get_matrix(lidar_data.transform)
                vehicle_matrix = get_matrix(vehicle_transform)
                
                # 2. Compute Sensor -> Vehicle Transform
                # T_sensor_local = inv(T_vehicle_global) * T_sensor_global
                # Assuming vehicle_matrix is standard 4x4, we can invert it.
                vehicle_inv = np.linalg.inv(vehicle_matrix)
                rel_transform = np.dot(vehicle_inv, sensor_matrix)
                
                # 3. Apply Transform to Points
                # Points: (N, 3). Add homogeneous 1.
                num_points = points.shape[0]
                points_hom = np.hstack((points, np.ones((num_points, 1))))
                
                # Transformed = (M_rel @ P_hom.T).T
                # Optimization: P_transformed = P @ M_rel[:3,:3].T + M_rel[:3,3]
                R_rel = rel_transform[:3, :3]
                T_rel = rel_transform[:3, 3]
                
                points = np.dot(points, R_rel.T) + T_rel
                
            except Exception as e:
                print(f"LocalMapper Error: Failed to transform points: {e}")
        # ----------------------------------------

        # --- OPTIMIZED: DOWNSAMPLING STEP ---
        # Quantize points to 10cm voxels
        voxel_size = 0.1
        quantized = voxel_filter(points, voxel_size)
        
        # Optimization: Use 1D hashing for fast unique filtering
        # Assuming points are within +/- 200m
        # int32 range is +/- 2 billion.
        # Indices around +/- 2000.
        # Packed index: x + y*MAX_WIDTH + z*MAX_WIDTH*MAX_HEIGHT
        
        # Shift to positive range to avoid negative modulo/hash issues (though numpy handles negatives)
        # Offset 5000 is enough for 500m range
        offset = 5000 
        
        # Limit the coordinates to avoid overflow/index errors
        # Note: 'quantized' is int32.
        # We cast to int64 for packing to be safe.
        
        q_x = quantized[:, 0].astype(np.int64) + offset
        q_y = quantized[:, 1].astype(np.int64) + offset
        q_z = quantized[:, 2].astype(np.int64) + offset
        
        # Multipliers
        # X range ~10000. Y range ~10000.
        mul_y = 10000
        mul_z = 100000000
        
        packed = q_x + q_y * mul_y + q_z * mul_z
        
        _, unique_indices = np.unique(packed, return_index=True)
        
        # Filter data
        points = points[unique_indices]
        tags = tags[unique_indices]
        # ------------------------------

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
        # OPTIMIZATION: Also use 1D hashing for this unique check
        # But indices are small here (0..80).
        # Standard unique is fine, but we can use:
        # packed_endpoints = valid_r * grid_dim + valid_c
        # _, unique_ep_idx = np.unique(packed_endpoints, return_index=True)
        
        packed_endpoints = valid_r.astype(np.int32) * self.grid_dim + valid_c.astype(np.int32)
        _, unique_ep_idx = np.unique(packed_endpoints, return_index=True)
        
        if len(unique_ep_idx) > 0:
            end_r = valid_r[unique_ep_idx]
            end_c = valid_c[unique_ep_idx]
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
