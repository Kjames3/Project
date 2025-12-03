#!/usr/bin/env python

# Copyright (c) 2024
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Fusion Server for Multi-Agent Cooperative Perception
"""

import numpy as np
import cv2
from numba import jit

@jit(nopython=True)
def fast_fusion_update(log_odds_map, local_indices_r, local_indices_c, 
                      gx, gy, gyaw, grid_size, max_x, min_y, grid_dim, 
                      local_center, update_val):
    
    cos_a = np.cos(gyaw)
    sin_a = np.sin(gyaw)
    
    for i in range(len(local_indices_r)):
        r = local_indices_r[i]
        c = local_indices_c[i]
        
        # Local -> Vehicle
        x_veh = (local_center - r) * grid_size
        y_veh = (c - local_center) * grid_size
        
        # Vehicle -> Global
        X_global = x_veh * cos_a - y_veh * sin_a + gx
        Y_global = x_veh * sin_a + y_veh * cos_a + gy
        
        # Global -> Grid Index
        global_r = int((max_x - X_global) / grid_size)
        global_c = int((Y_global - min_y) / grid_size)
        
        # Boundary Check
        if 0 <= global_r < grid_dim and 0 <= global_c < grid_dim:
            log_odds_map[global_r, global_c] += update_val

@jit(nopython=True)
def _render_map_jit(log_odds_map, output_image):
    # output_image is (W, H, 3) for Pygame
    out_w, out_h, _ = output_image.shape
    map_h, map_w = log_odds_map.shape # Grid is (Rows, Cols)
    
    scale_x = map_w / out_w
    scale_y = map_h / out_h
    
    for x in range(out_w):
        for y in range(out_h):
            # Map x (col) -> mc
            # Map y (row) -> mr
            mc = int(x * scale_x)
            mr = int(y * scale_y)
            
            if mr >= 0 and mr < map_h and mc >= 0 and mc < map_w:
                l_val = log_odds_map[mr, mc]
                
                # Sigmoid: p = 1 / (1 + exp(-l))
                # Optimization: check sign of l_val first to avoid exp?
                # Or just do it.
                prob = 1.0 / (1.0 + np.exp(-l_val))
                
                # Color
                if prob < 0.45:
                    # Free -> Black
                    output_image[x, y, 0] = 0
                    output_image[x, y, 1] = 0
                    output_image[x, y, 2] = 0
                elif prob > 0.55:
                    # Occupied -> White
                    output_image[x, y, 0] = 255
                    output_image[x, y, 1] = 255
                    output_image[x, y, 2] = 255
                else:
                    # Unknown -> Gray
                    output_image[x, y, 0] = 50
                    output_image[x, y, 1] = 50
                    output_image[x, y, 2] = 50

class FusionServer(object):
    """
    FusionServer acts as a central authority for merging local maps from multiple agents.
    """

    def __init__(self, grid_size=1, map_dim=600):
        """
        Constructor method
        :param grid_size: Resolution in meters per pixel
        :param map_dim: Dimension of the global map in meters (square)
        """
        self.grid_size = grid_size
        self.map_dim = map_dim
        self.grid_dim = int(self.map_dim / self.grid_size)
        
        self.min_x = -self.map_dim / 2.0
        self.max_x = self.map_dim / 2.0
        self.min_y = -self.map_dim / 2.0
        self.max_y = self.map_dim / 2.0
        
        # Log-Odds Map
        # 0.0 = Unknown (p=0.5, log(1) = 0)
        self.log_odds_map = np.zeros((self.grid_dim, self.grid_dim), dtype=np.float32)
        
        # Constants for Log-Odds
        self.l_occ = np.log(0.7 / 0.3)  # p(occ) = 0.7
        self.l_free = np.log(0.4 / 0.6) # p(free) = 0.4
        self.l_max = 5.0 # Clamp
        self.l_min = -5.0 # Clamp
        
        # Trajectories: agent_id -> list of (x, y)
        self.trajectories = {}
        
        # Local Maps: agent_id -> local_occupancy_grid
        self.maps = {}

    def update_map(self, agent_id, local_occupancy_grid, pose):
        """
        Updates the global map with a local occupancy grid from an agent.
        """
        if local_occupancy_grid is None:
            return

        # Store local map
        self.maps[agent_id] = local_occupancy_grid

        rows, cols = local_occupancy_grid.shape
        local_center = rows // 2
        
        # Find indices
        # Free: < 0.45
        free_indices = np.where(local_occupancy_grid < 0.45)
        # Occupied: > 0.55
        occ_indices = np.where(local_occupancy_grid > 0.55)
        
        # Process Free
        if len(free_indices[0]) > 0:
            fast_fusion_update(self.log_odds_map, free_indices[0], free_indices[1], 
                               pose[0], pose[1], np.radians(pose[2]), self.grid_size, 
                               self.max_x, self.min_y, self.grid_dim, 
                               local_center, self.l_free)
                           
        # Process Occupied
        if len(occ_indices[0]) > 0:
            fast_fusion_update(self.log_odds_map, occ_indices[0], occ_indices[1], 
                               pose[0], pose[1], np.radians(pose[2]), self.grid_size, 
                               self.max_x, self.min_y, self.grid_dim, 
                               local_center, self.l_occ)

    def update_trajectory(self, agent_id, pose):
        """
        Updates the trajectory history for an agent.
        :param agent_id: ID of the agent
        :param pose: (x, y, yaw)
        """
        if agent_id not in self.trajectories:
            self.trajectories[agent_id] = []
        
        # Only add point if it's far enough from the last one (e.g., 1 meter)
        if self.trajectories[agent_id]:
            last_pose = self.trajectories[agent_id][-1]
            dist = np.sqrt((pose[0]-last_pose[0])**2 + (pose[1]-last_pose[1])**2)
            if dist < 1.0:
                return

        # Add current position
        self.trajectories[agent_id].append((pose[0], pose[1]))
        
        # Limit history if needed (e.g. 2000 points)
        if len(self.trajectories[agent_id]) > 2000:
            self.trajectories[agent_id].pop(0)

    def get_global_map(self):
        """
        Returns the current fused global map as probabilities.
        :return: 2D numpy array representing the global occupancy grid (0.0-1.0)
        """
        return 1.0 / (1.0 + np.exp(-self.log_odds_map))
        
    def get_map_image(self, width, height):
        """
        Returns a rendered RGB image of the map, scaled to width x height.
        :return: Numpy array (width, height, 3) ready for pygame.surfarray.make_surface
        """
        output_image = np.zeros((width, height, 3), dtype=np.uint8)
        _render_map_jit(self.log_odds_map, output_image)
        return output_image

    def calculate_coverage(self):
        """Calculates mapped area in square meters"""
        # Count cells that are NOT unknown (approx 0.5)
        # We assume anything < 0.45 or > 0.55 has been 'seen'
        probs = self.get_global_map()
        mapped_cells = np.count_nonzero((probs < 0.45) | (probs > 0.55))
        
        # Area = cells * (resolution^2)
        area_m2 = mapped_cells * (self.grid_size * self.grid_size)
        return area_m2

    def save_map_to_disk(self, filename="final_map.png"):
        """Saves the current map as an image"""
        probs = self.get_global_map()
        
        # Convert to 0-255 image
        image = np.full(probs.shape, 127, dtype=np.uint8)
        image[probs < 0.45] = 255 # Free is White
        image[probs > 0.55] = 0   # Occupied is Black
        
        cv2.imwrite(filename, image)
        print(f"Map saved to {filename}")
