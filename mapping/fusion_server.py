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

class FusionServer(object):
    """
    FusionServer acts as a central authority for merging local maps from multiple agents.
    """

    def __init__(self, grid_size=0.5, map_dim=500):
        """
        Constructor method
        :param grid_size: Resolution in meters per pixel
        :param map_dim: Dimension of the global map in meters (square)
        """
        self.grid_size = grid_size
        self.map_dim = map_dim
        self.grid_dim = int(self.map_dim / self.grid_size)
        
        # Global Map Origin (World Coordinates)
        # Centered at (0,0) -> Top-Left of Grid is (-map_dim/2, -map_dim/2) ??
        # No, usually (0,0) is center.
        # Let's define Top-Left World Coordinate:
        # X_min = -map_dim / 2
        # Y_min = -map_dim / 2
        # But CARLA Y is inverted relative to standard image?
        # Let's stick to:
        # Grid Row 0 = Max X (Forward)
        # Grid Col 0 = Min Y (Left)
        # This aligns with standard "North Up" map if X is North.
        
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

    def update_map(self, agent_id, local_occupancy_grid, pose):
        """
        Updates the global map with a local occupancy grid from an agent.

        :param agent_id: ID of the agent sending the update
        :param local_occupancy_grid: 2D numpy array (0.0=Free, 1.0=Occ, 0.5=Unknown)
        :param pose: Tuple (x, y, yaw) representing the agent's global pose (yaw in degrees)
        """
        if local_occupancy_grid is None:
            return

        # 1. Get Local Grid Indices (Occupied and Free)
        # Local Grid is (N, N). Center is vehicle.
        # We need to iterate over all cells? Or just non-unknown ones.
        # Vectorized approach is better.
        
        local_dim = local_occupancy_grid.shape[0]
        local_center = local_dim // 2
        
        # Find indices of Free and Occupied cells
        # Free: < 0.4 (approx 0.0)
        # Occupied: > 0.6 (approx 1.0)
        # Unknown: ~0.5
        
        free_indices = np.where(local_occupancy_grid < 0.4)
        occ_indices = np.where(local_occupancy_grid > 0.6)
        
        # Combine to process
        # We need to transform these indices to Global Frame
        
        # Local Indices (r, c) -> Vehicle Frame (x, y)
        # In mapping.py:
        # px_r = center - x/res => x = (center - px_r) * res
        # px_c = center + y/res => y = (px_c - center) * res
        
        def transform_indices(indices, is_occupied):
            r, c = indices
            
            # 1. Local Grid -> Vehicle Frame
            x_veh = (local_center - r) * self.grid_size
            y_veh = (c - local_center) * self.grid_size
            
            # 2. Vehicle Frame -> Global Frame
            # Pose: (x, y, yaw_deg)
            gx, gy, gyaw_deg = pose
            gyaw = np.radians(gyaw_deg)
            
            cos_a = np.cos(gyaw)
            sin_a = np.sin(gyaw)
            
            # Rotation + Translation
            # X_global = x_veh * cos - y_veh * sin + gx
            # Y_global = x_veh * sin + y_veh * cos + gy
            
            X_global = x_veh * cos_a - y_veh * sin_a + gx
            Y_global = x_veh * sin_a + y_veh * cos_a + gy
            
            # DEBUG: Check ranges
            # if len(X_global) > 0:
            #      print(f"Global Range X: [{np.min(X_global):.2f}, {np.max(X_global):.2f}], Y: [{np.min(Y_global):.2f}, {np.max(Y_global):.2f}]")
            #      print(f"Map Bounds X: [{self.min_x}, {self.max_x}], Y: [{self.min_y}, {self.max_y}]")
            #      print(f"Agent Pose: {pose}")
            
            # 3. Global Frame -> Global Grid Indices
            # Row = (Max_X - X_global) / res
            # Col = (Y_global - Min_Y) / res
            
            global_r = ((self.max_x - X_global) / self.grid_size).astype(np.int32)
            global_c = ((Y_global - self.min_y) / self.grid_size).astype(np.int32)
            
            # Filter valid indices
            valid_mask = (global_r >= 0) & (global_r < self.grid_dim) & \
                         (global_c >= 0) & (global_c < self.grid_dim)
            
            valid_r = global_r[valid_mask]
            valid_c = global_c[valid_mask]
            
            # Update Log-Odds
            update_val = self.l_occ if is_occupied else self.l_free
            
            # We use 'at' for unbuffered add if there are duplicates?
            # Or just simple add.
            # np.add.at allows handling duplicate indices correctly if multiple local cells map to same global cell
            np.add.at(self.log_odds_map, (valid_r, valid_c), update_val)
            
            # DEBUG
            if len(valid_r) > 0:
                print(f"Agent {agent_id}: Updated {len(valid_r)} cells. Val: {update_val}")

        # Process Free
        transform_indices(free_indices, is_occupied=False)
        
        # Process Occupied
        transform_indices(occ_indices, is_occupied=True)
        
        # Clamp
        np.clip(self.log_odds_map, self.l_min, self.l_max, out=self.log_odds_map)

    def update_trajectory(self, agent_id, pose):
        """
        Updates the trajectory history for an agent.
        :param agent_id: ID of the agent
        :param pose: (x, y, yaw)
        """
        if agent_id not in self.trajectories:
            self.trajectories[agent_id] = []
        
        # Add current position
        self.trajectories[agent_id].append((pose[0], pose[1]))
        
        # Limit history if needed (e.g. 10000 points)
        if len(self.trajectories[agent_id]) > 10000:
            self.trajectories[agent_id].pop(0)

    def get_global_map(self):
        """
        Returns the current fused global map as probabilities.

        :return: 2D numpy array representing the global occupancy grid (0.0-1.0)
        """
        # Sigmoid: p = 1 / (1 + exp(-l))
        # Sigmoid: p = 1 / (1 + exp(-l))
        return 1.0 / (1.0 + np.exp(-self.log_odds_map))

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
        # Free (Low Prob) -> 255 (White)
        # Occupied (High Prob) -> 0 (Black)
        # Unknown -> 127 (Gray)
        
        image = np.full(probs.shape, 127, dtype=np.uint8)
        image[probs < 0.45] = 255 # Free is White
        image[probs > 0.55] = 0   # Occupied is Black
        
        # Flip to match visualization if needed
        # image = cv2.flip(image, 0)
        
        cv2.imwrite(filename, image)
        print(f"Map saved to {filename}")
