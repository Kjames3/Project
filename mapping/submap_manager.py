#!/usr/bin/env python

# Copyright (c) 2024
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Submap Manager for MNE-Lite Distributed SLAM Architecture

Manages a collection of Submaps for a single agent, handling:
- Automatic submap creation based on agent position
- Spatial indexing for efficient submap lookup
- Pose history tracking for loop closure detection
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from .submap import Submap


class SubmapManager:
    """
    Manages a collection of Submaps for a single agent.
    
    Submaps are organized in a spatial grid indexed by chunk coordinates.
    When an agent moves to a new area, a new submap is automatically created.
    """
    
    def __init__(self, submap_size=50.0, grid_size=0.5):
        """
        Constructor method.
        
        :param submap_size: Size of each submap in meters (default 50m)
        :param grid_size: Resolution within each submap (default 0.5m)
        """
        self.submap_size = submap_size
        self.grid_size = grid_size
        
        # Submaps indexed by chunk coordinates
        # Key: (chunk_x, chunk_y) where chunk coords = floor(world_pos / submap_size)
        self.submaps: Dict[Tuple[int, int], Submap] = {}
        
        # Pose history for loop closure detection
        # Each entry: (timestamp, x, y, yaw)
        self.pose_history: List[Tuple[float, float, float, float]] = []
        
        # Maximum pose history length
        self.max_history_length = 5000
        
        # Statistics
        self.total_updates = 0
    
    def _get_chunk_coords(self, wx, wy) -> Tuple[int, int]:
        """
        Convert world coordinates to chunk indices.
        
        :param wx: World X coordinate
        :param wy: World Y coordinate
        :return: (chunk_x, chunk_y) tuple
        """
        chunk_x = int(np.floor(wx / self.submap_size))
        chunk_y = int(np.floor(wy / self.submap_size))
        return (chunk_x, chunk_y)
    
    def _chunk_to_origin(self, chunk_x, chunk_y) -> Tuple[float, float]:
        """
        Convert chunk indices to submap origin coordinates.
        
        :param chunk_x: Chunk X index
        :param chunk_y: Chunk Y index
        :return: (origin_x, origin_y) world coordinates
        """
        origin_x = chunk_x * self.submap_size
        origin_y = chunk_y * self.submap_size
        return (origin_x, origin_y)
    
    def get_or_create_submap(self, wx, wy) -> Submap:
        """
        Get existing submap for a world position, or create a new one.
        
        :param wx: World X coordinate
        :param wy: World Y coordinate
        :return: Submap instance
        """
        chunk = self._get_chunk_coords(wx, wy)
        
        if chunk not in self.submaps:
            origin_x, origin_y = self._chunk_to_origin(chunk[0], chunk[1])
            self.submaps[chunk] = Submap(
                origin_x, origin_y, 
                self.submap_size, self.grid_size
            )
        
        return self.submaps[chunk]
    
    def get_submap_at(self, wx, wy) -> Optional[Submap]:
        """
        Get submap at a world position, or None if doesn't exist.
        
        :param wx: World X coordinate
        :param wy: World Y coordinate
        :return: Submap or None
        """
        chunk = self._get_chunk_coords(wx, wy)
        return self.submaps.get(chunk, None)
    
    def update(self, local_grid, pose):
        """
        Update submaps with a local occupancy grid from the agent.
        
        :param local_grid: 2D numpy array from LocalMapper
        :param pose: (x, y, yaw) tuple in world frame
        """
        if local_grid is None:
            return
        
        x, y, yaw = pose
        
        # Get or create submap for current position
        submap = self.get_or_create_submap(x, y)
        submap.update(local_grid, x, y, yaw)
        
        # Also update neighboring submaps if the local grid overlaps
        # Local grid typically covers ~40m, so check adjacent chunks
        local_radius = (local_grid.shape[0] * self.grid_size) / 2.0
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                # Check point at edge of local grid
                check_x = x + dx * local_radius * 0.8
                check_y = y + dy * local_radius * 0.8
                
                neighbor_chunk = self._get_chunk_coords(check_x, check_y)
                if neighbor_chunk != self._get_chunk_coords(x, y):
                    neighbor = self.get_or_create_submap(check_x, check_y)
                    neighbor.update(local_grid, x, y, yaw)
        
        # Record pose history
        timestamp = time.time()
        self.pose_history.append((timestamp, x, y, yaw))
        
        # Limit history size
        if len(self.pose_history) > self.max_history_length:
            self.pose_history.pop(0)
        
        self.total_updates += 1
    
    def get_nearby_submaps(self, wx, wy, radius=100.0) -> List[Submap]:
        """
        Get all submaps within a radius of a world position.
        
        :param wx: World X coordinate
        :param wy: World Y coordinate
        :param radius: Search radius in meters
        :return: List of Submaps
        """
        result = []
        
        # Calculate chunk range to search
        chunk_radius = int(np.ceil(radius / self.submap_size)) + 1
        center_chunk = self._get_chunk_coords(wx, wy)
        
        for dcx in range(-chunk_radius, chunk_radius + 1):
            for dcy in range(-chunk_radius, chunk_radius + 1):
                chunk = (center_chunk[0] + dcx, center_chunk[1] + dcy)
                if chunk in self.submaps:
                    # Check if submap is actually within radius
                    submap = self.submaps[chunk]
                    center_x = submap.origin_x + self.submap_size / 2
                    center_y = submap.origin_y + self.submap_size / 2
                    
                    dist = np.sqrt((center_x - wx)**2 + (center_y - wy)**2)
                    if dist <= radius + self.submap_size / 2:
                        result.append(submap)
        
        return result
    
    def get_all_submaps(self) -> List[Submap]:
        """
        Get all submaps managed by this manager.
        
        :return: List of all Submaps
        """
        return list(self.submaps.values())
    
    def get_pose_at_time(self, target_time, tolerance=5.0) -> Optional[Tuple[float, float, float]]:
        """
        Get the agent's pose at a specific time (for loop closure).
        
        :param target_time: Unix timestamp
        :param tolerance: Maximum time difference in seconds
        :return: (x, y, yaw) or None
        """
        if not self.pose_history:
            return None
        
        best_match = None
        best_diff = float('inf')
        
        for ts, x, y, yaw in self.pose_history:
            diff = abs(ts - target_time)
            if diff < best_diff and diff <= tolerance:
                best_diff = diff
                best_match = (x, y, yaw)
        
        return best_match
    
    def get_poses_in_region(self, wx, wy, radius, min_age=120.0) -> List[Tuple[float, float, float, float]]:
        """
        Get historical poses within a region, older than min_age seconds.
        Used for loop closure detection.
        
        :param wx: Center X coordinate
        :param wy: Center Y coordinate
        :param radius: Search radius in meters
        :param min_age: Minimum age of poses in seconds
        :return: List of (timestamp, x, y, yaw)
        """
        current_time = time.time()
        result = []
        
        for ts, x, y, yaw in self.pose_history:
            age = current_time - ts
            if age < min_age:
                continue
            
            dist = np.sqrt((x - wx)**2 + (y - wy)**2)
            if dist <= radius:
                result.append((ts, x, y, yaw))
        
        return result
    
    def get_fused_map(self, center_x, center_y, view_size=200.0):
        """
        Generate a fused occupancy grid view around a center point.
        Used for visualization and path planning.
        
        :param center_x: Center X in world coordinates
        :param center_y: Center Y in world coordinates
        :param view_size: Size of the view in meters
        :return: 2D numpy array of probabilities, and (min_x, min_y) origin
        """
        view_dim = int(view_size / self.grid_size)
        fused = np.full((view_dim, view_dim), 0.5, dtype=np.float32)
        
        min_x = center_x - view_size / 2
        max_x = center_x + view_size / 2
        min_y = center_y - view_size / 2
        
        # Get relevant submaps
        submaps = self.get_nearby_submaps(center_x, center_y, view_size)
        
        for submap in submaps:
            probs = submap.get_probability_map()
            
            for r in range(submap.dim):
                for c in range(submap.dim):
                    # Skip unknown cells
                    if 0.45 <= probs[r, c] <= 0.55:
                        continue
                    
                    # Get world position
                    wx, wy = submap.grid_to_world(r, c)
                    
                    # Check bounds
                    if not (min_x <= wx < max_x and min_y <= wy < center_y + view_size / 2):
                        continue
                    
                    # Convert to fused grid indices
                    fr = int((max_x - wx) / self.grid_size)
                    fc = int((wy - min_y) / self.grid_size)
                    
                    if 0 <= fr < view_dim and 0 <= fc < view_dim:
                        fused[fr, fc] = probs[r, c]
        
        return fused, (min_x, min_y)
    
    def calculate_total_coverage(self):
        """
        Calculate total mapped area across all submaps.
        
        :return: Total coverage in m²
        """
        total = 0.0
        for submap in self.submaps.values():
            total += submap.calculate_coverage()
        return total
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the submap manager.
        
        :return: Dictionary of statistics
        """
        return {
            'num_submaps': len(self.submaps),
            'total_updates': self.total_updates,
            'pose_history_length': len(self.pose_history),
            'total_coverage_m2': self.calculate_total_coverage()
        }
    
    def __repr__(self):
        return f"SubmapManager(submaps={len(self.submaps)}, updates={self.total_updates})"
