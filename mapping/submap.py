#!/usr/bin/env python

# Copyright (c) 2024
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Submap Module for MNE-Lite Distributed SLAM Architecture

A Submap represents a 50x50m occupancy grid chunk that can be independently
updated and merged with other submaps during loop closure events.
"""

import numpy as np
import time

# Try to import numba for JIT acceleration, fall back to pure Python if unavailable
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Dummy decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


def _update_log_odds_fast(log_odds, local_grid, offset_r, offset_c, 
                          dim, l_occ, l_free, l_min, l_max):
    """
    Fast log-odds update. Uses Numba JIT if available, otherwise pure Python/NumPy.
    Maps local grid values to submap coordinates.
    """
    local_h, local_w = local_grid.shape
    
    for lr in range(local_h):
        for lc in range(local_w):
            # Target position in submap
            sr = offset_r + lr
            sc = offset_c + lc
            
            # Boundary check
            if 0 <= sr < dim and 0 <= sc < dim:
                val = local_grid[lr, lc]
                
                if val < 0.45:  # Free
                    log_odds[sr, sc] += l_free
                elif val > 0.55:  # Occupied
                    log_odds[sr, sc] += l_occ
                
                # Clamp
                if log_odds[sr, sc] > l_max:
                    log_odds[sr, sc] = l_max
                elif log_odds[sr, sc] < l_min:
                    log_odds[sr, sc] = l_min


# Apply JIT decorator if numba is available
if NUMBA_AVAILABLE:
    _update_log_odds_fast = jit(nopython=True)(_update_log_odds_fast)


class Submap:
    """
    A 50x50m occupancy grid chunk for distributed SLAM.
    
    Each submap maintains:
    - Log-odds occupancy grid
    - Origin in world coordinates
    - Timestamp of last update
    - Point cloud cache for ICP alignment
    """
    
    def __init__(self, origin_x, origin_y, submap_size=50.0, grid_size=0.5):
        """
        Constructor method.
        
        :param origin_x: X coordinate of submap origin (bottom-left corner) in world frame
        :param origin_y: Y coordinate of submap origin (bottom-left corner) in world frame
        :param submap_size: Size of submap in meters (default 50m)
        :param grid_size: Resolution in meters per cell (default 0.5m)
        """
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.submap_size = submap_size
        self.grid_size = grid_size
        
        # Grid dimensions
        self.dim = int(submap_size / grid_size)  # 50m / 0.5m = 100 cells
        
        # Log-odds map (0.0 = unknown, negative = free, positive = occupied)
        self.log_odds = np.zeros((self.dim, self.dim), dtype=np.float32)
        
        # Metadata
        self.last_updated = time.time()
        self.update_count = 0
        
        # Log-odds constants
        self.l_occ = np.log(0.7 / 0.3)   # p(occ) = 0.7
        self.l_free = np.log(0.4 / 0.6)  # p(free) = 0.4
        self.l_max = 5.0
        self.l_min = -5.0
        
        # Cached probability map
        self._prob_cache = None
        self._cache_dirty = True
    
    def world_to_grid(self, wx, wy):
        """
        Convert world coordinates to grid indices.
        
        :param wx: World X coordinate
        :param wy: World Y coordinate
        :return: (row, col) in grid coordinates, or None if out of bounds
        """
        # Relative position from origin
        rel_x = wx - self.origin_x
        rel_y = wy - self.origin_y
        
        # Grid indices (row = Y direction, col = X direction in standard convention)
        # Using same convention as FusionServer: row corresponds to X, col to Y
        row = int((self.submap_size - rel_x) / self.grid_size)
        col = int(rel_y / self.grid_size)
        
        if 0 <= row < self.dim and 0 <= col < self.dim:
            return (row, col)
        return None
    
    def grid_to_world(self, row, col):
        """
        Convert grid indices to world coordinates.
        
        :param row: Grid row index
        :param col: Grid column index
        :return: (wx, wy) world coordinates
        """
        rel_x = self.submap_size - row * self.grid_size
        rel_y = col * self.grid_size
        
        wx = self.origin_x + rel_x
        wy = self.origin_y + rel_y
        
        return (wx, wy)
    
    def update(self, local_grid, agent_x, agent_y, agent_yaw):
        """
        Update submap with a local occupancy grid from an agent.
        
        :param local_grid: 2D numpy array from LocalMapper (probability 0-1)
        :param agent_x: Agent X position in world frame
        :param agent_y: Agent Y position in world frame
        :param agent_yaw: Agent yaw angle in degrees
        """
        if local_grid is None:
            return
        
        local_h, local_w = local_grid.shape
        local_center = local_h // 2
        
        # Agent position in submap grid
        agent_grid = self.world_to_grid(agent_x, agent_y)
        if agent_grid is None:
            return
        
        agent_r, agent_c = agent_grid
        
        # Offset to align local grid center with agent position
        offset_r = agent_r - local_center
        offset_c = agent_c - local_center
        
        # Update using fast Numba function
        _update_log_odds_fast(
            self.log_odds, local_grid, 
            offset_r, offset_c, self.dim,
            self.l_occ, self.l_free, self.l_min, self.l_max
        )
        
        self.last_updated = time.time()
        self.update_count += 1
        self._cache_dirty = True
    
    def get_probability_map(self):
        """
        Get occupancy probabilities (0.0 to 1.0).
        
        :return: 2D numpy array of probabilities
        """
        if self._cache_dirty or self._prob_cache is None:
            self._prob_cache = 1.0 / (1.0 + np.exp(-self.log_odds))
            self._cache_dirty = False
        return self._prob_cache
    
    def get_occupied_points(self, threshold=0.55):
        """
        Extract occupied cells as a 2D point cloud for ICP alignment.
        
        :param threshold: Probability threshold for occupied cells
        :return: Nx2 numpy array of (x, y) world coordinates
        """
        probs = self.get_probability_map()
        occupied = np.argwhere(probs > threshold)
        
        if len(occupied) == 0:
            return np.array([]).reshape(0, 2)
        
        # Convert to world coordinates
        points = []
        for r, c in occupied:
            wx, wy = self.grid_to_world(r, c)
            points.append([wx, wy])
        
        return np.array(points, dtype=np.float32)
    
    def get_free_points(self, threshold=0.45):
        """
        Extract free cells as a 2D point cloud.
        
        :param threshold: Probability threshold for free cells
        :return: Nx2 numpy array of (x, y) world coordinates
        """
        probs = self.get_probability_map()
        free = np.argwhere(probs < threshold)
        
        if len(free) == 0:
            return np.array([]).reshape(0, 2)
        
        points = []
        for r, c in free:
            wx, wy = self.grid_to_world(r, c)
            points.append([wx, wy])
        
        return np.array(points, dtype=np.float32)
    
    def merge(self, other_submap, transform=None):
        """
        Merge another submap into this one.
        
        :param other_submap: Submap instance to merge
        :param transform: Optional 3x3 transformation matrix (for ICP-aligned merge)
        """
        other_probs = other_submap.get_probability_map()
        
        for r in range(other_submap.dim):
            for c in range(other_submap.dim):
                # Get world coordinate from other submap
                wx, wy = other_submap.grid_to_world(r, c)
                
                # Apply transform if provided
                if transform is not None:
                    pt = np.array([wx, wy, 1.0])
                    transformed = transform @ pt
                    wx, wy = transformed[0], transformed[1]
                
                # Map to this submap's grid
                target = self.world_to_grid(wx, wy)
                if target is None:
                    continue
                
                tr, tc = target
                
                # Merge log-odds (average)
                other_log = other_submap.log_odds[r, c]
                if abs(other_log) > 0.01:  # Only merge non-unknown cells
                    self.log_odds[tr, tc] = (self.log_odds[tr, tc] + other_log) / 2.0
        
        self._cache_dirty = True
    
    def contains_point(self, wx, wy):
        """
        Check if a world point falls within this submap's boundaries.
        
        :param wx: World X coordinate
        :param wy: World Y coordinate
        :return: True if point is within submap
        """
        return (self.origin_x <= wx < self.origin_x + self.submap_size and
                self.origin_y <= wy < self.origin_y + self.submap_size)
    
    def get_bounds(self):
        """
        Get world coordinate bounds of this submap.
        
        :return: (min_x, max_x, min_y, max_y)
        """
        return (
            self.origin_x,
            self.origin_x + self.submap_size,
            self.origin_y,
            self.origin_y + self.submap_size
        )
    
    def calculate_coverage(self):
        """
        Calculate mapped area in square meters.
        
        :return: Area in m² that has been observed
        """
        probs = self.get_probability_map()
        mapped_cells = np.count_nonzero((probs < 0.45) | (probs > 0.55))
        return mapped_cells * (self.grid_size ** 2)
    
    def __repr__(self):
        return f"Submap(origin=({self.origin_x:.1f}, {self.origin_y:.1f}), updates={self.update_count})"
