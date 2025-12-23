#!/usr/bin/env python

# Copyright (c) 2024
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
MNE-Lite Fusion Server for Multi-Agent Cooperative Perception

This is an enhanced version of FusionServer that uses the MNE-Lite distributed
architecture with submaps, loop closure detection, and pose graph optimization.
"""

import numpy as np
import cv2
import time
from typing import Dict, List, Optional, Tuple

# Try to import numba for JIT acceleration, fall back to pure Python if unavailable
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from .submap import Submap
from .submap_manager import SubmapManager
from .loop_closure import LoopClosureDetector, LoopClosure
from .pose_graph import PoseGraph


def fast_fusion_update(log_odds_map, local_indices_r, local_indices_c, 
                      gx, gy, gyaw, grid_size, max_x, min_y, grid_dim, 
                      local_center, update_val, min_val, max_val):
    """Legacy fast fusion update for backward compatibility."""
    cos_a = np.cos(gyaw)
    sin_a = np.sin(gyaw)
    
    for i in range(len(local_indices_r)):
        r = local_indices_r[i]
        c = local_indices_c[i]
        
        x_veh = (local_center - r) * grid_size
        y_veh = (c - local_center) * grid_size
        
        X_global = x_veh * cos_a - y_veh * sin_a + gx
        Y_global = x_veh * sin_a + y_veh * cos_a + gy
        
        global_r = int((max_x - X_global) / grid_size)
        global_c = int((Y_global - min_y) / grid_size)
        
        if 0 <= global_r < grid_dim and 0 <= global_c < grid_dim:
            new_val = log_odds_map[global_r, global_c] + update_val
            if new_val > max_val:
                new_val = max_val
            elif new_val < min_val:
                new_val = min_val
            log_odds_map[global_r, global_c] = new_val


# Apply JIT decorator if numba is available
if NUMBA_AVAILABLE:
    fast_fusion_update = jit(nopython=True)(fast_fusion_update)


class FusionServerMNELite:
    """
    MNE-Lite Fusion Server using distributed submap architecture.
    
    Key features:
    - Each agent maintains independent submaps via SubmapManager
    - Loop closure detection triggers ICP alignment
    - Pose graph optimization ensures global consistency
    - On-demand map fusion for visualization (not every frame)
    """
    
    def __init__(self, grid_size=0.5, submap_size=50.0, 
                 loop_check_interval=50, enable_loop_closure=True):
        """
        Constructor method.
        
        :param grid_size: Resolution in meters per cell
        :param submap_size: Size of each submap in meters
        :param loop_check_interval: Check for loop closures every N updates
        :param enable_loop_closure: Enable/disable loop closure detection
        """
        self.grid_size = grid_size
        self.submap_size = submap_size
        
        # Per-agent submap managers
        self.submap_managers: Dict[int, SubmapManager] = {}
        
        # Loop closure detection
        self.loop_detector = LoopClosureDetector(
            proximity_threshold=15.0,
            time_threshold=120.0,
            min_overlap_points=30,
            icp_fitness_threshold=0.25
        )
        self.enable_loop_closure = enable_loop_closure
        self.loop_check_interval = loop_check_interval
        
        # Pose graph optimization
        self.pose_graph = PoseGraph()
        self._next_node_id = 0
        self._agent_to_nodes: Dict[int, List[int]] = {}  # agent_id -> list of node_ids
        
        # Detected loop closures
        self.loop_closures: List[LoopClosure] = []
        
        # Update counter
        self._update_count = 0
        
        # Trajectories for visualization (compatible with existing code)
        self.trajectories: Dict[int, List[Tuple[float, float]]] = {}
        
        # Local maps cache for legacy compatibility
        self.maps: Dict[int, np.ndarray] = {}
        
        # Cached global map for visualization
        self._cached_global_map = None
        self._global_map_dirty = True
        self._last_global_map_time = 0
        self._global_map_cache_duration = 0.5  # seconds
        
        # Map dimensions for legacy compatibility
        self.map_dim = 600  # Default, adjusted dynamically
        self.grid_dim = int(self.map_dim / self.grid_size)
        self.min_x = -self.map_dim / 2.0
        self.max_x = self.map_dim / 2.0
        self.min_y = -self.map_dim / 2.0
        self.max_y = self.map_dim / 2.0
    
    def update_map(self, agent_id: int, local_occupancy_grid: np.ndarray, 
                   pose: Tuple[float, float, float]):
        """
        Update map with local occupancy grid from an agent.
        
        :param agent_id: Agent identifier
        :param local_occupancy_grid: 2D numpy array from LocalMapper
        :param pose: (x, y, yaw) tuple in world frame
        """
        if local_occupancy_grid is None:
            return
        
        # Store reference for legacy compatibility
        self.maps[agent_id] = local_occupancy_grid
        
        x, y, yaw = pose
        
        # Get or create submap manager for this agent
        if agent_id not in self.submap_managers:
            self.submap_managers[agent_id] = SubmapManager(
                self.submap_size, self.grid_size
            )
            self._agent_to_nodes[agent_id] = []
        
        manager = self.submap_managers[agent_id]
        
        # Update submaps
        manager.update(local_occupancy_grid, pose)
        
        # Update trajectory for visualization
        self.update_trajectory(agent_id, pose)
        
        # Add pose graph node periodically (every 10 updates per agent)
        if manager.total_updates % 10 == 0:
            node_id = self._next_node_id
            self._next_node_id += 1
            
            # First node is fixed as anchor
            fixed = len(self._agent_to_nodes[agent_id]) == 0
            
            self.pose_graph.add_node(
                node_id, x, y, np.radians(yaw), fixed=fixed
            )
            
            # Add odometry edge to previous node
            if self._agent_to_nodes[agent_id]:
                prev_node_id = self._agent_to_nodes[agent_id][-1]
                prev_pose = self.pose_graph.get_pose(prev_node_id)
                if prev_pose:
                    dx = x - prev_pose[0]
                    dy = y - prev_pose[1]
                    dth = np.radians(yaw) - prev_pose[2]
                    self.pose_graph.add_odometry_edge(
                        prev_node_id, node_id, dx, dy, dth
                    )
            
            self._agent_to_nodes[agent_id].append(node_id)
        
        self._update_count += 1
        self._global_map_dirty = True
        
        # Check for loop closures periodically
        if self.enable_loop_closure and self._update_count % self.loop_check_interval == 0:
            self._process_loop_closures()
    
    def _process_loop_closures(self):
        """Check for and process loop closures."""
        if len(self.submap_managers) < 1:
            return
        
        closures = self.loop_detector.check_loop_closures(self.submap_managers)
        
        for closure in closures:
            self.loop_closures.append(closure)
            print(f"[MNE-Lite] Loop closure detected: Agent {closure.agent_a} <-> "
                  f"Agent {closure.agent_b} at ({closure.position[0]:.1f}, "
                  f"{closure.position[1]:.1f}), fitness={closure.fitness_score:.2f}")
            
            # Add loop closure edge to pose graph
            # Find nearest nodes for both agents
            nodes_a = self._agent_to_nodes.get(closure.agent_a, [])
            nodes_b = self._agent_to_nodes.get(closure.agent_b, [])
            
            if nodes_a and nodes_b:
                # Use latest nodes as approximation
                node_a = nodes_a[-1]
                node_b = nodes_b[-1]
                
                # Extract relative pose from ICP transform
                dx = closure.transform[0, 2]
                dy = closure.transform[1, 2]
                dth = np.arctan2(closure.transform[1, 0], closure.transform[0, 0])
                
                self.pose_graph.add_loop_closure_edge(
                    node_a, node_b, dx, dy, dth
                )
        
        # Run pose graph optimization if we have loop closures
        if closures:
            converged = self.pose_graph.optimize(max_iterations=50)
            if converged:
                print(f"[MNE-Lite] Pose graph optimized, residual={self.pose_graph.last_residual:.6f}")
    
    def update_trajectory(self, agent_id: int, pose: Tuple[float, float, float]):
        """
        Update trajectory history for an agent.
        
        :param agent_id: Agent identifier
        :param pose: (x, y, yaw) tuple
        """
        if agent_id not in self.trajectories:
            self.trajectories[agent_id] = []
        
        x, y, _ = pose
        
        # Only add if moved enough
        if self.trajectories[agent_id]:
            last_x, last_y = self.trajectories[agent_id][-1]
            dist = np.sqrt((x - last_x)**2 + (y - last_y)**2)
            if dist < 1.0:
                return
        
        self.trajectories[agent_id].append((x, y))
        
        # Limit history
        if len(self.trajectories[agent_id]) > 2000:
            self.trajectories[agent_id].pop(0)
    
    def get_global_map(self) -> np.ndarray:
        """
        Get fused global occupancy map (on-demand fusion).
        
        :return: 2D numpy array of probabilities (0-1)
        """
        current_time = time.time()
        
        # Use cache if available and fresh
        if (self._cached_global_map is not None and 
            not self._global_map_dirty and
            current_time - self._last_global_map_time < self._global_map_cache_duration):
            return self._cached_global_map
        
        # Compute center from all agent positions
        all_x = []
        all_y = []
        for traj in self.trajectories.values():
            if traj:
                all_x.append(traj[-1][0])
                all_y.append(traj[-1][1])
        
        if not all_x:
            # Return empty map
            return np.full((self.grid_dim, self.grid_dim), 0.5, dtype=np.float32)
        
        center_x = np.mean(all_x)
        center_y = np.mean(all_y)
        
        # Fuse submaps from all agents
        view_size = self.map_dim
        fused = np.full((self.grid_dim, self.grid_dim), 0.5, dtype=np.float32)
        
        for agent_id, manager in self.submap_managers.items():
            local_fused, _ = manager.get_fused_map(center_x, center_y, view_size)
            
            # Merge into global map (average non-unknown cells)
            mask = (local_fused < 0.45) | (local_fused > 0.55)
            
            # Simple averaging merge
            for r in range(min(local_fused.shape[0], fused.shape[0])):
                for c in range(min(local_fused.shape[1], fused.shape[1])):
                    if mask[r, c]:
                        if 0.45 <= fused[r, c] <= 0.55:
                            # Current is unknown, use new value
                            fused[r, c] = local_fused[r, c]
                        else:
                            # Average existing and new
                            fused[r, c] = (fused[r, c] + local_fused[r, c]) / 2.0
        
        self._cached_global_map = fused
        self._global_map_dirty = False
        self._last_global_map_time = current_time
        
        return fused
    
    def get_submaps_in_range(self, start_loc, end_loc) -> List[Submap]:
        """
        Get all submaps that intersect with the line from start to end.
        Used by HybridRoutePlanner for path checking.
        
        :param start_loc: Start location (x, y) or carla.Location
        :param end_loc: End location (x, y) or carla.Location
        :return: List of relevant Submaps
        """
        sx = start_loc.x if hasattr(start_loc, 'x') else start_loc[0]
        sy = start_loc.y if hasattr(start_loc, 'y') else start_loc[1]
        ex = end_loc.x if hasattr(end_loc, 'x') else end_loc[0]
        ey = end_loc.y if hasattr(end_loc, 'y') else end_loc[1]
        
        # Center point and radius
        cx = (sx + ex) / 2
        cy = (sy + ey) / 2
        radius = np.sqrt((ex - sx)**2 + (ey - sy)**2) / 2 + self.submap_size
        
        submaps = []
        for manager in self.submap_managers.values():
            submaps.extend(manager.get_nearby_submaps(cx, cy, radius))
        
        return submaps
    
    def get_map_image(self, width: int, height: int) -> np.ndarray:
        """
        Get rendered RGB image of the map for visualization.
        
        :param width: Output width
        :param height: Output height
        :return: RGB numpy array (width, height, 3)
        """
        probs = self.get_global_map()
        
        # Generate RGB map
        rgb_map = np.zeros((probs.shape[0], probs.shape[1], 3), dtype=np.uint8)
        
        # Colors
        rgb_map[(probs > 0.45) & (probs < 0.55)] = [50, 50, 50]  # Unknown
        rgb_map[probs <= 0.45] = [0, 0, 0]  # Free
        rgb_map[probs >= 0.55] = [255, 255, 255]  # Occupied
        
        # Transpose and resize
        rgb_map_t = rgb_map.transpose(1, 0, 2)
        resized = cv2.resize(rgb_map_t, (height, width), interpolation=cv2.INTER_NEAREST)
        
        return resized
    
    def calculate_coverage(self) -> float:
        """
        Calculate total mapped area in square meters.
        
        :return: Coverage area in m²
        """
        total = 0.0
        for manager in self.submap_managers.values():
            total += manager.calculate_total_coverage()
        return total
    
    def save_map_to_disk(self, filename: str = "final_map_mne.png"):
        """Save current map as an image."""
        probs = self.get_global_map()
        
        image = np.full(probs.shape, 127, dtype=np.uint8)
        image[probs < 0.45] = 255  # Free = White
        image[probs > 0.55] = 0    # Occupied = Black
        
        cv2.imwrite(filename, image)
        print(f"[MNE-Lite] Map saved to {filename}")
    
    def get_statistics(self) -> dict:
        """Get comprehensive statistics."""
        stats = {
            'num_agents': len(self.submap_managers),
            'total_updates': self._update_count,
            'num_loop_closures': len(self.loop_closures),
            'total_coverage_m2': self.calculate_coverage(),
        }
        
        # Per-agent stats
        for agent_id, manager in self.submap_managers.items():
            stats[f'agent_{agent_id}_submaps'] = len(manager.submaps)
            stats[f'agent_{agent_id}_updates'] = manager.total_updates
        
        # Pose graph stats
        stats.update({
            f'pose_graph_{k}': v 
            for k, v in self.pose_graph.get_statistics().items()
        })
        
        return stats


# Alias for backward compatibility
FusionServer = FusionServerMNELite
