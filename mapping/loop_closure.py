#!/usr/bin/env python

# Copyright (c) 2024
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Loop Closure Detection for MNE-Lite Distributed SLAM Architecture

Detects when agents revisit areas or encounter regions mapped by other agents.
Uses ICP (Iterative Closest Point) to align overlapping submaps for consistent fusion.
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import time
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass


@dataclass
class LoopClosure:
    """Represents a detected loop closure event."""
    agent_a: int
    agent_b: int
    timestamp: float
    position: Tuple[float, float]  # Where the closure was detected
    transform: np.ndarray  # 3x3 transformation matrix from B to A
    fitness_score: float  # ICP alignment quality (0-1)
    type: str  # 'intra' (same agent) or 'inter' (different agents)


class ICPResult(NamedTuple):
    """Result of ICP alignment."""
    transform: np.ndarray  # 3x3 transformation matrix
    fitness: float  # Alignment quality (0-1, higher is better)
    inlier_rmse: float  # Root mean square error of inliers
    converged: bool  # Whether ICP converged


def compute_icp_2d(source_points: np.ndarray, 
                   target_points: np.ndarray,
                   max_iterations: int = 50,
                   tolerance: float = 1e-6,
                   max_correspondence_dist: float = 5.0) -> ICPResult:
    """
    2D Iterative Closest Point alignment using scipy.
    
    :param source_points: Nx2 array of source points
    :param target_points: Mx2 array of target points
    :param max_iterations: Maximum number of ICP iterations
    :param tolerance: Convergence tolerance
    :param max_correspondence_dist: Maximum distance for point correspondences
    :return: ICPResult with transformation and quality metrics
    """
    if len(source_points) < 3 or len(target_points) < 3:
        return ICPResult(
            transform=np.eye(3),
            fitness=0.0,
            inlier_rmse=float('inf'),
            converged=False
        )
    
    # Build KD-tree for target points
    target_tree = KDTree(target_points)
    
    # Initialize transformation (identity)
    R_total = np.eye(2)
    t_total = np.zeros(2)
    
    # Working copy of source points
    src = source_points.copy()
    
    prev_error = float('inf')
    
    for iteration in range(max_iterations):
        # Step 1: Find nearest neighbors
        distances, indices = target_tree.query(src, k=1)
        
        # Filter by max correspondence distance
        valid_mask = distances < max_correspondence_dist
        if np.sum(valid_mask) < 3:
            break
        
        src_valid = src[valid_mask]
        tgt_valid = target_points[indices[valid_mask]]
        
        # Step 2: Compute centroids
        src_centroid = np.mean(src_valid, axis=0)
        tgt_centroid = np.mean(tgt_valid, axis=0)
        
        # Step 3: Center the points
        src_centered = src_valid - src_centroid
        tgt_centered = tgt_valid - tgt_centroid
        
        # Step 4: Compute rotation using SVD
        H = src_centered.T @ tgt_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Step 5: Compute translation
        t = tgt_centroid - R @ src_centroid
        
        # Step 6: Apply transformation
        src = (R @ src.T).T + t
        
        # Accumulate transformation
        R_total = R @ R_total
        t_total = R @ t_total + t
        
        # Check convergence
        mean_error = np.mean(distances[valid_mask])
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    
    # Compute final metrics
    final_distances, _ = target_tree.query(src, k=1)
    inlier_mask = final_distances < max_correspondence_dist
    num_inliers = np.sum(inlier_mask)
    
    fitness = num_inliers / len(source_points) if len(source_points) > 0 else 0.0
    inlier_rmse = np.sqrt(np.mean(final_distances[inlier_mask]**2)) if num_inliers > 0 else float('inf')
    
    # Build 3x3 transformation matrix
    transform = np.eye(3)
    transform[:2, :2] = R_total
    transform[:2, 2] = t_total
    
    return ICPResult(
        transform=transform,
        fitness=fitness,
        inlier_rmse=inlier_rmse,
        converged=iteration < max_iterations - 1
    )


class LoopClosureDetector:
    """
    Detects loop closures between agents based on spatial and temporal proximity.
    
    A loop closure occurs when:
    - Agent A is near a position where Agent B was at least `time_threshold` seconds ago
    - The overlap between their submaps exceeds a minimum threshold
    """
    
    def __init__(self, 
                 proximity_threshold: float = 10.0,
                 time_threshold: float = 120.0,
                 min_overlap_points: int = 50,
                 icp_fitness_threshold: float = 0.3):
        """
        Constructor method.
        
        :param proximity_threshold: Distance in meters to trigger loop closure check
        :param time_threshold: Minimum time in seconds since agent B visited the area
        :param min_overlap_points: Minimum overlapping points for valid closure
        :param icp_fitness_threshold: Minimum ICP fitness score (0-1)
        """
        self.proximity_threshold = proximity_threshold
        self.time_threshold = time_threshold
        self.min_overlap_points = min_overlap_points
        self.icp_fitness_threshold = icp_fitness_threshold
        
        # Track processed loop closures to avoid duplicates
        self._processed_closures = set()
        
        # Statistics
        self.closures_detected = 0
        self.closures_accepted = 0
    
    def check_loop_closures(self, 
                            agent_managers: Dict[int, 'SubmapManager']) -> List[LoopClosure]:
        """
        Check for loop closures between all agent pairs.
        
        :param agent_managers: Dictionary mapping agent_id to SubmapManager
        :return: List of detected LoopClosure events
        """
        closures = []
        agent_ids = list(agent_managers.keys())
        current_time = time.time()
        
        for i, agent_a_id in enumerate(agent_ids):
            manager_a = agent_managers[agent_a_id]
            
            if not manager_a.pose_history:
                continue
            
            # Current position of agent A
            _, curr_x, curr_y, curr_yaw = manager_a.pose_history[-1]
            
            # Check against all other agents (inter-agent closure)
            for agent_b_id in agent_ids:
                manager_b = agent_managers[agent_b_id]
                
                # Get historical poses of agent B near current position of A
                old_poses = manager_b.get_poses_in_region(
                    curr_x, curr_y, 
                    self.proximity_threshold,
                    min_age=self.time_threshold
                )
                
                if not old_poses:
                    continue
                
                # Check for valid loop closure
                closure = self._try_loop_closure(
                    agent_a_id, agent_b_id,
                    manager_a, manager_b,
                    curr_x, curr_y,
                    current_time
                )
                
                if closure is not None:
                    closures.append(closure)
        
        return closures
    
    def _try_loop_closure(self,
                          agent_a_id: int,
                          agent_b_id: int,
                          manager_a: 'SubmapManager',
                          manager_b: 'SubmapManager',
                          x: float, y: float,
                          timestamp: float) -> Optional[LoopClosure]:
        """
        Attempt to establish a loop closure between two agents at a position.
        
        :return: LoopClosure if successful, None otherwise
        """
        # Create closure key to avoid duplicates
        closure_key = (agent_a_id, agent_b_id, int(x / 10), int(y / 10))
        if closure_key in self._processed_closures:
            return None
        
        # Get submaps from both agents near the closure point
        submaps_a = manager_a.get_nearby_submaps(x, y, self.proximity_threshold * 2)
        submaps_b = manager_b.get_nearby_submaps(x, y, self.proximity_threshold * 2)
        
        if not submaps_a or not submaps_b:
            return None
        
        # Extract occupied points from both sets
        points_a = []
        for submap in submaps_a:
            pts = submap.get_occupied_points()
            if len(pts) > 0:
                points_a.append(pts)
        
        points_b = []
        for submap in submaps_b:
            pts = submap.get_occupied_points()
            if len(pts) > 0:
                points_b.append(pts)
        
        if not points_a or not points_b:
            return None
        
        # Combine points
        all_points_a = np.vstack(points_a)
        all_points_b = np.vstack(points_b)
        
        if len(all_points_a) < self.min_overlap_points or len(all_points_b) < self.min_overlap_points:
            return None
        
        self.closures_detected += 1
        
        # Run ICP to align B to A
        icp_result = compute_icp_2d(
            all_points_b, all_points_a,
            max_iterations=50,
            max_correspondence_dist=self.proximity_threshold
        )
        
        if icp_result.fitness < self.icp_fitness_threshold:
            return None
        
        self.closures_accepted += 1
        self._processed_closures.add(closure_key)
        
        closure_type = 'intra' if agent_a_id == agent_b_id else 'inter'
        
        return LoopClosure(
            agent_a=agent_a_id,
            agent_b=agent_b_id,
            timestamp=timestamp,
            position=(x, y),
            transform=icp_result.transform,
            fitness_score=icp_result.fitness,
            type=closure_type
        )
    
    def get_statistics(self) -> dict:
        """Get loop closure statistics."""
        return {
            'closures_detected': self.closures_detected,
            'closures_accepted': self.closures_accepted,
            'acceptance_rate': self.closures_accepted / max(1, self.closures_detected)
        }
    
    def reset(self):
        """Reset the detector state."""
        self._processed_closures.clear()
        self.closures_detected = 0
        self.closures_accepted = 0
