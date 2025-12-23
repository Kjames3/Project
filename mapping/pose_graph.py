#!/usr/bin/env python

# Copyright (c) 2024
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Pose Graph Optimization for MNE-Lite Distributed SLAM Architecture

Lightweight pose graph optimizer using scipy for Gauss-Newton optimization.
Maintains relative poses between submaps and propagates corrections after loop closure.
"""

import numpy as np
from scipy.optimize import least_squares
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class PoseNode:
    """A node in the pose graph representing a submap pose."""
    id: int
    x: float
    y: float
    theta: float  # In radians
    fixed: bool = False  # Fixed nodes don't get optimized
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta])
    
    @staticmethod
    def from_array(id: int, arr: np.ndarray, fixed: bool = False) -> 'PoseNode':
        return PoseNode(id, arr[0], arr[1], arr[2], fixed)


@dataclass
class PoseEdge:
    """An edge in the pose graph representing a constraint between nodes."""
    from_id: int
    to_id: int
    delta_x: float
    delta_y: float
    delta_theta: float
    information: np.ndarray  # 3x3 information matrix (inverse covariance)
    edge_type: str  # 'odometry' or 'loop_closure'
    
    def measurement(self) -> np.ndarray:
        return np.array([self.delta_x, self.delta_y, self.delta_theta])


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def pose_to_transform(x, y, theta) -> np.ndarray:
    """Convert pose (x, y, theta) to 3x3 transformation matrix."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, x],
        [s, c, y],
        [0, 0, 1]
    ])


def transform_to_pose(T: np.ndarray) -> Tuple[float, float, float]:
    """Extract pose (x, y, theta) from 3x3 transformation matrix."""
    x = T[0, 2]
    y = T[1, 2]
    theta = np.arctan2(T[1, 0], T[0, 0])
    return (x, y, theta)


def compute_error(pose_i, pose_j, measurement) -> np.ndarray:
    """
    Compute the error between predicted and measured relative pose.
    
    :param pose_i: (x, y, theta) of node i
    :param pose_j: (x, y, theta) of node j
    :param measurement: (dx, dy, dtheta) measured between i and j
    :return: 3D error vector
    """
    xi, yi, thi = pose_i
    xj, yj, thj = pose_j
    dx_m, dy_m, dth_m = measurement
    
    # Predicted relative pose (j in frame of i)
    c, s = np.cos(thi), np.sin(thi)
    dx_pred = c * (xj - xi) + s * (yj - yi)
    dy_pred = -s * (xj - xi) + c * (yj - yi)
    dth_pred = normalize_angle(thj - thi)
    
    # Error
    error = np.array([
        dx_pred - dx_m,
        dy_pred - dy_m,
        normalize_angle(dth_pred - dth_m)
    ])
    
    return error


class PoseGraph:
    """
    Pose graph for multi-agent SLAM.
    
    Stores nodes (submap poses) and edges (odometry and loop closure constraints).
    Uses Gauss-Newton optimization via scipy to minimize pose errors.
    """
    
    def __init__(self):
        """Constructor method."""
        self.nodes: Dict[int, PoseNode] = {}
        self.edges: List[PoseEdge] = []
        
        # Default information matrices
        self._odometry_info = np.diag([100.0, 100.0, 1000.0])  # High confidence
        self._loop_closure_info = np.diag([50.0, 50.0, 500.0])  # Lower confidence
        
        # Optimization parameters
        self.max_iterations = 100
        self.convergence_threshold = 1e-6
        
        # Statistics
        self.optimization_count = 0
        self.last_residual = 0.0
    
    def add_node(self, node_id: int, x: float, y: float, theta: float, 
                 fixed: bool = False) -> PoseNode:
        """
        Add a node to the pose graph.
        
        :param node_id: Unique identifier for the node
        :param x: X position
        :param y: Y position
        :param theta: Orientation in radians
        :param fixed: If True, node won't be optimized
        :return: The created PoseNode
        """
        node = PoseNode(node_id, x, y, theta, fixed)
        self.nodes[node_id] = node
        return node
    
    def add_odometry_edge(self, from_id: int, to_id: int, 
                          delta_x: float, delta_y: float, delta_theta: float,
                          information: Optional[np.ndarray] = None):
        """
        Add an odometry constraint between two nodes.
        
        :param from_id: Source node ID
        :param to_id: Target node ID
        :param delta_x: Relative X translation
        :param delta_y: Relative Y translation
        :param delta_theta: Relative rotation
        :param information: Optional 3x3 information matrix
        """
        if from_id not in self.nodes or to_id not in self.nodes:
            return
        
        info = information if information is not None else self._odometry_info
        
        edge = PoseEdge(
            from_id=from_id,
            to_id=to_id,
            delta_x=delta_x,
            delta_y=delta_y,
            delta_theta=delta_theta,
            information=info,
            edge_type='odometry'
        )
        self.edges.append(edge)
    
    def add_loop_closure_edge(self, from_id: int, to_id: int,
                               delta_x: float, delta_y: float, delta_theta: float,
                               information: Optional[np.ndarray] = None):
        """
        Add a loop closure constraint between two nodes.
        
        :param from_id: Source node ID
        :param to_id: Target node ID
        :param delta_x: Relative X translation from ICP
        :param delta_y: Relative Y translation from ICP
        :param delta_theta: Relative rotation from ICP
        :param information: Optional 3x3 information matrix
        """
        if from_id not in self.nodes or to_id not in self.nodes:
            return
        
        info = information if information is not None else self._loop_closure_info
        
        edge = PoseEdge(
            from_id=from_id,
            to_id=to_id,
            delta_x=delta_x,
            delta_y=delta_y,
            delta_theta=delta_theta,
            information=info,
            edge_type='loop_closure'
        )
        self.edges.append(edge)
    
    def _pack_poses(self) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Pack node poses into a single vector for optimization.
        
        :return: (pose_vector, variable_ids, fixed_ids)
        """
        variable_ids = []
        fixed_ids = []
        poses = []
        
        for node_id in sorted(self.nodes.keys()):
            node = self.nodes[node_id]
            if node.fixed:
                fixed_ids.append(node_id)
            else:
                variable_ids.append(node_id)
                poses.extend([node.x, node.y, node.theta])
        
        return np.array(poses), variable_ids, fixed_ids
    
    def _unpack_poses(self, pose_vector: np.ndarray, variable_ids: List[int]):
        """
        Unpack optimized poses back to nodes.
        
        :param pose_vector: Optimized pose vector
        :param variable_ids: List of variable node IDs
        """
        for i, node_id in enumerate(variable_ids):
            idx = i * 3
            self.nodes[node_id].x = pose_vector[idx]
            self.nodes[node_id].y = pose_vector[idx + 1]
            self.nodes[node_id].theta = normalize_angle(pose_vector[idx + 2])
    
    def _residual_function(self, pose_vector: np.ndarray, 
                           variable_ids: List[int]) -> np.ndarray:
        """
        Compute residuals for all edges.
        
        :param pose_vector: Current pose estimates
        :param variable_ids: List of variable node IDs
        :return: Weighted residual vector
        """
        # Create ID to pose mapping
        id_to_pose = {}
        for i, node_id in enumerate(variable_ids):
            idx = i * 3
            id_to_pose[node_id] = (
                pose_vector[idx],
                pose_vector[idx + 1],
                pose_vector[idx + 2]
            )
        
        # Add fixed nodes
        for node_id, node in self.nodes.items():
            if node.fixed:
                id_to_pose[node_id] = (node.x, node.y, node.theta)
        
        # Compute residuals
        residuals = []
        for edge in self.edges:
            if edge.from_id not in id_to_pose or edge.to_id not in id_to_pose:
                continue
            
            pose_i = id_to_pose[edge.from_id]
            pose_j = id_to_pose[edge.to_id]
            measurement = edge.measurement()
            
            error = compute_error(pose_i, pose_j, measurement)
            
            # Weight by square root of information matrix (Cholesky-like)
            L = np.linalg.cholesky(edge.information)
            weighted_error = L @ error
            
            residuals.extend(weighted_error)
        
        return np.array(residuals)
    
    def optimize(self, max_iterations: Optional[int] = None) -> bool:
        """
        Optimize the pose graph using Gauss-Newton via scipy.
        
        :param max_iterations: Optional override for max iterations
        :return: True if optimization converged
        """
        if len(self.nodes) < 2 or len(self.edges) == 0:
            return True
        
        iterations = max_iterations if max_iterations else self.max_iterations
        
        # Pack poses
        initial_poses, variable_ids, fixed_ids = self._pack_poses()
        
        if len(variable_ids) == 0:
            return True  # All nodes are fixed
        
        # Run optimization
        result = least_squares(
            self._residual_function,
            initial_poses,
            args=(variable_ids,),
            method='lm',  # Levenberg-Marquardt
            max_nfev=iterations,
            ftol=self.convergence_threshold,
            xtol=self.convergence_threshold
        )
        
        # Unpack results
        self._unpack_poses(result.x, variable_ids)
        
        self.optimization_count += 1
        self.last_residual = np.sum(result.fun ** 2)
        
        return result.success
    
    def get_pose(self, node_id: int) -> Optional[Tuple[float, float, float]]:
        """
        Get the current pose of a node.
        
        :param node_id: Node ID
        :return: (x, y, theta) or None
        """
        if node_id not in self.nodes:
            return None
        node = self.nodes[node_id]
        return (node.x, node.y, node.theta)
    
    def get_correction(self, node_id: int, 
                       original_pose: Tuple[float, float, float]) -> np.ndarray:
        """
        Compute the correction transform for a node after optimization.
        
        :param node_id: Node ID
        :param original_pose: Original (x, y, theta) before optimization
        :return: 3x3 correction transformation matrix
        """
        current_pose = self.get_pose(node_id)
        if current_pose is None:
            return np.eye(3)
        
        T_original = pose_to_transform(*original_pose)
        T_current = pose_to_transform(*current_pose)
        
        # Correction: T_current = T_correction @ T_original
        # T_correction = T_current @ inv(T_original)
        T_correction = T_current @ np.linalg.inv(T_original)
        
        return T_correction
    
    def get_statistics(self) -> dict:
        """Get optimization statistics."""
        num_odometry = sum(1 for e in self.edges if e.edge_type == 'odometry')
        num_loop = sum(1 for e in self.edges if e.edge_type == 'loop_closure')
        
        return {
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges),
            'num_odometry_edges': num_odometry,
            'num_loop_closure_edges': num_loop,
            'optimization_count': self.optimization_count,
            'last_residual': self.last_residual
        }
    
    def clear(self):
        """Clear all nodes and edges."""
        self.nodes.clear()
        self.edges.clear()
        self.optimization_count = 0
        self.last_residual = 0.0
