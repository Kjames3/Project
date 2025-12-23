#!/usr/bin/env python

# Copyright (c) 2024
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Unit tests for MNE-Lite Pose Graph Optimization.
"""

import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mapping.pose_graph import PoseGraph, PoseNode, PoseEdge, normalize_angle, compute_error


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_normalize_angle(self):
        """Test angle normalization."""
        self.assertAlmostEqual(normalize_angle(0.0), 0.0)
        self.assertAlmostEqual(normalize_angle(np.pi), np.pi, places=5)
        self.assertAlmostEqual(normalize_angle(-np.pi), -np.pi, places=5)
        self.assertAlmostEqual(normalize_angle(2 * np.pi), 0.0, places=5)
        self.assertAlmostEqual(normalize_angle(3 * np.pi), np.pi, places=5)
        self.assertAlmostEqual(normalize_angle(-3 * np.pi), -np.pi, places=5)
    
    def test_compute_error(self):
        """Test error computation."""
        pose_i = (0.0, 0.0, 0.0)
        pose_j = (1.0, 0.0, 0.0)
        measurement = (1.0, 0.0, 0.0)
        
        error = compute_error(pose_i, pose_j, measurement)
        
        np.testing.assert_array_almost_equal(error, [0.0, 0.0, 0.0])
    
    def test_compute_error_with_rotation(self):
        """Test error computation with rotation."""
        pose_i = (0.0, 0.0, np.pi/2)  # Facing up
        pose_j = (0.0, 1.0, np.pi/2)
        measurement = (1.0, 0.0, 0.0)  # Should see j as 1m ahead
        
        error = compute_error(pose_i, pose_j, measurement)
        
        # Error should be small
        self.assertLess(np.abs(error[0]), 0.1)
        self.assertLess(np.abs(error[1]), 0.1)


class TestPoseGraph(unittest.TestCase):
    """Test cases for PoseGraph class."""
    
    def test_initialization(self):
        """Test pose graph initialization."""
        graph = PoseGraph()
        
        self.assertEqual(len(graph.nodes), 0)
        self.assertEqual(len(graph.edges), 0)
    
    def test_add_node(self):
        """Test adding nodes."""
        graph = PoseGraph()
        
        node = graph.add_node(0, 1.0, 2.0, 0.5, fixed=True)
        
        self.assertEqual(len(graph.nodes), 1)
        self.assertEqual(node.id, 0)
        self.assertEqual(node.x, 1.0)
        self.assertEqual(node.y, 2.0)
        self.assertEqual(node.theta, 0.5)
        self.assertTrue(node.fixed)
    
    def test_add_odometry_edge(self):
        """Test adding odometry edges."""
        graph = PoseGraph()
        
        graph.add_node(0, 0.0, 0.0, 0.0)
        graph.add_node(1, 1.0, 0.0, 0.0)
        
        graph.add_odometry_edge(0, 1, 1.0, 0.0, 0.0)
        
        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(graph.edges[0].edge_type, 'odometry')
    
    def test_add_loop_closure_edge(self):
        """Test adding loop closure edges."""
        graph = PoseGraph()
        
        graph.add_node(0, 0.0, 0.0, 0.0)
        graph.add_node(1, 5.0, 0.0, 0.0)
        
        graph.add_loop_closure_edge(0, 1, 5.0, 0.0, 0.0)
        
        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(graph.edges[0].edge_type, 'loop_closure')
    
    def test_get_pose(self):
        """Test getting node pose."""
        graph = PoseGraph()
        
        graph.add_node(5, 3.0, 4.0, 1.5)
        
        pose = graph.get_pose(5)
        
        self.assertIsNotNone(pose)
        self.assertEqual(pose[0], 3.0)
        self.assertEqual(pose[1], 4.0)
        self.assertEqual(pose[2], 1.5)
        
        # Non-existent node
        self.assertIsNone(graph.get_pose(999))
    
    def test_optimize_simple_chain(self):
        """Test optimization of a simple chain."""
        graph = PoseGraph()
        
        # Create a chain of 3 nodes
        graph.add_node(0, 0.0, 0.0, 0.0, fixed=True)  # Anchor
        graph.add_node(1, 1.0, 0.0, 0.0)
        graph.add_node(2, 2.0, 0.0, 0.0)
        
        # Add odometry edges
        graph.add_odometry_edge(0, 1, 1.0, 0.0, 0.0)
        graph.add_odometry_edge(1, 2, 1.0, 0.0, 0.0)
        
        # Optimize
        converged = graph.optimize()
        
        self.assertTrue(converged)
        
        # Poses should remain close to initial (no conflicting constraints)
        pose1 = graph.get_pose(1)
        pose2 = graph.get_pose(2)
        
        self.assertAlmostEqual(pose1[0], 1.0, delta=0.5)
        self.assertAlmostEqual(pose2[0], 2.0, delta=0.5)
    
    def test_optimize_with_loop_closure(self):
        """Test optimization with loop closure constraint."""
        graph = PoseGraph()
        
        # Create a square path with accumulated drift
        graph.add_node(0, 0.0, 0.0, 0.0, fixed=True)
        graph.add_node(1, 10.0, 0.0, np.pi/2)
        graph.add_node(2, 10.0, 10.0, np.pi)
        graph.add_node(3, 0.0, 10.0, -np.pi/2)
        graph.add_node(4, 0.5, 0.5, 0.0)  # Drifted back, should be (0,0)
        
        # Odometry edges
        graph.add_odometry_edge(0, 1, 10.0, 0.0, np.pi/2)
        graph.add_odometry_edge(1, 2, 10.0, 0.0, np.pi/2)
        graph.add_odometry_edge(2, 3, 10.0, 0.0, np.pi/2)
        graph.add_odometry_edge(3, 4, 10.0, 0.0, np.pi/2)
        
        # Loop closure: node 4 should be at node 0
        graph.add_loop_closure_edge(0, 4, 0.0, 0.0, 0.0)
        
        # Optimize
        converged = graph.optimize(max_iterations=100)
        
        self.assertTrue(converged)
        
        # Node 4 should have moved closer to origin
        pose4 = graph.get_pose(4)
        dist_to_origin = np.sqrt(pose4[0]**2 + pose4[1]**2)
        self.assertLess(dist_to_origin, 1.0)  # Should be close to 0
    
    def test_statistics(self):
        """Test statistics collection."""
        graph = PoseGraph()
        
        graph.add_node(0, 0.0, 0.0, 0.0)
        graph.add_node(1, 1.0, 0.0, 0.0)
        graph.add_odometry_edge(0, 1, 1.0, 0.0, 0.0)
        graph.add_loop_closure_edge(0, 1, 1.0, 0.0, 0.0)
        
        stats = graph.get_statistics()
        
        self.assertEqual(stats['num_nodes'], 2)
        self.assertEqual(stats['num_edges'], 2)
        self.assertEqual(stats['num_odometry_edges'], 1)
        self.assertEqual(stats['num_loop_closure_edges'], 1)
    
    def test_clear(self):
        """Test clearing the graph."""
        graph = PoseGraph()
        
        graph.add_node(0, 0.0, 0.0, 0.0)
        graph.add_node(1, 1.0, 0.0, 0.0)
        graph.add_odometry_edge(0, 1, 1.0, 0.0, 0.0)
        
        graph.clear()
        
        self.assertEqual(len(graph.nodes), 0)
        self.assertEqual(len(graph.edges), 0)


if __name__ == '__main__':
    unittest.main()
