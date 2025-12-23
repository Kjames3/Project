#!/usr/bin/env python

# Copyright (c) 2024
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Unit tests for MNE-Lite Loop Closure Detection and ICP.
"""

import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mapping.loop_closure import compute_icp_2d, LoopClosureDetector, ICPResult


class TestICP(unittest.TestCase):
    """Test cases for ICP algorithm."""
    
    def test_identity_alignment(self):
        """Test ICP with identical point clouds."""
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5]
        ], dtype=np.float32)
        
        result = compute_icp_2d(points, points)
        
        self.assertTrue(result.converged)
        self.assertGreater(result.fitness, 0.9)
        
        # Transform should be close to identity
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(result.transform, expected, decimal=2)
    
    def test_translation_alignment(self):
        """Test ICP with translated point cloud."""
        source = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ], dtype=np.float32)
        
        # Translate by (2, 3)
        target = source + np.array([2.0, 3.0])
        
        result = compute_icp_2d(source, target)
        
        self.assertTrue(result.converged)
        self.assertGreater(result.fitness, 0.9)
        
        # Check translation component
        self.assertAlmostEqual(result.transform[0, 2], 2.0, delta=0.5)
        self.assertAlmostEqual(result.transform[1, 2], 3.0, delta=0.5)
    
    def test_rotation_alignment(self):
        """Test ICP with rotated point cloud."""
        source = np.array([
            [0.0, 0.0],
            [5.0, 0.0],
            [0.0, 5.0],
            [5.0, 5.0],
            [2.5, 2.5]
        ], dtype=np.float32)
        
        # Rotate by 45 degrees
        angle = np.pi / 4
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        target = (R @ source.T).T
        
        result = compute_icp_2d(source, target, max_iterations=100)
        
        # Should converge with reasonable fitness
        self.assertGreater(result.fitness, 0.5)
    
    def test_insufficient_points(self):
        """Test ICP with too few points."""
        source = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        target = np.array([[0.0, 0.0]], dtype=np.float32)
        
        result = compute_icp_2d(source, target)
        
        self.assertFalse(result.converged)
        self.assertEqual(result.fitness, 0.0)
    
    def test_empty_point_clouds(self):
        """Test ICP with empty point clouds."""
        empty = np.array([]).reshape(0, 2)
        
        result = compute_icp_2d(empty, empty)
        
        self.assertFalse(result.converged)


class TestLoopClosureDetector(unittest.TestCase):
    """Test cases for LoopClosureDetector."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = LoopClosureDetector(
            proximity_threshold=10.0,
            time_threshold=60.0,
            min_overlap_points=20,
            icp_fitness_threshold=0.3
        )
        
        self.assertEqual(detector.proximity_threshold, 10.0)
        self.assertEqual(detector.time_threshold, 60.0)
        self.assertEqual(detector.min_overlap_points, 20)
        self.assertEqual(detector.icp_fitness_threshold, 0.3)
    
    def test_empty_managers(self):
        """Test with no agents."""
        detector = LoopClosureDetector()
        
        closures = detector.check_loop_closures({})
        
        self.assertEqual(len(closures), 0)
    
    def test_statistics(self):
        """Test statistics tracking."""
        detector = LoopClosureDetector()
        
        stats = detector.get_statistics()
        
        self.assertEqual(stats['closures_detected'], 0)
        self.assertEqual(stats['closures_accepted'], 0)
        self.assertEqual(stats['acceptance_rate'], 0.0)
    
    def test_reset(self):
        """Test detector reset."""
        detector = LoopClosureDetector()
        detector.closures_detected = 10
        detector.closures_accepted = 5
        
        detector.reset()
        
        self.assertEqual(detector.closures_detected, 0)
        self.assertEqual(detector.closures_accepted, 0)


if __name__ == '__main__':
    unittest.main()
