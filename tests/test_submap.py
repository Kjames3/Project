#!/usr/bin/env python

# Copyright (c) 2024
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Unit tests for MNE-Lite Submap module.
"""

import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mapping.submap import Submap


class TestSubmap(unittest.TestCase):
    """Test cases for Submap class."""
    
    def test_initialization(self):
        """Test submap initialization."""
        submap = Submap(origin_x=0.0, origin_y=0.0, submap_size=50.0, grid_size=0.5)
        
        self.assertEqual(submap.origin_x, 0.0)
        self.assertEqual(submap.origin_y, 0.0)
        self.assertEqual(submap.submap_size, 50.0)
        self.assertEqual(submap.grid_size, 0.5)
        self.assertEqual(submap.dim, 100)  # 50m / 0.5m
        self.assertEqual(submap.log_odds.shape, (100, 100))
        
    def test_world_to_grid(self):
        """Test world to grid coordinate conversion."""
        submap = Submap(origin_x=0.0, origin_y=0.0, submap_size=50.0, grid_size=0.5)
        
        # Center of submap
        result = submap.world_to_grid(25.0, 25.0)
        self.assertIsNotNone(result)
        
        # Origin corner
        result = submap.world_to_grid(0.0, 0.0)
        self.assertIsNotNone(result)
        
        # Out of bounds
        result = submap.world_to_grid(-10.0, 0.0)
        self.assertIsNone(result)
        
        result = submap.world_to_grid(60.0, 25.0)
        self.assertIsNone(result)
    
    def test_grid_to_world(self):
        """Test grid to world coordinate conversion."""
        submap = Submap(origin_x=10.0, origin_y=20.0, submap_size=50.0, grid_size=0.5)
        
        # Test round-trip conversion
        for wx, wy in [(35.0, 45.0), (15.0, 25.0), (55.0, 65.0)]:
            grid_pos = submap.world_to_grid(wx, wy)
            if grid_pos:
                wx2, wy2 = submap.grid_to_world(grid_pos[0], grid_pos[1])
                self.assertAlmostEqual(wx, wx2, delta=submap.grid_size)
                self.assertAlmostEqual(wy, wy2, delta=submap.grid_size)
    
    def test_contains_point(self):
        """Test point containment check."""
        submap = Submap(origin_x=0.0, origin_y=0.0, submap_size=50.0, grid_size=0.5)
        
        self.assertTrue(submap.contains_point(25.0, 25.0))
        self.assertTrue(submap.contains_point(0.0, 0.0))
        self.assertTrue(submap.contains_point(49.9, 49.9))
        
        self.assertFalse(submap.contains_point(-1.0, 25.0))
        self.assertFalse(submap.contains_point(50.0, 25.0))
        self.assertFalse(submap.contains_point(25.0, 50.0))
    
    def test_update(self):
        """Test updating submap with local grid."""
        submap = Submap(origin_x=0.0, origin_y=0.0, submap_size=50.0, grid_size=0.5)
        
        # Create a simple local grid
        local_grid = np.full((40, 40), 0.5, dtype=np.float32)
        local_grid[20, 20] = 0.0  # Free at center
        local_grid[15, 20] = 1.0  # Occupied ahead
        
        # Update at center of submap
        submap.update(local_grid, 25.0, 25.0, 0.0)
        
        self.assertEqual(submap.update_count, 1)
        
        # Check probability map
        probs = submap.get_probability_map()
        self.assertEqual(probs.shape, (100, 100))
    
    def test_get_occupied_points(self):
        """Test extraction of occupied points."""
        submap = Submap(origin_x=0.0, origin_y=0.0, submap_size=50.0, grid_size=0.5)
        
        # Manually set some cells as occupied
        submap.log_odds[50, 50] = 3.0  # High confidence occupied
        submap.log_odds[51, 50] = 3.0
        submap._cache_dirty = True
        
        points = submap.get_occupied_points(threshold=0.55)
        
        self.assertEqual(len(points), 2)
        self.assertEqual(points.shape[1], 2)  # x, y columns
    
    def test_get_bounds(self):
        """Test bounds calculation."""
        submap = Submap(origin_x=10.0, origin_y=20.0, submap_size=50.0, grid_size=0.5)
        
        bounds = submap.get_bounds()
        
        self.assertEqual(bounds[0], 10.0)   # min_x
        self.assertEqual(bounds[1], 60.0)   # max_x
        self.assertEqual(bounds[2], 20.0)   # min_y
        self.assertEqual(bounds[3], 70.0)   # max_y
    
    def test_calculate_coverage(self):
        """Test coverage calculation."""
        submap = Submap(origin_x=0.0, origin_y=0.0, submap_size=50.0, grid_size=0.5)
        
        # Initially no coverage
        self.assertEqual(submap.calculate_coverage(), 0.0)
        
        # Set some cells as observed
        submap.log_odds[0:10, 0:10] = 3.0  # 10x10 occupied
        submap.log_odds[20:30, 20:30] = -3.0  # 10x10 free
        submap._cache_dirty = True
        
        coverage = submap.calculate_coverage()
        expected = 200 * (0.5 ** 2)  # 200 cells * 0.25 m²
        self.assertEqual(coverage, expected)


class TestSubmapMerge(unittest.TestCase):
    """Test cases for Submap merging."""
    
    def test_merge_without_transform(self):
        """Test merging two overlapping submaps."""
        submap_a = Submap(origin_x=0.0, origin_y=0.0, submap_size=50.0, grid_size=0.5)
        submap_b = Submap(origin_x=0.0, origin_y=0.0, submap_size=50.0, grid_size=0.5)
        
        # Set different cells in each
        submap_a.log_odds[50, 50] = 3.0
        submap_b.log_odds[60, 60] = -3.0
        
        # Merge B into A
        submap_a.merge(submap_b)
        
        # Both cells should now have values
        self.assertGreater(submap_a.log_odds[50, 50], 0)
        self.assertLess(submap_a.log_odds[60, 60], 0)


if __name__ == '__main__':
    unittest.main()
