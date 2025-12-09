
import unittest
import numpy as np
import sys
import os

# Adjust path to import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sys
import types

# MOCK NUMBA if not present
try:
    import numba
except ImportError:
    # Create a mock numba module
    numba = types.ModuleType('numba')
    
    # Mock njit decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    numba.njit = njit
    sys.modules['numba'] = numba

# MOCK CARLA if not present
try:
    import carla
except ImportError:
    carla = types.ModuleType('carla')
    
    class MockTransform:
        def __init__(self):
            # Location with distance method
            class Location:
                def __init__(self, x, y, z):
                    self.x = x
                    self.y = y
                    self.z = z
                def distance(self, other):
                    return ((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)**0.5
            
            self.location = Location(0,0,0)
            self.rotation = type('obj', (object,), {'yaw':0, 'pitch':0, 'roll':0})
            
    class MockWaypoint:
        def __init__(self, x=0, y=0, yaw=0, is_junction=False):
            self.transform = MockTransform()
            self.transform.location.x = x
            self.transform.location.y = y
            self.transform.location.z = 0
            self.transform.rotation.yaw = yaw
            self.is_junction = is_junction

    carla.Transform = MockTransform
    carla.Waypoint = MockWaypoint
    carla.VehicleControl = type('VehicleControl', (object,), {'throttle':0, 'brake':0, 'steer':0})
    
    sys.modules['carla'] = carla

from mapping.occupancy_grid import voxel_filter
from agents.navigation.controller import HybridLateralController

class MockVehicle:
    def __init__(self, location=(0,0,0), yaw=0, velocity=(0,0,0)):
        # Define Location class locally or reuse from MockTransform if available
        class Location:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
            def distance(self, other):
                return ((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)**0.5
                
        self.location = Location(location[0], location[1], location[2])
        self.rotation = type('obj', (object,), {'yaw': yaw})
        self.velocity = type('obj', (object,), {'x': velocity[0], 'y': velocity[1], 'z': velocity[2]})
        self.transform = type('obj', (object,), {'location': self.location, 'rotation': self.rotation})

    def get_transform(self):
        return self.transform

    def get_velocity(self):
        return self.velocity
        
    def get_location(self):
        return self.location

class TestOptimizations(unittest.TestCase):
    
    def test_voxel_filter(self):
        print("\n--- Testing Voxel Filter ---")
        # Create a cloud with points very close together
        points = np.array([
            [1.0, 1.0, 0.0],
            [1.02, 1.02, 0.0], # Should be same voxel as above
            [1.0, 1.0, 0.05],  # Should be same voxel as above
            [2.0, 2.0, 0.0],   # Different voxel
            [1.5, 1.5, 0.0]    # Different voxel
        ], dtype=np.float32)
        
        voxel_size = 0.1
        
        indices = voxel_filter(points, voxel_size)
        
        # Unique rows in indices
        unique_indices = np.unique(indices, axis=0)
        
        print(f"Original points: {len(points)}")
        print(f"Unique voxels: {len(unique_indices)}")
        
        # We expect 3 unique voxels (approx)
        self.assertEqual(len(unique_indices), 3)

    def test_hybrid_controller_logic(self):
        print("\n--- Testing Hybrid Controller Logic ---")
        from collections import deque
        
        # Vehicle at (0,0), Yaw 0.
        vehicle = MockVehicle(location=(0,0,0), yaw=0, velocity=(5,0,0))
        controller = HybridLateralController(vehicle, wheelbase=2.875)
        
        # Case 1: Straight Road (Stanley)
        # Waypoint at (10, 2), Yaw 0. Not junction.
        wp_straight = MockWaypoint(10, 2, 0, is_junction=False)
        waypoints_straight = deque([(wp_straight, None)])
        
        steer_straight = controller.run_step(waypoints_straight)
        print(f"Straight/Stanley Steer: {steer_straight:.3f}")
        # Target at (10, 2) which is Right (Y+) of vehicle (0,0).
        # Should steer Right (Positive).
        self.assertGreater(steer_straight, 0.0, "Should steer Right (Positive) towards target at Y=2")

        # Case 2: Junction (Pure Pursuit)
        # Waypoint at (10, 2). is_junction=True.
        wp_junction = MockWaypoint(10, 2, 0, is_junction=True)
        waypoints_junction = deque([(wp_junction, None)])
        
        steer_junction = controller.run_step(waypoints_junction)
        print(f"Junction/PP Steer: {steer_junction:.3f}")
        self.assertNotEqual(steer_junction, 0.0)
        
        # Verify they might produce slightly different values or at least run without error
        # Pure Pursuit vs Stanley logic is different.

    def test_fast_frontier(self):
        print("\n--- Testing Fast Frontier ---")
        from hybrid_planner import HybridRoutePlanner
        
        # Mock Fusion Server
        class MockFusionServer:
            def __init__(self):
                self.grid_size = 1.0
                self.grid_dim = 100
                self.min_x = 0
                self.min_y = 0
                self.max_x = 100
                self.max_y = 100
                # Create a map with a frontier at (90, 90) -> indices (10, 90)
                # Map coords: x=0..100, y=0..100.
                # Grid indices: r = (max_x - x)/res. c = (y - min_y)/res.
                self.global_map = np.zeros((100, 100), dtype=np.float32)
                
                # Set a region to Unknown (0.5)
                # Set adjacent to Free (0.0)
                
                # Let's put a frontier at middle (50, 50). Indices (50, 50).
                # r=50, c=50.
                self.global_map[50, 50] = 0.5 # Unknown
                self.global_map[50, 49] = 0.0 # Free
                # Frontier should be detected at (50, 49) where it touches unknown.
                
                # Put another far away at (10, 10). Indices (90, 10)
                self.global_map[90, 10] = 0.5
                self.global_map[90, 9] = 0.0
                
            def get_global_map(self):
                return self.global_map

        fusion = MockFusionServer()
        planner = HybridRoutePlanner(fusion)
        
        # 1. Search near center (50, 50). Radius 10.
        # Should find frontier at (50, 49) indices.
        # Global coords for indices (50, 49):
        # x = 100 - 50*1 = 50.
        # y = 0 + 49*1 = 49.
        # Ego at (50, 45).
        
        ego_loc = (50, 45) # Tuple (x,y)
        frontiers = planner.get_frontiers(ego_location=ego_loc, search_radius=10.0)
        
        print(f"Frontiers found (Radius 10): {len(frontiers)}")
        found_center = False
        for f in frontiers:
            if abs(f[0] - 50) < 0.1 and abs(f[1] - 49) < 0.1:
                found_center = True
        
        self.assertTrue(found_center, "Should find frontier near center")
        
        # Should NOT find the one at (10, 10) [indices 90, 10] -> Global x=10, y=9.
        # Distance from (50, 45) to (10, 9) is ~53m. Radius is 10.
        
        found_far = False
        for f in frontiers:
             if abs(f[0] - 10) < 0.1 and abs(f[1] - 9) < 0.1:
                found_far = True
        
        self.assertFalse(found_far, "Should NOT find frontier far away")
        
if __name__ == '__main__':
    unittest.main()
