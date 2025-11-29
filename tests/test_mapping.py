import unittest
import numpy as np
from mapping.occupancy_grid import LocalMapper

class MockVehicle:
    pass

class TestLocalMapper(unittest.TestCase):
    def test_process_lidar(self):
        vehicle = MockVehicle()
        mapper = LocalMapper(vehicle, grid_size=0.5, window_size=20.0) # 40x40 grid
        
        # Create mock Semantic LiDAR data
        # Format: x, y, z, cos, idx, tag
        # Center is (20, 20) in grid indices
        # Grid size 0.5 -> 20m / 0.5 = 40 cells
        
        # Point 1: Obstacle (Building, tag 3) at x=5.0, y=0.0
        # Point 2: Road (tag 1) at x=2.0, y=0.0
        # Point 3: Vehicle (tag 10) at x=3.0, y=0.0
        
        # x=5.0 -> 10 cells forward. 
        # Center is 20. Top is 0.
        # Row = 20 - (5.0/0.5) = 20 - 10 = 10.
        # Col = 20 + (0.0/0.5) = 20.
        
        # x=2.0 -> 4 cells forward. Row = 16.
        # x=3.0 -> 6 cells forward. Row = 14.
        
        data = np.array([
            (5.0, 0.0, 0.0, 0.0, 0, 3), # Building
            (2.0, 0.0, 0.0, 0.0, 0, 1), # Road
            (3.0, 0.0, 0.0, 0.0, 0, 10) # Vehicle
        ], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('cos', 'f4'), ('idx', 'u4'), ('tag', 'u4')])
        
        class MockLidarData:
            def __init__(self, data):
                self.raw_data = data.tobytes()
                
        lidar_data = MockLidarData(data)
        
        mapper.process_lidar(lidar_data, None)
        local_map = mapper.get_local_map()
        
        # Check Obstacle (Building) -> Occupied (1.0)
        self.assertEqual(local_map[10, 20], 1.0)
        
        # Check Road -> Free (0.0) (because raycast passes through or ends there)
        # Raycast should clear path to (16, 20).
        # And (16, 20) itself should be Free because Road is not Static Obstacle.
        self.assertEqual(local_map[16, 20], 0.0)
        
        # Check Vehicle -> Free (0.0) (Dynamic object ignored as obstacle)
        self.assertEqual(local_map[14, 20], 0.0)
        
        # Check Free Space between center (20, 20) and points
        # e.g. (18, 20) should be Free
        self.assertEqual(local_map[18, 20], 0.0)
        
        print("TestLocalMapper Passed")

if __name__ == '__main__':
    unittest.main()
