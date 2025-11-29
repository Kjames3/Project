import unittest
import numpy as np
from mapping.fusion_server import FusionServer

class TestFusionServer(unittest.TestCase):
    def test_update_map(self):
        server = FusionServer(grid_size=1.0, map_dim=100) # 100x100 grid
        # Center (0,0) is at index (50, 50) roughly?
        # Min X = -50, Max X = 50.
        # Row = (50 - X) / 1.
        # Col = (Y - (-50)) / 1 = Y + 50.
        
        # Agent at (0, 0, 0) (Global)
        pose = (0.0, 0.0, 0.0)
        
        # Local Map: 3x3
        # Center (1, 1) corresponds to Agent (0,0)
        # (0, 1) is Forward (x=1, y=0) -> Global (1, 0)
        # (1, 2) is Right (x=0, y=1) -> Global (0, 1)
        
        local_map = np.array([
            [0.5, 1.0, 0.5], # Row 0: Forward. (0,1) is Occupied.
            [0.5, 0.0, 0.5], # Row 1: Center. (1,1) is Free.
            [0.5, 0.5, 0.5]
        ], dtype=np.float32)
        
        server.update_map(0, local_map, pose)
        global_map = server.get_global_map()
        
        # Check Global (1, 0) -> Occupied
        # Row = 50 - 1 = 49.
        # Col = 0 + 50 = 50.
        # Should be > 0.5
        self.assertGreater(global_map[49, 50], 0.5)
        
        # Check Global (0, 0) -> Free
        # Row = 50. Col = 50.
        # Should be < 0.5
        self.assertLess(global_map[50, 50], 0.5)
        
    def test_update_trajectory(self):
        server = FusionServer()
        server.update_trajectory(0, (10.0, 20.0, 0.0))
        server.update_trajectory(0, (11.0, 21.0, 0.0))
        
        traj = server.trajectories[0]
        self.assertEqual(len(traj), 2)
        self.assertEqual(traj[0], (10.0, 20.0))
        self.assertEqual(traj[1], (11.0, 21.0))
        
        print("TestFusionServer Trajectory Passed")
        
        print("TestFusionServer Passed")

if __name__ == '__main__':
    unittest.main()
