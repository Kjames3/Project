import unittest
from unittest.mock import MagicMock
import sys

# Mock carla module before importing BehaviorAgent
mock_carla = MagicMock()
class MockMap:
    def __init__(self):
        self.get_waypoint = MagicMock()
    def get_topology(self):
        return []
class MockLocation:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    def distance(self, other):
        import math
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
mock_carla.Map = MockMap
mock_carla.Location = MockLocation
sys.modules['carla'] = mock_carla

# Mock shapely
mock_shapely = MagicMock()
sys.modules['shapely'] = mock_shapely
sys.modules['shapely.geometry'] = mock_shapely.geometry

# Mock networkx
mock_nx = MagicMock()
sys.modules['networkx'] = mock_nx

# Mock matplotlib
mock_plt = MagicMock()
sys.modules['matplotlib'] = mock_plt
sys.modules['matplotlib.pyplot'] = mock_plt

from agents.navigation.behavior_agent import BehaviorAgent

class TestNavigation(unittest.TestCase):
    def test_reroute_to_frontier_snapping(self):
        # Mock Vehicle and World
        vehicle = MagicMock()
        vehicle.get_location.return_value = mock_carla.Location(0, 0, 0)
        vehicle.bounding_box.extent.x = 1.0
        vehicle.bounding_box.extent.y = 1.0
        vehicle.bounding_box.extent.z = 1.0
        
        # Mock Map
        carla_map = mock_carla.Map()
        
        # Mock Waypoint
        mock_wp = MagicMock()
        mock_wp.transform.location = mock_carla.Location(10.5, 20.5, 0) # Snapped location
        
        # Setup get_waypoint behavior
        carla_map.get_waypoint.return_value = mock_wp
        
        # Mock Planner
        planner = MagicMock()
        # Frontier at (10, 20)
        planner.get_frontiers.return_value = [(10.0, 20.0)]
        
        # Initialize Agent
        agent = BehaviorAgent(vehicle, map_inst=carla_map, fusion_server=MagicMock())
        agent._hybrid_planner = planner
        
        # Override set_destination to check result
        agent.set_destination = MagicMock()
        
        # Run reroute
        success = agent.reroute_to_frontier()
        
        self.assertTrue(success)
        
        # Check if get_waypoint was called
        carla_map.get_waypoint.assert_called()
        
        # Check if set_destination was called with the SNAPPED location
        args, _ = agent.set_destination.call_args
        target_loc = args[0]
        self.assertEqual(target_loc.x, 10.5)
        self.assertEqual(target_loc.y, 20.5)
        
        print("TestNavigation Snapping Passed")

    def test_stuck_detection(self):
        # Mock Vehicle
        vehicle = MagicMock()
        vehicle.get_location.return_value = mock_carla.Location(0, 0, 0)
        
        # Initialize Agent
        agent = BehaviorAgent(vehicle, map_inst=mock_carla.Map(), fusion_server=MagicMock())
        
        # Simulate stuck
        agent._last_location = mock_carla.Location(0, 0, 0)
        agent._stuck_timer = 0
        
        # Update stuck status (no movement)
        agent._update_stuck_status()
        self.assertEqual(agent._stuck_timer, 1)
        
        # Update again
        agent._update_stuck_status()
        self.assertEqual(agent._stuck_timer, 2)
        
        # Simulate movement
        vehicle.get_location.return_value = mock_carla.Location(10, 0, 0)
        agent._update_stuck_status()
        self.assertEqual(agent._stuck_timer, 0)
        
        # Simulate stuck threshold
        agent._stuck_timer = 101
        self.assertTrue(agent.is_stuck())
        
        print("TestNavigation Stuck Detection Passed")

if __name__ == '__main__':
    unittest.main()
