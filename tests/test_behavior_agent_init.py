import unittest
from unittest.mock import MagicMock
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock carla
sys.modules['carla'] = MagicMock()
import carla
class MockMap:
    pass
carla.Map = MockMap


# Mock shapely
sys.modules['shapely'] = MagicMock()
sys.modules['shapely.geometry'] = MagicMock()

# Mock networkx
sys.modules['networkx'] = MagicMock()

# Mock matplotlib
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()

# Mock numba
sys.modules['numba'] = MagicMock()
# Make sure njit decorator works
sys.modules['numba'].njit = lambda *args, **kwargs: (lambda func: func)





# Mock hybrid_planner
mock_hybrid = MagicMock()
sys.modules['hybrid_planner'] = mock_hybrid
# We need to make sure HybridRoutePlanner is available in hybrid_planner module
# and that it accepts strict arguments to reproduce the issue if we were using real classes,
# but since we are importing the actual BehaviorAgent, we want to see how it calls it.
# However, BehaviorAgent imports HybridRoutePlanner inside __init__ (nested import).
# This makes it tricky to mock the class 'inside' the function unless we patch sys.modules beforehand or use unittest.mock.patch.

from agents.navigation.behavior_agent import BehaviorAgent

class TestBehaviorAgentInit(unittest.TestCase):
    def test_init_fusion_server(self):
        # Setup mocks
        vehicle = MagicMock()
        vehicle.bounding_box.extent.x = 1
        vehicle.bounding_box.extent.y = 1
        vehicle.bounding_box.extent.z = 1
        
        fusion_server = MagicMock()
        
        # We need to actually define a dummy HybridRoutePlanner that enforces arg count
        # to reproduce the TypeError if the code is wrong.
        class MockHybridRoutePlanner:
            def __init__(self, fusion_server):
                self.fusion_server = fusion_server
        
        # Patch the local import in BehaviorAgent
        # Since it does 'from hybrid_planner import HybridRoutePlanner' inside __init__,
        # we can just patch sys.modules['hybrid_planner'].HybridRoutePlanner
        
        sys.modules['hybrid_planner'].HybridRoutePlanner = MockHybridRoutePlanner
        
        print("Attempting to initialize BehaviorAgent with FusionServer...")
        try:
            agent = BehaviorAgent(vehicle, fusion_server=fusion_server)
            print("Successfully initialized BehaviorAgent.")
        except TypeError as e:
            print(f"Caught expected TypeError during init: {e}")
            raise e

if __name__ == '__main__':
    unittest.main()
