import unittest
import numpy as np
from mapping.occupancy_grid import LocalMapper

# Mocks for CARLA classes
class MockLocation:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class MockRotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

class MockTransform:
    def __init__(self, location=None, rotation=None):
        self.location = location if location else MockLocation()
        self.rotation = rotation if rotation else MockRotation()

    def get_matrix(self):
        # Simple 2D matrix for X, Y, Yaw (ignoring Z, Pitch, Roll for simplicity or full 4x4)
        # CARLA uses 4x4.
        # Let's implement a basic 4x4 builder if needed, but for now
        # the fix might assume we calculate it ourselves or use get_matrix.
        # If the code uses get_matrix, I need to implement it.
        # If the code manually calculates from location/rotation, I don't need this.
        # I'll implement a basic one to be safe.

        cy = np.cos(np.radians(self.rotation.yaw))
        sy = np.sin(np.radians(self.rotation.yaw))
        cr = np.cos(np.radians(self.rotation.roll))
        sr = np.sin(np.radians(self.rotation.roll))
        cp = np.cos(np.radians(self.rotation.pitch))
        sp = np.sin(np.radians(self.rotation.pitch))

        matrix = np.identity(4)
        matrix[0, 3] = self.location.x
        matrix[1, 3] = self.location.y
        matrix[2, 3] = self.location.z

        # Rotation (Yaw * Pitch * Roll) - Sequence depends on convention.
        # CARLA is usually Roll->Pitch->Yaw (Intrinsic)?
        # For this test, we only care about Translation (Offset).
        # So Rotation is Identity.
        return matrix

class MockLidarData:
    def __init__(self, data, transform):
        self.raw_data = data.tobytes()
        self.transform = transform

class MockVehicle:
    pass

class TestOffsetMapping(unittest.TestCase):
    def test_sensor_offset(self):
        vehicle = MockVehicle()
        mapper = LocalMapper(vehicle, grid_size=0.1, window_size=20.0)
        # Grid Dim = 200. Center = 100.
        # Res = 0.1

        # 1. Setup Transforms
        # Vehicle at (0, 0, 0)
        veh_transform = MockTransform(MockLocation(0,0,0), MockRotation(0,0,0))

        # Sensor at (0.7, 0, 0) relative to Vehicle
        # Global Transform of Sensor is same as relative if Vehicle is at 0,0.
        sensor_transform = MockTransform(MockLocation(0.7, 0.0, 0.0), MockRotation(0,0,0))

        # 2. Setup Lidar Point
        # Point is at x=5.0 in Sensor Frame.
        # So it is at x=5.7 in Vehicle Frame.
        data = np.array([
            (5.0, 0.0, 0.0, 0.0, 0, 3) # Tag 3 = Building (Static Obstacle)
        ], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('cos', 'f4'), ('idx', 'u4'), ('tag', 'u4')])

        lidar_data = MockLidarData(data, sensor_transform)

        # 3. Process
        mapper.process_lidar(lidar_data, veh_transform)
        local_map = mapper.get_local_map()

        # 4. Verification
        # Center index is 100.
        # We expect the obstacle at 5.7m.
        # Grid index = Center - (X / res) = 100 - (5.7 / 0.1) = 100 - 57 = 43.
        # Currently (with bug), it will be at 5.0m.
        # Grid index = 100 - (5.0 / 0.1) = 100 - 50 = 50.

        # Check if index 50 is occupied (Bug behavior)
        if local_map[50, 100] == 1.0:
            print("Bug Confirmed: Obstacle mapped at 5.0m (Sensor Frame) instead of 5.7m (Vehicle Frame)")

        # Check if index 43 is occupied (Correct behavior)
        if local_map[43, 100] == 1.0:
            print("Fix Verified: Obstacle mapped at 5.7m")

        # Assertion for the Correct Behavior (Testing for failure first)
        self.assertEqual(local_map[43, 100], 1.0, "Obstacle should be at 5.7m (Index 43)")

if __name__ == '__main__':
    unittest.main()
