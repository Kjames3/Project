
import unittest
import numpy as np

# Mocking the CameraManager behavior
class MockCameraManager:
    def __init__(self, width, height):
        self.hud = type('obj', (object,), {'dim': (width, height)})()
        self.surface = None

    def project_to_lidar_pygame(self, points):
        """Transform lidar points from LiDAR 3D coordinates to pygame BEV 2D plane
            lidar_data: lidar points (N,x) in LiDAR 3D coordinates
        """
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(self.hud.dim) / 100.0
        lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
        # lidar_data = np.fabs(lidar_data)  # REMOVED
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))

        # Added CLIPPING logic
        lidar_data_t = lidar_data.T
        lidar_data_t[0] = np.clip(lidar_data_t[0], 0, self.hud.dim[0] - 1)
        lidar_data_t[1] = np.clip(lidar_data_t[1], 0, self.hud.dim[1] - 1)
        lidar_data = lidar_data_t.T

        return lidar_data

    def _parse_image(self, points):
        lidar_data = self.project_to_lidar_pygame(points)
        lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
        lidar_img = np.zeros(lidar_img_size)
        # Should NOT crash now
        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
        return lidar_img

class TestLidarFix(unittest.TestCase):
    def test_lidar_clipping(self):
        # Setup: 100x100 screen
        width, height = 100, 100
        manager = MockCameraManager(width, height)

        # Point far away: x=200m -> 250. Should be clipped to 99.
        points_crash = np.array([[200.0, 0.0, 0.0, 0.0]])

        lidar_data = manager.project_to_lidar_pygame(points_crash)
        self.assertEqual(lidar_data[0][0], 99)

        # Ensure it doesn't crash
        try:
            manager._parse_image(points_crash)
        except IndexError:
            self.fail("Fix failed: Still crashed on positive overflow")

    def test_lidar_negative_handling(self):
        # Setup: 100x100 screen
        width, height = 100, 100
        manager = MockCameraManager(width, height)

        # x = -200m -> -150. Should be clipped to 0.
        # PREVIOUSLY: fabs(-150) = 150 -> Crash.
        points_neg = np.array([[-200.0, 0.0, 0.0, 0.0]])

        lidar_data = manager.project_to_lidar_pygame(points_neg)
        self.assertEqual(lidar_data[0][0], 0)

        try:
            manager._parse_image(points_neg)
        except IndexError:
            self.fail("Fix failed: Still crashed on negative overflow")

    def test_visual_mirroring_fix(self):
        # Setup: 100x100 screen
        width, height = 100, 100
        manager = MockCameraManager(width, height)

        # x = -60m -> -10.
        # PREVIOUSLY: fabs(-10) = 10. (Visual Bug)
        # NOW: clipped to 0.
        # This is acceptable behavior for a 2D projection (clamping to edge).
        # Alternatively, we could filter them out completely, but clipping is safer for array indexing
        # and simple to implement in numpy.
        # The key is that it's not +10 (mirroring).

        points_mirror = np.array([[-60.0, 0.0, 0.0, 0.0]])

        lidar_data = manager.project_to_lidar_pygame(points_mirror)
        mapped_x = lidar_data[0][0]

        print(f"Mapped X for input -60 (expected 0, got {mapped_x})")
        self.assertEqual(mapped_x, 0, "Visual mirroring logic persists or clipping wrong")
        self.assertNotEqual(mapped_x, 10, "Still mirroring!")

if __name__ == '__main__':
    unittest.main()
