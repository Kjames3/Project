import numpy as np
import cv2
import math
import traceback

class Odometry:
    def __init__(self):
        print("[ODOMETRY] Ready (Ground Truth Mode).")

    def sensors(self):
        return [
            {'type': 'sensor.camera.rgb', 'x': 2.0, 'y': -0.8, 'z': 1.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -10.0,
             'width': 1280, 'height': 720, 'fov': 100, 'id': 'Left'},
        ]

    def get_pose(self, sensor_data, prev_sensor_data):
        # We are using Ground Truth from the vehicle directly in the agent 
        # To avoid breaking calls, we are returning a dummy or expected format.
        # The original returned [x, y, z, roll, pitch, yaw].
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]