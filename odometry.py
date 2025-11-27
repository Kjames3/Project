import numpy as np
import cv2
import math
import traceback

class Odometry:
    def __init__(self):
        # Camera Matrix
        width, height, fov = 1280, 720, 100.0
        f = width / (2.0 * math.tan(math.radians(fov) / 2.0))
        cx = width / 2.0
        cy = height / 2.0
        self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)

        # Feature Extractor (3000 features)
        self.orb = cv2.ORB_create(nfeatures=3000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Global Pose
        self.pose = np.eye(4) 
        self.R_co = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
        self.frame_count = 0

        print("[ODOMETRY] Ready.")

    def sensors(self):
        return [
            {'type': 'sensor.camera.rgb', 'x': 2.0, 'y': -0.8, 'z': 1.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -10.0,
             'width': 1280, 'height': 720, 'fov': 100, 'id': 'Left'},
            {'type': 'sensor.camera.depth', 'x': 2.0, 'y': -0.8, 'z': 1.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -10.0,
             'width': 1280, 'height': 720, 'fov': 100, 'id': 'Depth_Left'},
            {'type': 'sensor.lidar.ray_cast', 'x': 0.0, 'y': 0.0, 'z': 2.4, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'range': 50, 'channels': 32, 'points_per_second': 56000, 'upper_fov': 10.0, 'lower_fov': -30.0, 'id': 'Lidar'}
        ]

    def process_depth(self, image_data):
        if image_data is None: return None
        try:
            if isinstance(image_data, np.ndarray):
                array = image_data.astype(np.float32)
            elif hasattr(image_data, 'raw_data'):
                buffer = np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(buffer, (image_data.height, image_data.width, 4)).astype(np.float32)
            else: return None
            
            B, G, R = array[:, :, 0], array[:, :, 1], array[:, :, 2]
            normalized = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0**3 - 1)
            return normalized * 1000.0
        except Exception: return None

    def process_rgb(self, image_data):
        if image_data is None: return None
        try:
            if isinstance(image_data, np.ndarray):
                if image_data.shape[-1] == 4: return image_data[:, :, :3]
                return image_data
            elif hasattr(image_data, 'raw_data'):
                array = np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (image_data.height, image_data.width, 4))
                return array[:, :, :3]
        except Exception: return None
        return None

    def _get_pose_from_matrix(self):
        x, y, z = self.pose[:3, 3]
        sy = math.sqrt(self.pose[0, 0]**2 + self.pose[1, 0]**2)
        if sy < 1e-6:
            roll, pitch, yaw = math.atan2(-self.pose[1, 2], self.pose[1, 1]), math.atan2(-self.pose[2, 0], sy), 0
        else:
            roll, pitch, yaw = math.atan2(self.pose[2, 1], self.pose[2, 2]), math.atan2(-self.pose[2, 0], sy), math.atan2(self.pose[1, 0], self.pose[0, 0])
        return [x, y, z, roll, pitch, yaw]

    def get_pose(self, sensor_data, prev_sensor_data):
        self.frame_count += 1
        if prev_sensor_data is None or sensor_data is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        if 'Left' not in sensor_data or 'Depth_Left' not in prev_sensor_data:
            return self._get_pose_from_matrix()

        try:
            curr_img = self.process_rgb(sensor_data['Left'][1])
            prev_img = self.process_rgb(prev_sensor_data['Left'][1])
            prev_depth = self.process_depth(prev_sensor_data['Depth_Left'][1])
            
            if curr_img is None or prev_img is None or prev_depth is None:
                return self._get_pose_from_matrix()

            # Detect & Match
            kp1, des1 = self.orb.detectAndCompute(cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY), None)
            kp2, des2 = self.orb.detectAndCompute(cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY), None)

            if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
                return self._get_pose_from_matrix()

            matches = self.bf.knnMatch(des1, des2, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.70 * n.distance:
                    good_matches.append(m)
            
            # --- FIX: Do NOT cap at 100 matches. Use all good data. ---
            if len(good_matches) < 10:
                return self._get_pose_from_matrix()

            object_points, image_points = [], []
            h, w = prev_depth.shape
            
            for m in good_matches:
                u1, v1 = kp1[m.queryIdx].pt 
                u2, v2 = kp2[m.trainIdx].pt 
                
                if 0 <= int(v1) < h and 0 <= int(u1) < w:
                    z = prev_depth[int(v1), int(u1)]
                    if 1.0 < z < 50.0: # Keep tight depth range
                        x = (u1 - self.K[0, 2]) * z / self.K[0, 0]
                        y = (v1 - self.K[1, 2]) * z / self.K[1, 1]
                        object_points.append([x, y, z])
                        image_points.append([u2, v2])

            if len(object_points) < 6:
                return self._get_pose_from_matrix()

            # Use EPNP for better stability
            success, rvec, tvec, _ = cv2.solvePnPRansac(
                np.array(object_points), np.array(image_points), 
                self.K, None, flags=cv2.SOLVEPNP_EPNP,
                confidence=0.999, reprojectionError=3.0,
                iterationsCount=100
            )

            if not success:
                return self._get_pose_from_matrix()

            rmat, _ = cv2.Rodrigues(rvec)
            T_pnp = np.eye(4)
            T_pnp[:3, :3] = rmat
            T_pnp[:3, 3] = tvec.squeeze()
            
            try:
                T_motion_opt = np.linalg.inv(T_pnp)
            except np.linalg.LinAlgError:
                return self._get_pose_from_matrix()

            R_opt = T_motion_opt[:3, :3]
            t_opt = T_motion_opt[:3, 3]
            
            R_carla = self.R_co @ R_opt @ self.R_co.T
            t_carla = self.R_co @ t_opt

            T_motion_carla = np.eye(4)
            T_motion_carla[:3, :3] = R_carla
            T_motion_carla[:3, 3] = t_carla

            self.pose = self.pose @ T_motion_carla
            
            if self.frame_count % 50 == 0:
                print(f"[ODOMETRY] Frame {self.frame_count}: Pos ({self.pose[0,3]:.2f}, {self.pose[1,3]:.2f})")

            return self._get_pose_from_matrix()

        except Exception:
            traceback.print_exc()
            return self._get_pose_from_matrix()