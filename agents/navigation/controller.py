import collections
from collections import deque
import math
import numpy as np
import carla
from agents.tools.misc import get_speed

class VehiclePIDController():
    """
    Hybrid Controller: Uses PID for Speed, and a mix of Stanley/PurePursuit for Steering.
    """
    def __init__(self, vehicle, args_lateral, args_longitudinal, offset=0, max_throttle=0.75, max_brake=0.3, max_steering=0.8):
        self.max_brake = max_brake
        self.max_throt = max_throttle
        self.max_steer = max_steering
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        
        # Speed Control (PID is fine for this)
        self._lon_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        
        # Steering Control (Hybrid)
        self._lat_controller = HybridLateralController(self._vehicle, offset=offset)

    def run_step(self, target_speed, waypoints, dt=None):
        # 1. Longitudinal Control (Speed)
        acceleration = self._lon_controller.run_step(target_speed)
        
        # 2. Lateral Control (Steering) - Pass the full waypoint list
        current_steering = self._lat_controller.run_step(waypoints)

        # 3. Apply Control Limits
        control = carla.VehicleControl()
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        # Clamp Steering
        control.steer = max(min(current_steering, self.max_steer), -self.max_steer)
        
        control.hand_brake = False
        control.manual_gear_shift = False

        return control

class PIDLongitudinalController():
    """Standard PID for Speed Control"""
    def __init__(self, vehicle, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_speed):
        current_speed = get_speed(self._vehicle)
        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

class HybridLateralController():
    """
    Hybrid Controller that switches between Stanley and Pure Pursuit.
    - Pure Pursuit: Used in Intersections (Stability)
    - Stanley: Used on Straight/Curves (Accuracy)
    """
    def __init__(self, vehicle, offset=0, stanley_k=0.15, pp_kdd=1.0, wheelbase=2.875):
        self._vehicle = vehicle
        self._offset = offset
        self._k = stanley_k      # Stanley Gain
        self._kdd = pp_kdd       # Pure Pursuit Lookahead Gain
        self._L = wheelbase

    def run_step(self, waypoints):
        if not waypoints: return 0.0
        
        # 1. Check if we are in a Junction
        # We use the first few waypoints to detect if we are turning at an intersection
        is_junction = waypoints[0][0].is_junction

        # 2. Select Strategy
        if is_junction:
            # CRITICAL FIX: Use Pure Pursuit in junctions to prevent looping
            return self._pure_pursuit_control(waypoints)
        else:
            # Use Stanley for lane keeping accuracy on roads
            return self._stanley_control(waypoints)

    def _pure_pursuit_control(self, waypoints):
        # Get Vehicle State
        t = self._vehicle.get_transform()
        loc = t.location
        vel = self._vehicle.get_velocity()
        speed = np.sqrt(vel.x**2 + vel.y**2)
        
        # Lookahead distance
        # Tuned for City Driving (Low Speeds)
        # 3.0m min for tight corners, 15.0m max
        # Gain 0.5 * speed approx 
        ld = np.clip(self._kdd * speed, 3.0, 15.0)

        # Find target
        target_wp = None
        for wp_tuple in waypoints:
            wp = wp_tuple[0]
            if loc.distance(wp.transform.location) >= ld:
                target_wp = wp
                break
        
        if not target_wp: target_wp = waypoints[-1][0]

        # Math
        target_loc = target_wp.transform.location
        yaw = np.radians(t.rotation.yaw)
        alpha = math.atan2(target_loc.y - loc.y, target_loc.x - loc.x) - yaw
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi # Normalize
        
        delta = math.atan2(2.0 * self._L * np.sin(alpha), ld)
        return np.clip(delta, -1.0, 1.0)

    def _stanley_control(self, waypoints):
        t = self._vehicle.get_transform()
        v_loc = t.location
        
        # CRITICAL FIX: Calculate from Front Axle
        yaw_rad = np.radians(t.rotation.yaw)
        axle_x = v_loc.x + self._L * math.cos(yaw_rad)
        axle_y = v_loc.y + self._L * math.sin(yaw_rad)
        
        vel = self._vehicle.get_velocity()
        speed = np.sqrt(vel.x**2 + vel.y**2)

        # Find nearest waypoint (Track Error)
        min_dist = float('inf')
        nearest_wp = None
        
        # Only check first few points to prevent latching onto wrong road parts
        import itertools
        for wp_tuple in itertools.islice(waypoints, 5): 
            wp = wp_tuple[0]
            # Dist from front axle
            dist = math.sqrt((wp.transform.location.x - axle_x)**2 + (wp.transform.location.y - axle_y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_wp = wp

        if not nearest_wp: return 0.0

        # Heading Error
        road_yaw = np.radians(nearest_wp.transform.rotation.yaw)
        heading_error = road_yaw - yaw_rad
        
        # CRITICAL FIX: Normalize Heading Error to [-pi, pi]
        # This prevents the "spinning" when crossing 0/360 degrees
        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

        # Cross Track Error (CTE)
        # Vector from Axle to Waypoint
        dx = nearest_wp.transform.location.x - axle_x
        dy = nearest_wp.transform.location.y - axle_y
        
        # Project onto path normal to find sign (Left/Right)
        path_x = math.cos(road_yaw)
        path_y = math.sin(road_yaw)
        cross_product = dx * path_y - dy * path_x
        
        # FIX: Cross Product (z-component) of (D x P).
        # D=(dx, dy), P=(px, py).
        # z = dx*py - dy*px.
        # If D is LEFT of P, z > 0.
        # If D is Left of P, we want to steer RIGHT.
        # Stanley: delta = ... + atan2(k*e, ...).
        # We want delta > 0. So e must be > 0.
        # So if Left (z > 0), e > 0.
        # Wait, earlier trace said dy=10 (Right). D=(0,10). P=(1,0).
        # z = 0*0 - 10*1 = -10. (Negative).
        # We want steer Right (Positive).
        # So if z < 0 -> e > 0?
        # My code said: cte = min_dist if cross_product > 0 else -min_dist.
        # If z < 0 (-10), cte = -min_dist (-10).
        # Steer = atan(-10) -> Left. WRONG.
        # So we want: if z < 0 -> cte > 0.
        
        cte = -min_dist if cross_product > 0 else min_dist

        # Stanley Law
        # Steer = Heading Error + atan( k * cte / (speed + epsilon) )
        steer_angle = heading_error + math.atan2(self._k * cte, speed + 5.0)
        
        return np.clip(steer_angle, -1.0, 1.0)