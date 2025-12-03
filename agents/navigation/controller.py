# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains PID controllers to perform lateral and longitudinal control. """

from collections import deque
import math
import numpy as np
import carla
from agents.tools.misc import get_speed


class VehiclePIDController():
    """
    VehiclePIDController is the combination of two PID controllers
    (lateral and longitudinal) to perform the
    low level control a vehicle from client side
    """


    def __init__(self, vehicle, args_lateral, args_longitudinal, offset=0, max_throttle=0.75, max_brake=0.3,
                 max_steering=0.8):
        """
        Constructor method.

        :param vehicle: actor to apply to local planner logic onto
        :param args_lateral: dictionary of arguments to set the lateral PID controller
        using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        :param args_longitudinal: dictionary of arguments to set the longitudinal
        PID controller using the following semantics:
            K_P -- Proportional term
            K_D -- Differential term
            K_I -- Integral term
        :param offset: If different than zero, the vehicle will drive displaced from the center line.
        Positive values imply a right offset while negative ones mean a left one. Numbers high enough
        to cause the vehicle to drive through other lanes might break the controller.
        """

        self.max_brake = max_brake
        self.max_throt = max_throttle
        self.max_steer = max_steering

        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self.past_steering = self._vehicle.get_control().steer
        self._lon_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        self._lon_controller = PIDLongitudinalController(self._vehicle, **args_longitudinal)
        
        # Always use Pure Pursuit
        self._lat_controller = PurePursuitLateralController(
            self._vehicle,
            L=args_lateral.get('L', 2.875),
            Kdd=args_lateral.get('Kdd', 4.0)
        )

    def run_step(self, target_speed, waypoints, dt=None):
        """
        Execute one step of control invoking both lateral and longitudinal
        PID controllers to reach a target waypoint
        at a given target_speed.

            :param target_speed: desired vehicle speed
            :param waypoints: list of (carla.Waypoint, RoadOption) from the LocalPlanner queue
            :param dt: time differential in seconds
            :return: distance (in meters) to the waypoint
        """
        if dt is not None:
            self._lon_controller._dt = dt
            # Only set dt if using PID (Pure Pursuit doesn't use dt directly)
            if hasattr(self._lat_controller, '_dt'):
                self._lat_controller._dt = dt

        acceleration = self._lon_controller.run_step(target_speed)
        
        # Pure Pursuit expects list of waypoints
        current_steering = self._lat_controller.run_step(waypoints)
        control = carla.VehicleControl()
        if acceleration >= 0.0:
            control.throttle = min(acceleration, self.max_throt)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)

        # Steering regulation: changes cannot happen abruptly, can't steer too much.

        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1

        if current_steering >= 0:
            steering = min(self.max_steer, current_steering)
        else:
            steering = max(-self.max_steer, current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering

        return control


    def change_longitudinal_PID(self, args_longitudinal):
        """Changes the parameters of the PIDLongitudinalController"""
        self._lon_controller.change_parameters(**args_longitudinal)

    def change_lateral_PID(self, args_lateral):
        """Changes the parameters of the PIDLateralController"""
        self._lat_controller.change_parameters(**args_lateral)

    def set_offset(self, offset):
        """Changes the offset"""
        self._lat_controller.set_offset(offset)


class PIDLongitudinalController():
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.

            :param target_speed: target speed in Km/h
            :param debug: boolean for debugging
            :return: throttle control
        """
        current_speed = get_speed(self._vehicle)

        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle/brake of the vehicle based on the PID equations

            :param target_speed:  target speed in Km/h
            :param current_speed: current speed of the vehicle in Km/h
            :return: throttle/brake control
        """

        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt




class PurePursuitLateralController:
    """
    PurePursuitLateralController implements lateral control using the Pure Pursuit algorithm.
    """
    def __init__(self, vehicle, L=2.875, Kdd=4.0):
        self._vehicle = vehicle
        self.L = L      # Wheelbase length
        self.Kdd = Kdd  # Lookahead gain
        self.args_lateral_dict = {}

    def run_step(self, waypoints):
        """
        Execute one step of lateral control.
        :param waypoints: list of (carla.Waypoint, RoadOption) from the LocalPlanner queue
        """
        return self._pure_pursuit_control(waypoints)

    def _pure_pursuit_control(self, waypoints):
        # 1. Get Vehicle State
        vehicle_transform = self._vehicle.get_transform()
        vehicle_loc = vehicle_transform.location
        vehicle_vel = self._vehicle.get_velocity()
        vf = np.sqrt(vehicle_vel.x**2 + vehicle_vel.y**2)

        # 2. Calculate Lookahead Distance (ld)
        # ld = Kdd * velocity, clamped to a reasonable range (e.g., 3m to 20m)
        ld = np.clip(self.Kdd * vf, 3.0, 20.0)

        # 3. Find Target Waypoint
        # Search the queue for the first waypoint that is at least 'ld' distance away
        target_wp = None
        for wp_tuple in waypoints:
            wp = wp_tuple[0] # Extract carla.Waypoint from tuple
            dist = vehicle_loc.distance(wp.transform.location)
            if dist >= ld:
                target_wp = wp
                break
        
        # Fallback: if no point is far enough, take the last one
        if target_wp is None:
            if not waypoints:
                return 0.0 # No path
            target_wp = waypoints[-1][0]

        # 4. Calculate Steering Angle (Pure Pursuit Math)
        target_loc = target_wp.transform.location
        yaw = np.radians(vehicle_transform.rotation.yaw)
        
        # Alpha: Angle between vehicle heading and vector to target
        alpha = math.atan2(target_loc.y - vehicle_loc.y, target_loc.x - vehicle_loc.x) - yaw
        
        # Normalize alpha to range [-pi, pi]
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi

        # Steering calculation: delta = atan2(2 * L * sin(alpha), ld)
        delta = math.atan2(2 * self.L * np.sin(alpha), ld)
        
        # Clamp steering to CARLA limits [-1.0, 1.0]
        delta = np.clip(delta, -1.0, 1.0)

        return delta

    def change_parameters(self, **kwargs):
        """Update parameters at runtime if needed"""
        if 'L' in kwargs: self.L = kwargs['L']
        if 'Kdd' in kwargs: self.Kdd = kwargs['Kdd']
