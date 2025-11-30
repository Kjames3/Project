# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import random
import numpy as np
import carla
import math
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import RoadOption
from agents.navigation.behavior_types import Cautious, Aggressive, Normal

from agents.tools.misc import get_speed, positive, is_within_distance, compute_distance

from odometry import Odometry  # pylint: disable=import-rror
import pdb

from utils.ate import AbsoluteTrajectoryError
from mapping.occupancy_grid import LocalMapper # Import LocalMapper
from hybrid_planner import HybridRoutePlanner # Import HybridRoutePlanner

class BehaviorAgent(BasicAgent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment such as tailgating avoidance.
    Adding to these are possible behaviors, the agent can also keep safety distance
    from a car in front of it by tracking the instantaneous time to collision
    and keeping it in a certain range. Finally, different sets of behaviors
    are encoded in the agent, from cautious to a more aggressive ones.
    """

    def __init__(self, vehicle, target_speed=20, behavior='normal', opt_dict={}, map_inst=None, grp_inst=None, fusion_server=None):
        """
        Initialization the agent paramters, the local and the global planner.

            :param vehicle: actor to apply to agent logic onto
            :param target_speed: speed (in Km/h) at which the vehicle will move
            :param opt_dict: dictionary in case some of its parameters want to be changed.
                This also applies to parameters related to the LocalPlanner.
            :param map_inst: carla.Map instance to avoid the expensive call of getting it.
            :param grp_inst: GlobalRoutePlanner instance to avoid the expensive call of getting it.
            :param fusion_server: FusionServer instance for cooperative planning.

        """

        # PID Controller Tuning
        if 'lateral_control_dict' not in opt_dict:
            opt_dict['lateral_control_dict'] = {
                'K_P': 0.6,
                'K_D': 0.05,
                'K_I': 0.05,
                'dt': 0.05
            }
        if 'longitudinal_control_dict' not in opt_dict:
            opt_dict['longitudinal_control_dict'] = {
                'K_P': 1.0,
                'K_D': 0.0,
                'K_I': 0.05,
                'dt': 0.05
            }
        
        # Ensure traffic rules are respected
        opt_dict['ignore_traffic_lights'] = False
        opt_dict['ignore_stop_signs'] = False
        opt_dict['ignore_vehicles'] = False

        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self._look_ahead_steps = 0

        # Vehicle information
        self._speed = 0
        self._speed_limit = 0
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._behavior = None
        self._sampling_resolution = 4.5

        self.bound_x = vehicle.bounding_box.extent.x
        self.bound_y = vehicle.bounding_box.extent.y
        self.bound_z = vehicle.bounding_box.extent.z

        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()

        # Initalize Odometry
        self._odometry = Odometry()

        # Latest pose
        self.latest_pose = None

        # Keep track of prev sensor data
        self.prev_sensor_data = None

        # Initialize Local Mapper
        self._local_mapper = LocalMapper(vehicle)
        
        # Initialize Hybrid Planner
        self._hybrid_planner = None
        if fusion_server:
            self._hybrid_planner = HybridRoutePlanner(fusion_server)
            self._check_interval = 20 # Check every 20 steps
            self._step_count = 0
            
        # Stuck Detection
        self._stuck_timer = 0
        self._last_location = None
        self._stuck_threshold = 100 # steps (approx 5s)

    def destroy(self, gt_traj, est_traj):
        ate = AbsoluteTrajectoryError(gt_traj, est_traj)
        ate.traj_err = ate.compute_trajectory_error()
        stats = ate.get_statistics()

        # Print statistics
        print("\nTrajectory Error Statistics:")
        print(f"RMSE Position Error: {stats['rmse_position']:.3f} meters")
        print(f"Mean Position Error: {stats['mean_position']:.3f} meters")
        print(f"Median Position Error: {stats['median_position']:.3f} meters")
        print(f"Std Position Error: {stats['std_position']:.3f} meters")
        print(f"\nRMSE Rotation Error: {stats['rmse_rotation']:.3f} radians")
        print(f"Mean Rotation Error: {stats['mean_rotation']:.3f} radians")
        print(f"Median Rotation Error: {stats['median_rotation']:.3f} radians")
        print(f"Std Rotation Error: {stats['std_rotation']:.3f} radians")

        ate.plot_traj()
    
    def sensors(self):  # pylint: disable=no-self-use
        sensors = self._odometry.sensors()
        
        # Add Semantic LiDAR for Mapping
        sensors.append({
            'type': 'sensor.lidar.ray_cast_semantic', 
            'x': 0.7, 'y': 0.0, 'z': 1.60, 
            'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
            'range': 50, 
            'rotation_frequency': 20, 
            'channels': 32, 
            'upper_fov': 10, 
            'lower_fov': -30, 
            'points_per_second': 56000,
            'id': 'LIDAR'
        })

        for s in sensors:
            s['x'] = s['x']*self.bound_x
            s['y'] = s['y']*self.bound_y
            s['z'] = s['z']*self.bound_z
        return sensors

    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(self._speed_limit)
        self._direction = self._local_planner.target_road_option
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = max(5, int((self._speed_limit) / 10))

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(lights_list)

        return affected

    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """

        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        behind_vehicle_state, behind_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max(
            self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, low_angle_th=160)
        if behind_vehicle_state and self._speed < get_speed(behind_vehicle):
            if (right_turn == carla.LaneChange.Right or right_turn ==
                    carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the right!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         right_wpt.transform.location)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the left!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         left_wpt.transform.location)

    def collision_and_car_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        and managing possible tailgating chances.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """

        vehicle_list = self._world.get_actors().filter("*vehicle*")
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id]

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=30)

            # Check for tailgating
            if not vehicle_state and self._direction == RoadOption.LANEFOLLOW \
                    and not waypoint.is_junction and self._speed > 10 \
                    and self._behavior.tailgate_counter == 0:
                self._tailgating(waypoint, vehicle_list)

        return vehicle_state, vehicle, distance

    def pedestrian_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """

        walker_list = self._world.get_actors().filter("*walker.pedestrian*")
        def dist(w): return w.get_location().distance(waypoint.transform.location)
        walker_list = [w for w in walker_list if dist(w) < 10]

        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)

        return walker_state, walker, distance

    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """

        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        # Under safety time distance, slow down.
        if self._behavior.safety_time > ttc > 0.0:
            target_speed = min([
                positive(vehicle_speed - self._behavior.speed_decrease),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            target_speed = min([
                max(self._min_speed, vehicle_speed),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Normal behavior.
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

    def run_step(self, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        # SLAM (Modify as needed)
        self._update_stuck_status()
        sensor_data = self.get_sensor_data()

        # Update latest estimated pose
        self.latest_pose = self._odometry.get_pose(sensor_data, self.prev_sensor_data)

        # Update prev sensor data
        self.prev_sensor_data = sensor_data

        # Mapping Update
        if 'LIDAR' in sensor_data:
            lidar_data = sensor_data['LIDAR'][1] # [timestamp, data]
            self._local_mapper.process_lidar(lidar_data, self._vehicle.get_transform())

        # Cooperative Replanning Check
        if self._hybrid_planner:
            self._step_count += 1
            if self._step_count % self._check_interval == 0:
                # Check path ahead
                # Get next few waypoints
                plan = self._local_planner.get_plan()
                if plan:
                    # Check up to 20 meters ahead
                    start_loc = self._vehicle.get_location()
                    # Find a waypoint ~20m ahead or last one
                    target_loc = plan[min(len(plan)-1, 10)][0].transform.location
                    
                    if self._hybrid_planner.is_path_blocked(start_loc, target_loc):
                        print(f"Agent {self._vehicle.id}: Path blocked! Stopping/Rerouting...")
                        # Simple behavior: Emergency Stop for now
                        # In real implementation, we would call set_destination(new_dest)
                        hazard_detected = True

        # Behavior Agent Updates (Do not modify):
        
        self._update_information()

        control = None
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        # 1: Red lights and stops behavior
        if self.traffic_light_manager():
            return self.emergency_stop()

        # 2.1: Pedestrian avoidance behaviors
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)

        if walker_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = w_distance - max(
                walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance:
                return self.emergency_stop()

        # 2.2: Car following behaviors
        vehicle_state, vehicle, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)

        if vehicle_state:
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = distance - max(
                vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance:
                return self.emergency_stop()
            else:
                control = self.car_following_manager(vehicle, distance)

        # 3: Intersection behavior
        elif self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]):
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - 5])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # 4: Normal behavior
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control
        
    def is_stuck(self):
        """
        Checks if the agent is stuck.
        :return: True if stuck, False otherwise
        """
        return self._stuck_timer > self._stuck_threshold

    def _update_stuck_status(self):
        """
        Updates the stuck timer based on movement.
        """
        current_loc = self._vehicle.get_location()
        if self._last_location:
            dist = current_loc.distance(self._last_location)
            if dist < 0.1: # Not moving much
                self._stuck_timer += 1
            else:
                self._stuck_timer = 0
        self._last_location = current_loc

    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

    def reroute_to_frontier(self):
        """
        Reroutes the agent to the nearest frontier point.
        :return: True if successful, False otherwise
        """
        if not self._hybrid_planner:
            return False
            
        frontiers = self._hybrid_planner.get_frontiers()
        if not frontiers:
            return False
            
        # Find nearest frontier
        ego_loc = self._vehicle.get_location()
        min_dist = float('inf')
        best_frontier = None
        
        for fx, fy in frontiers:
            dist = math.sqrt((ego_loc.x - fx)**2 + (ego_loc.y - fy)**2)
            
            # Filter frontiers too close (prevent loops)
            if dist < 5.0:
                continue
                
            if dist < min_dist:
                min_dist = dist
                best_frontier = (fx, fy)
                
        if best_frontier:
            # Convert to Location
            # Use ego Z for simplicity, or 0
            target_loc = carla.Location(x=best_frontier[0], y=best_frontier[1], z=ego_loc.z)
            
            print(f"Agent {self._vehicle.id}: Rerouting to frontier at ({best_frontier[0]:.1f}, {best_frontier[1]:.1f})")
            
            # Snap to Road Network
            # Find nearest waypoint on the road
            try:
                wp = self._map.get_waypoint(target_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                if wp:
                    target_loc = wp.transform.location
                    print(f"Agent {self._vehicle.id}: Snapped to road at ({target_loc.x:.1f}, {target_loc.y:.1f})")
            except Exception as e:
                print(f"Agent {self._vehicle.id}: Failed to snap to road: {e}")
            
            self.set_destination(target_loc)
            return True
            
        return False
