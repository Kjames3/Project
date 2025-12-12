#!/usr/bin/env python
import glob
import os
import sys
import math
import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    try:
         sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except IndexError:
        pass

import carla
from agents.navigation.behavior_agent import BehaviorAgent

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 1. Setup - Sync Mode for accuracy
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    try:
        # 2. Spawn Agent
        bp = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        # Pick a spawn point likely to have a turn (or hardcode a known difficult junction location)
        spawn = spawn_points[0] 
        vehicle = world.spawn_actor(bp, spawn)
        
        # 3. Init Agent (Hybrid Controller is inside BehaviorAgent)
        agent = BehaviorAgent(vehicle, behavior='normal')
        
        # Set a destination that forces a turn
        # (Random for now, but in real experiment, hardcode coordinates for repeatability)
        dest = spawn_points[10].location 
        agent.set_destination(dest)
        
        print(f"Testing Stability: Driving from {spawn.location} to {dest}")

        ground_truth_path = []
        actual_trajectory = []

        # 4. Run Test
        for _ in range(400): # Run for 20 seconds (400 ticks * 0.05)
            world.tick()
            
            # Get Control
            control = agent.run_step()
            vehicle.apply_control(control)
            
            # Record Data
            # A. Actual Position (Where the car is)
            v_loc = vehicle.get_location()
            actual_trajectory.append(np.array([v_loc.x, v_loc.y]))
            
            # B. "Ideal" Position (The nearest waypoint on the center of the lane)
            # This represents where the car *should* be.
            wp = world.get_map().get_waypoint(v_loc)
            w_loc = wp.transform.location
            ground_truth_path.append(np.array([w_loc.x, w_loc.y]))

            if agent.done():
                break

        # 5. Calculate RMSE (Cross Track Error approximation)
        error_sum = 0
        for i in range(len(actual_trajectory)):
            act = actual_trajectory[i]
            ideal = ground_truth_path[i]
            dist = np.linalg.norm(act - ideal)
            error_sum += dist**2
        
        rmse = math.sqrt(error_sum / len(actual_trajectory))
        
        print("-" * 30)
        print(f"STABILITY RESULT:")
        print(f"Total Frames: {len(actual_trajectory)}")
        print(f"RMSE (Deviation from Centerline): {rmse:.4f} meters")
        print("-" * 30)

    finally:
        vehicle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)

if __name__ == '__main__':
    main()