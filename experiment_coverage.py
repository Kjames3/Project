#!/usr/bin/env python
import glob
import os
import sys
import time
import random
import csv

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    try:
         print(f"Searching: ../../carla/dist/carla-*%d.%d-%s.egg" % (sys.version_info.major, sys.version_info.minor, 'win-amd64' if os.name == 'nt' else 'linux-x86_64'))
         sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except IndexError:
        print("CARLA egg NOT found.")
        pass

import carla
from agents.navigation.behavior_agent import BehaviorAgent
from mapping.fusion_server import FusionServer
from mapping.occupancy_grid import LocalMapper

def main():
    # --- CONFIGURATION ---
    DURATION = 120  # Seconds to run
    LOG_FILE = "results_coverage_single.csv"
    NUM_AGENTS = 1  # Change to 1 for baseline
    # ---------------------

    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.get_world()

    # 1. Setup Synchronous Mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05 # 20 FPS
    world.apply_settings(settings)

    # 2. Setup Fusion Server
    fusion_server = FusionServer(grid_size=0.5, map_dim=500) # Adjust dim if needed

    # 3. Spawn Agents
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
    spawn_points = world.get_map().get_spawn_points()
    
    vehicles = []
    agents = []
    sensors = []

    try:
        print(f"Starting Coverage Experiment with {NUM_AGENTS} agents...")
        for i in range(NUM_AGENTS):
            spawn = spawn_points[i*5] # Spread them out
            vehicle = world.spawn_actor(vehicle_bp, spawn)
            vehicles.append(vehicle)
            
            # Init Agent
            agent = BehaviorAgent(vehicle, behavior='normal', fusion_server=fusion_server)
            agent.set_destination(random.choice(spawn_points).location)
            agents.append(agent)

            # Init Sensor (Semantic LiDAR)
            lidar_bp = bp_lib.find('sensor.lidar.ray_cast_semantic')
            lidar_bp.set_attribute('range', '50')
            lidar_bp.set_attribute('rotation_frequency', '20')
            lidar_bp.set_attribute('points_per_second', '56000')
            lidar_bp.set_attribute('channels', '32')
            
            # Spawn & Attach
            lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
            lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
            sensors.append(lidar)
            
            # Callback
            # We need a lambda to pass data to the specific agent's mapper
            # Note: In a real script, use the proper SensorInterface/Callback classes.
            # Here we hack it for simplicity of the test script.
            # Callback to update LocalMapper
            def on_lidar_data(weak_agent, weak_veh, data):
                agent = weak_agent()
                veh = weak_veh()
                if not agent or not veh:
                    return
                # Process Lidar data into LocalMapper
                # Note: agent._local_mapper.process_lidar expects raw CARLA LidarMeasurement
                agent._local_mapper.process_lidar(data, veh.get_transform())

            import weakref
            lidar.listen(lambda data: on_lidar_data(weakref.ref(agent), weakref.ref(vehicle), data))


        # 4. Data Collection Loop
        start_time = time.time()
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Time_Sec", "Area_M2"])
            
            step = 0
            while True:
                # Tick Server
                world.tick()
                
                # Check Time
                elapsed = time.time() - start_time
                if elapsed > DURATION:
                    break

                # Control Agents & Update Map
                for idx, agent in enumerate(agents):
                    agent.run_step() # This updates local mapper internally if hooked up correctly
                    
                    # Manually push to fusion server for this test script to ensure capture
                    # (Assuming agent has _local_mapper populated by sensor callback in run_step)
                    pose = vehicles[idx].get_transform()
                    loc = (pose.location.x, pose.location.y, pose.rotation.yaw)
                    local_grid = agent._local_mapper.get_local_map()
                    fusion_server.update_map(idx, local_grid, loc)

                # Log Metric (every 1 second / 20 frames)
                if step % 20 == 0:
                    area = fusion_server.calculate_coverage()
                    print(f"Time: {elapsed:.1f}s | Coverage: {area:.2f} m2")
                    writer.writerow([round(elapsed, 1), round(area, 2)])
                
                step += 1

    finally:
        print("Cleaning up actors...")
        for s in sensors: s.destroy()
        for v in vehicles: v.destroy()
        # Reset settings
        settings.synchronous_mode = False
        world.apply_settings(settings)
        print(f"Coverage data saved to {LOG_FILE}")

if __name__ == '__main__':
    main()