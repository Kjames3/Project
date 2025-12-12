#!/usr/bin/env python
import time
import statistics
try:
    import pygame
except ImportError:
    pass

# We will just import the existing multi_agent_control logic but strip the visuals
# Or simpler: Re-implement the loop with timing hooks.

import glob
import os
import sys
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
from mapping.fusion_server import FusionServer
from mapping.occupancy_grid import LocalMapper

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    
    # Enable Sync
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Setup 2 Agents + Fusion
    fusion_server = FusionServer(grid_size=0.5)
    bp = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
    spawns = world.get_map().get_spawn_points()
    
    vehicles = []
    agents = []
    sensors = []
    
    for i in range(2):
        v = world.spawn_actor(bp, spawns[i])
        vehicles.append(v)
        a = BehaviorAgent(v, fusion_server=fusion_server)
        a.set_destination(spawns[i-1].location)
        agents.append(a)

        # Setup Lidar for realistic load
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('points_per_second', '56000')
        lidar_bp.set_attribute('channels', '32')
        
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=v)
        sensors.append(lidar)
        
        # Callback to update LocalMapper
        def on_data(agent_ref, veh_ref, data):
            ag = agent_ref()
            vh = veh_ref()
            if ag and vh:
                ag._local_mapper.process_lidar(data, vh.get_transform())

        import weakref
        lidar.listen(lambda d: on_data(weakref.ref(a), weakref.ref(v), d))

    print("Warming up (20 frames)...")
    for _ in range(20):
        world.tick()

    print("Starting Performance Profiling (60 seconds)...")
    
    frame_times = []
    start_test = time.time()
    
    try:
        while time.time() - start_test < 60:
            iter_start = time.time()
            
            # 1. Tick Server
            world.tick()
            
            # 2. Agent Logic (The heavy part)
            for i, agent in enumerate(agents):
                agent.run_step()
                # Simulate mapping overhead (calls Numba functions)
                if hasattr(agent, '_local_mapper'):
                    lm = agent._local_mapper.get_local_map()
                    pose = (vehicles[i].get_location().x, vehicles[i].get_location().y, 0)
                    fusion_server.update_map(i, lm, pose)
            
            # 3. Record Time
            iter_end = time.time()
            frame_times.append(iter_end - iter_start)

    finally:
        for s in sensors: s.destroy()
        for v in vehicles: v.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)

        # Statistics
        avg_time = statistics.mean(frame_times)
        avg_fps = 1.0 / avg_time if avg_time > 0 else 0
        min_fps = 1.0 / max(frame_times)
        
        print("\n" + "="*30)
        print("PERFORMANCE METRICS")
        print("="*30)
        print(f"Average Step Time: {avg_time*1000:.2f} ms")
        print(f"Average FPS:       {avg_fps:.2f}")
        print(f"Minimum FPS:       {min_fps:.2f} (Lag Spikes)")
        print("="*30)

if __name__ == '__main__':
    main()