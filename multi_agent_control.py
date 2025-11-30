#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import numpy.random as random
import re
import sys
import weakref

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
    from pygame.locals import K_TAB
    from pygame.locals import K_n
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    try:
        # Try one level deeper
        sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except IndexError:
        pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from mapping.fusion_server import FusionServer
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, hud, args):
        """Constructor method"""
        self._args = args
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.players = [] # List of players
        self.collision_sensors = []
        self.lane_invasion_sensors = []
        self.gnss_sensors = []
        self.camera_managers = []
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.active_agent_index = 0 # Which agent to follow with camera

    def restart(self, args):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_managers[0].index if self.camera_managers else 0
        cam_pos_id = cam_index

        # Get a blueprint.
        blueprint = self.world.get_blueprint_library().filter(self._actor_filter)[0]
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # Spawn the players.
        if self.players:
            self.destroy()
            self.players = []

        spawn_points = self.map.get_spawn_points()
        if len(spawn_points) < 2:
            print('Not enough spawn points!')
            sys.exit(1)

        # Spawn Agent A
        spawn_point_a = spawn_points[0]
        player_a = self.world.try_spawn_actor(blueprint, spawn_point_a)
        self.modify_vehicle_physics(player_a)
        self.players.append(player_a)

        # Spawn Agent B (far away if possible, or just next index)
        spawn_point_b = spawn_points[min(10, len(spawn_points)-1)] # Try to pick a distant one
        player_b = self.world.try_spawn_actor(blueprint, spawn_point_b)
        self.modify_vehicle_physics(player_b)
        self.players.append(player_b)

        if self._args.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Set up the sensors for EACH player
        for player in self.players:
            self.collision_sensors.append(CollisionSensor(player, self.hud))
            self.lane_invasion_sensors.append(LaneInvasionSensor(player, self.hud))
            self.gnss_sensors.append(GnssSensor(player))
            cm = CameraManager(player, self.hud)
            cm.set_sensor(cam_index, notify=False)
            self.camera_managers.append(cm)

        actor_type = get_actor_display_name(self.players[0])
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.players[0].get_world().set_weather(preset[0])

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)

    def render(self, display):
        """Render world"""
        # Render the active agent's camera
        self.camera_managers[self.active_agent_index].render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        for cm in self.camera_managers:
            cm.sensor.destroy()
            cm.sensor = None
            cm.index = None

    def destroy(self):
        """Destroys all actors"""
        actors = []
        for cm in self.camera_managers:
            if cm.sensor: actors.append(cm.sensor)
        for cs in self.collision_sensors:
            if cs.sensor: actors.append(cs.sensor)
        for ls in self.lane_invasion_sensors:
            if ls.sensor: actors.append(ls.sensor)
        for gs in self.gnss_sensors:
            if gs.sensor: actors.append(gs.sensor)
        for p in self.players:
            if p: actors.append(p)
            
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, world):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_TAB:
                    world.active_agent_index = (world.active_agent_index + 1) % len(world.players)
                    world.hud.notification(f"Switched to Agent {world.active_agent_index}")
                elif event.key == K_n:
                    world.camera_managers[world.active_agent_index].next_sensor()

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        
        # Display info for the active agent
        player = world.players[world.active_agent_index]
        transform = player.get_transform()
        vel = player.get_velocity()
        control = player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        
        # Use sensors from the active agent
        colhist = world.collision_sensors[world.active_agent_index].get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensors[world.active_agent_index].lat, world.gnss_sensors[world.active_agent_index].lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

    def render_side_map(self, display, fusion_server, players):
        """
        Renders the global occupancy grid and trajectories to the right side of the screen.
        """
        global_map = fusion_server.get_global_map()
        if global_map is None:
            return

        # Surface for the Map (Same size as Camera View)
        map_surface = pygame.Surface(self.dim)
        map_surface.fill((50, 50, 50)) # Default Gray (Unknown)
        
        # 1. Render Grid
        # Create RGB image buffer
        h, w = global_map.shape
        rgb_map = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Color Scheme
        # Unknown (approx 0.5) -> Gray (50, 50, 50)
        unknown_mask = (global_map > 0.45) & (global_map < 0.55)
        rgb_map[unknown_mask] = [50, 50, 50]
        
        # Free (< 0.45) -> Black (0, 0, 0)
        free_mask = global_map <= 0.45
        rgb_map[free_mask] = [0, 0, 0]
        
        # Occupied (>= 0.55) -> White (255, 255, 255)
        occ_mask = global_map >= 0.55
        rgb_map[occ_mask] = [255, 255, 255]
        
        # Create Pygame Surface
        # Pygame expects (W, H, 3) but numpy is (H, W, 3). Swap axes.
        surf_array = rgb_map.swapaxes(0, 1)
        temp_surf = pygame.surfarray.make_surface(surf_array)
        
        # Scale to fit
        scale = self.dim[1] / fusion_server.grid_dim
        map_w = int(fusion_server.grid_dim * scale)
        map_h = int(fusion_server.grid_dim * scale)
        
        offset_x = (self.dim[0] - map_w) // 2
        offset_y = 0
        
        scaled_surf = pygame.transform.scale(temp_surf, (map_w, map_h))
        map_surface.blit(scaled_surf, (offset_x, offset_y))
        
        # Helper for coordinate transform
        def to_screen(gx, gy):
            # Global -> Grid
            r = (fusion_server.max_x - gx) / fusion_server.grid_size
            c = (gy - fusion_server.min_y) / fusion_server.grid_size
            # Grid -> Screen
            sx = offset_x + int(c * scale)
            sy = offset_y + int(r * scale)
            return sx, sy

        # 2. Draw Trajectories
        for agent_id, traj in fusion_server.trajectories.items():
            if len(traj) > 1:
                points = [to_screen(p[0], p[1]) for p in traj]
                # Filter points? Pygame handles clipping.
                color = (0, 255, 255) if agent_id == 0 else (255, 0, 255)
                pygame.draw.lines(map_surface, color, False, points, 2)

        # 3. Draw Agents
        for i, player in enumerate(players):
            t = player.get_transform()
            sx, sy = to_screen(t.location.x, t.location.y)
            
            color = (0, 0, 255) if i == 0 else (255, 0, 0)
            pygame.draw.circle(map_surface, color, (sx, sy), 5)
            
            # Heading
            rad = math.radians(t.rotation.yaw)
            vx = math.cos(rad)
            vy = math.sin(rad)
            # Screen vector:
            # x_s ~ c ~ y_g
            # y_s ~ r ~ -x_g
            dx = vy * 15
            dy = -vx * 15
            pygame.draw.line(map_surface, (255, 255, 0), (sx, sy), (sx + dx, sy + dy), 2)
            
            # Label
            label = self._font_mono.render(f"A{i}", True, (255, 255, 255))
            map_surface.blit(label, (sx + 10, sy - 10))

        # Blit to Right Side of Main Display
        display.blit(map_surface, (self.dim[0], 0))
        
        # Draw Separator
        pygame.draw.line(display, (255, 255, 255), (self.dim[0], 0), (self.dim[0], self.dim[1]), 2)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), attachment.Rigid),
            (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), attachment.Rigid),
            (carla.Transform(carla.Location(x=-5.0, z=2.5), carla.Rotation(pitch=-10.0)), attachment.SpringArmGhost),
            ]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)'],
            ['virtual_bev_map', None, 'Occupancy Grid Map']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            if item[0] == 'virtual_bev_map':
                item.append(None)
                continue
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.index != index))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            
            if self.sensors[index][0] == 'virtual_bev_map':
                self.sensor = None
            else:
                self.sensor = self._parent.get_world().spawn_actor(
                    self.sensors[index][-1],
                    self._camera_transforms[self.transform_index][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[self.transform_index][1])
                # We need to pass the lambda a weak reference to
                # self to avoid circular reference.
                weak_self = weakref.ref(self)
                self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================


def game_loop(args):
    """
    Main loop of the simulation.
    """
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)

        display = pygame.display.set_mode(
            (args.width * 2, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)

        # Initialize Fusion Server
        fusion_server = FusionServer()

        # Create Agents
        agents = []
        for player in world.players:
            agent = BehaviorAgent(player, behavior=args.behavior, fusion_server=fusion_server)
            # Set destination to a random spawn point
            spawn_points = world.map.get_spawn_points()
            random.shuffle(spawn_points)
            if spawn_points[0].location != player.get_location():
                destination = spawn_points[0].location
            else:
                destination = spawn_points[1].location
            agent.set_destination(destination)
            agents.append(agent)

        clock = pygame.time.Clock()
        frame_count = 0
        MAPPING_FREQUENCY = 5
        VISUALIZATION_FREQUENCY = 10

        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(world):
                return

            frame_count += 1
            # Asynchronous update
            world.tick(clock)
            
            # Update Virtual Sensor (BEV Map) if active
            cm = world.camera_managers[world.active_agent_index]
            if cm.sensors[cm.index][0] == 'virtual_bev_map':
                # Render map to camera surface
                # Reuse render_global_map logic but scale to full screen?
                # Or just use the existing render_global_map but blit to cm.surface
                
                # Create surface if None
                if cm.surface is None:
                    cm.surface = pygame.Surface((args.width, args.height))
                
                # Fill black
                cm.surface.fill((0,0,0))
                
                # Render map
                # We can call a helper or just do it here.
                # Let's use a simplified version of render_global_map that fills the screen.
                global_map = fusion_server.get_global_map()
                if global_map is not None:
                    grid_img = (global_map * 255).astype(np.uint8)
                    surf_array = np.stack([grid_img.T]*3, axis=-1)
                    temp_surf = pygame.surfarray.make_surface(surf_array)
                    scaled_surf = pygame.transform.scale(temp_surf, (args.width, args.height))
                    cm.surface.blit(scaled_surf, (0, 0))
                    
                    # Draw Agents on top
                    # Scale factor
                    scale_factor = args.width / fusion_server.grid_dim # Assuming square aspect ratio
                    
                    for idx, player in enumerate(world.players):
                        t = player.get_transform()
                        loc = t.location
                        yaw = t.rotation.yaw
                        
                        r = (fusion_server.max_x - loc.x) / fusion_server.grid_size
                        c = (loc.y - fusion_server.min_y) / fusion_server.grid_size
                        
                        dx = int(c * scale_factor)
                        dy = int(r * scale_factor)
                        
                        color = (0, 0, 255) if idx == 0 else (255, 0, 0)
                        pygame.draw.circle(cm.surface, color, (dx, dy), 5)
                        
                        rad = math.radians(yaw)
                        vx_g = math.cos(rad)
                        vy_g = math.sin(rad)
                        dx_d = vy_g * 15
                        dy_d = -vx_g * 15
                        pygame.draw.line(cm.surface, (255, 255, 0), (dx, dy), (dx + dx_d, dy + dy_d), 2)

            world.render(display)
            
            # Render Global Map
            # Render Side Map (Right Side)
            if frame_count % VISUALIZATION_FREQUENCY == 0:
                world.hud.render_side_map(display, fusion_server, world.players)
            
            pygame.display.flip()

            # Control Loop for Agents
            for i, agent in enumerate(agents):
                if agent.done() or agent.is_stuck():
                    if agent.is_stuck():
                        print(f"Agent {i} is stuck! Rerouting...")
                    
                    # Try to reroute to frontier
                    if not agent.reroute_to_frontier():
                        # Pick a new random destination if no frontier or failed
                        spawn_points = world.map.get_spawn_points()
                        agent.set_destination(random.choice(spawn_points).location)
                        print(f"Agent {i} random rerouting...")
                
                control = agent.run_step()
                control.manual_gear_shift = False
                world.players[i].apply_control(control)
                
                # Map Fusion Update
                # Get local map from agent's mapper
                if hasattr(agent, '_local_mapper'):
                    local_map = agent._local_mapper.get_local_map()
                    # Get pose from vehicle (x, y, yaw)
                    t = world.players[i].get_transform()
                    pose = (t.location.x, t.location.y, t.rotation.yaw)
                    
                    if frame_count % MAPPING_FREQUENCY == 0:
                        fusion_server.update_map(i, local_map, pose)
                    
                    # Update Trajectory
                    fusion_server.update_trajectory(i, pose)
    finally:
        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Multi-Agent Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.tesla.model3',
        help='Actor filter (default: "vehicle.tesla.model3")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument(
        '--async',
        action='store_false',
        dest='sync',
        help='Use asynchronous mode execution')
    argparser.set_defaults(sync=True)
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening on %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as error:
        logging.exception(error)


if __name__ == '__main__':
    main()
