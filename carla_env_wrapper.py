# ==============================================================================
# VR-RL CARLA ENVIRONMENT WRAPPER
# ==============================================================================
# This file handles the connection to the CARLA simulator.
# It spawns the vehicle, attaches sensors, and translates the simulator state
# into a format that the Reinforcement Learning (RL) agent can understand.
#
# Key Features:
# 1. Reset: Spawns car at a random location.
# 2. Step: Applies steering/throttle and returns (Observation, Reward, Done).
# 3. Reward: Calculates score based on Speed, Lane Centering, and Heading.
# 4. Cleanup: Destroys actors to prevent memory leaks.
# ==============================================================================

import random
import math
import numpy as np
import gymnasium as gym
import carla
import weakref
import pygame
import os

# -------------------------------
# Constants
# -------------------------------
HOST = "localhost"
PORT = 2000
TIMEOUT = 20.0

# The vehicle model to spawn (Tesla Model 3 is standard for RL)
CAR_NAME = "model3"

# Maximum duration of one episode (50 seconds at 20 FPS)
EPISODE_MAX_STEPS = 1000

# Camera Settings
# We use Semantic Segmentation for the Agent (cleaner view of road/lanes)
# We use RGB for the Human View (Pygame window)
SSC_CAMERA = "sensor.camera.semantic_segmentation"
RGB_CAMERA = "sensor.camera.rgb"
IM_WIDTH = 160   # RL Input Width
IM_HEIGHT = 80   # RL Input Height

# Visual Display Window Size
VIS_WIDTH = 640
VIS_HEIGHT = 480

class CarlaEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, town="Town01", checkpoint_frequency=100, visual_display=True):
        super().__init__()
        self.town = town
        self.display_on = visual_display
        self.episode_count = 0

        # Initialize Pygame window for human visualization
        if self.display_on:
            pygame.init()
            self.display = pygame.display.set_mode(
                (VIS_WIDTH, VIS_HEIGHT), 
                pygame.HWSURFACE | pygame.DOUBLEBUF
            )
            pygame.display.set_caption("VR-RL Training View")

        # Connect to the CARLA Server
        print(f"[CarlaEnv] Connecting to {HOST}:{PORT}")
        self.client = carla.Client(HOST, PORT)
        self.client.set_timeout(TIMEOUT)

        # Load the Map (Town01 is simple: straight roads + 90 deg turns)
        self.world = self.client.load_world(town)
        self.map = self.world.get_map()
        self.bp_lib = self.world.get_blueprint_library()

        # Enable Synchronous Mode
        # This ensures the Python script and CARLA server run in lock-step (20 FPS).
        # Without this, the simulation would run faster/slower than the agent decisions.
        self._set_sync_mode(True)

        self.vehicle = None
        self.sensor_list = []
        self.actor_list = []
        
        # --- Define Gym Spaces ---
        # Action: [Steering (-1 to 1), Throttle (-1 to 1)]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,), dtype=np.float32
        )

        # Observation: Tuple(Image, NavigationVector)
        # Image: 80x160x3 (Semantic Segmentation)
        # NavVector: [Speed, DistCenter, Angle, Time, 0]
        self.observation_space = gym.spaces.Tuple([
            gym.spaces.Box(low=0, high=255, shape=(IM_HEIGHT, IM_WIDTH, 3), dtype=np.uint8),
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        ])

        # Initial cleanup to remove any zombie cars from previous runs
        self._hard_cleanup()

    # ==========================================================================
    # RESET: Starts a new episode
    # ==========================================================================
    def reset(self, *, seed=None, options=None):
        self.episode_count += 1
        
        # [Hard Reset] Every 50 episodes, reload the entire map.
        # This fixes a known CARLA bug where "zombie sockets" cause crashes over time.
        if self.episode_count % 50 == 0:
            print("[CarlaEnv] Performing Periodic World Reload (Hard Reset)...")
            self._hard_cleanup()
            self.world = self.client.reload_world(False) 
            self.map = self.world.get_map()
            self.bp_lib = self.world.get_blueprint_library()
            self._set_sync_mode(True)
        else:
            self._destroy_all()

        # Spawn the Agent Vehicle
        try:
            transform = random.choice(self.map.get_spawn_points())
            self.vehicle = self.world.try_spawn_actor(
                self.bp_lib.filter(CAR_NAME)[0], transform
            )
            # Retry spawn if collision occurred immediately
            if self.vehicle is None:
                for _ in range(5):
                    transform = random.choice(self.map.get_spawn_points())
                    self.vehicle = self.world.try_spawn_actor(
                        self.bp_lib.filter(CAR_NAME)[0], transform
                    )
                    if self.vehicle: break
            
            if self.vehicle is None:
                raise RuntimeError("Could not spawn vehicle.")
                
            self.actor_list.append(self.vehicle)

            # Attach Sensors
            # 1. RL Camera (Semantic Segmentation) -> Goes to Neural Network
            self.ss_cam = self._setup_sensor(
                SSC_CAMERA, 
                carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-15)), 
                self._ss_callback
            )
            # 2. Collision Sensor -> Detects crashes
            self.col_sensor = self._setup_sensor(
                "sensor.other.collision", 
                carla.Transform(), 
                self._col_callback
            )
            
            # 3. Visualization Camera (RGB) -> Goes to Pygame Window
            if self.display_on:
                bp = self.bp_lib.find(RGB_CAMERA)
                bp.set_attribute("image_size_x", str(VIS_WIDTH))
                bp.set_attribute("image_size_y", str(VIS_HEIGHT))
                bp.set_attribute("fov", "100")
                cam_transform = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=-15))
                self.rgb_cam = self.world.spawn_actor(bp, cam_transform, attach_to=self.vehicle)
                weak = weakref.ref(self)
                self.rgb_cam.listen(lambda data: self._rgb_callback(weak, data))
                self.sensor_list.append(self.rgb_cam)

        except RuntimeError as e:
            print(f"[CarlaEnv] Reset failed: {e}. Retrying hard reset...")
            self._hard_cleanup()
            return self.reset()

        # Reset buffers
        self.ss_image = None
        self.collision_hist = []
        
        # Warmup: Tick world 10 times to let physics settle and sensors wake up
        for _ in range(10):
            self.world.tick()

        # Reset Navigation State
        self.timesteps = 0
        self.velocity = 0.0
        self.distance_from_center = 0.0
        self.angle = 0.0
        self.target_speed = 22.0
        self.max_distance_from_center = 3.5

        return self._get_obs(), {}

    # ==========================================================================
    # STEP: Applies action and returns result
    # ==========================================================================
    def step(self, action):
        self.timesteps += 1
        
        # 1. Apply Controls
        # Direct Control (No Smoothing) for maximum responsiveness
        steer = float(np.clip(action[0], -1.0, 1.0))
        
        # Map throttle from [-1, 1] to [0, 1]
        throttle = float(np.clip((action[1] + 1.0) / 2.0, 0.0, 1.0))
        
        if self.vehicle and self.vehicle.is_alive:
            self.vehicle.apply_control(carla.VehicleControl(steer=steer, throttle=throttle))
            self.world.tick() # Advance simulation by 0.05s
            self._update_navigation() # Recalculate speed/position
        else:
            return self._get_obs(), 0, True, False, {}

        # 2. Calculate Reward
        reward, done, info = self._compute_reward_done()
        
        # Update Pygame window
        if self.display_on:
            pygame.event.pump()
            
        return self._get_obs(), reward, done, False, info

    # ==========================================================================
    # OBSERVATION: Returns what the agent sees
    # ==========================================================================
    def _get_obs(self):
        if self.ss_image is None:
            self.ss_image = np.zeros((IM_HEIGHT, IM_WIDTH, 3), dtype=np.uint8)
            
        # Normalize features for the Neural Network (approx 0-1 range)
        nav_features = np.array([
            self.velocity / 30.0,
            self.distance_from_center / self.max_distance_from_center,
            self.angle,
            self.timesteps / EPISODE_MAX_STEPS,
            0.0 
        ], dtype=np.float32)
        
        return (self.ss_image, nav_features)

    # ==========================================================================
    # REWARD FUNCTION (The "High Score" Version)
    # ==========================================================================
    def _compute_reward_done(self):
        done = False
        
        # --- 1. Calculate Factors (0.0 to 1.0) ---
        
        # Speed: Reward being close to 22 km/h
        v_diff = abs(self.velocity - self.target_speed)
        r_speed = 1.0 - (v_diff / self.target_speed)
        r_speed = max(0.0, r_speed)

        # Center: Reward being in the middle of the lane
        dist_norm = self.distance_from_center / self.max_distance_from_center
        r_center = 1.0 - (dist_norm ** 2)
        r_center = max(0.0, r_center)

        # Angle: Reward facing straight down the road
        r_angle = max(0.0, math.cos(self.angle))
        
        # --- 2. Basic Step Reward ---
        # Multiplicative: If ANY factor is bad, the whole reward drops.
        # This forces the agent to be fast AND centered AND aligned.
        step_reward = r_speed * r_center * r_angle

        # --- 3. Bonuses (Incentivize Perfection) ---
        # If driving fast and perfect, give extra points.
        if self.velocity > 15.0 and self.distance_from_center < 1.0:
            step_reward += 2.0 # Big Bonus!
        elif self.velocity > 5.0 and self.distance_from_center < 1.5:
             step_reward += 0.5

        # --- 4. Penalties & Termination ---
        if len(self.collision_hist) > 0:
            done = True
            reward = -100.0 # Crash Penalty
        elif self.distance_from_center > self.max_distance_from_center:
            done = True
            reward = -100.0 # Off-road Penalty
        elif self.timesteps >= EPISODE_MAX_STEPS:
            done = True
            reward = 0.0 # Safe finish
        else:
            reward = step_reward

        return reward, done, {}

    # ==========================================================================
    # SENSOR HELPERS
    # ==========================================================================
    def _setup_sensor(self, type_id, transform, callback):
        bp = self.bp_lib.find(type_id)
        if type_id == SSC_CAMERA:
            bp.set_attribute("image_size_x", str(IM_WIDTH))
            bp.set_attribute("image_size_y", str(IM_HEIGHT))
            bp.set_attribute("fov", "100")
        
        sensor = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        weak = weakref.ref(self)
        sensor.listen(lambda data: callback(weak, data))
        self.sensor_list.append(sensor)
        return sensor

    @staticmethod
    def _ss_callback(weak, image):
        # Process Semantic Segmentation Image
        self = weak()
        if not self: return
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        self.ss_image = array.reshape((image.height, image.width, 4))[:, :, :3]

    @staticmethod
    def _rgb_callback(weak, image):
        # Process RGB Image (For Human Visualization Only)
        self = weak()
        if not self or not self.display_on: return
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]
        array = array[:, :, ::-1] # BGR to RGB
        
        # Transpose for Pygame surface
        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.display:
            self.display.blit(surface, (0, 0))
            pygame.display.flip()

    @staticmethod
    def _col_callback(weak, event):
        # Record collisions
        self = weak()
        if not self: return
        self.collision_hist.append(event)

    # ==========================================================================
    # MATH HELPERS
    # ==========================================================================
    def _update_navigation(self):
        """Calculates Velocity, Lane Deviation, and Angle Error."""
        if not self.vehicle: return
        
        # 1. Velocity (m/s -> km/h)
        vel = self.vehicle.get_velocity()
        self.velocity = 3.6 * math.sqrt(vel.x**2 + vel.y**2)

        # 2. Get Waypoint (Center of current lane)
        t = self.vehicle.get_transform()
        loc = t.location
        waypoint = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if not waypoint: return

        # 3. Distance from Center (Cross Product)
        wp_loc = waypoint.transform.location
        vec_wp_to_car = np.array([loc.x - wp_loc.x, loc.y - wp_loc.y])
        wp_fwd = waypoint.transform.get_forward_vector()
        vec_wp_fwd = np.array([wp_fwd.x, wp_fwd.y])
        
        self.distance_from_center = abs(vec_wp_to_car[0]*vec_wp_fwd[1] - vec_wp_to_car[1]*vec_wp_fwd[0])
        
        # 4. Angle Error (Dot Product)
        car_fwd = t.get_forward_vector()
        vec_car_fwd = np.array([car_fwd.x, car_fwd.y])
        norm_car = np.linalg.norm(vec_car_fwd)
        norm_wp = np.linalg.norm(vec_wp_fwd)
        
        if norm_car > 0 and norm_wp > 0:
            dot = np.dot(vec_car_fwd, vec_wp_fwd) / (norm_car * norm_wp)
            dot = np.clip(dot, -1.0, 1.0)
            self.angle = math.acos(dot)
        else:
            self.angle = 0.0

    # ==========================================================================
    # CLEANUP
    # ==========================================================================
    def _set_sync_mode(self, active):
        settings = self.world.get_settings()
        settings.synchronous_mode = active
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    def _hard_cleanup(self):
        """Nuclear cleanup: Destroys ALL sensors in the world to fix socket errors."""
        for s in self.sensor_list:
            if s and s.is_alive: s.stop()
        
        if self.world:
            actors = self.world.get_actors()
            cleanup_list = actors.filter('*sensor*')
            vehicle_list = actors.filter('*vehicle*')
            self.client.apply_batch([carla.command.DestroyActor(x) for x in cleanup_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])

        try: self.world.tick()
        except: pass

        self.sensor_list = []
        self.actor_list = []
        self.collision_hist = []

    def _destroy_all(self):
        """Standard cleanup for end of episode."""
        for s in self.sensor_list:
            if s and s.is_alive: s.stop()
        
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        self.world.tick()
        
        self.sensor_list.clear()
        self.actor_list.clear()
        self.collision_hist = []

    def close(self):
        self._destroy_all()
        self._set_sync_mode(False)
        if self.display_on:
            pygame.quit()