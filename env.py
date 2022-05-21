# link to the code used to generate data: https://github.com/rail-berkeley/d4rl/blob/master/d4rl/carla/data_collection_agent_lane.py
import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math

# import carla egg
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# dont run an episode for more than this amount of seconds
SECONDS_PER_EPISODE = 100

def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))

class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        min_alt, max_alt = [20, 90]
        self.altitude = 0.5 * (max_alt + min_alt) + 0.5 * (max_alt - min_alt) * math.cos(self._t)

class Storm(object):
    def __init__(self, precipitation):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 90.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 60.0)
        self.rain = clamp(self._t, 0.0, 80.0)
        delay = -10.0 if self._increasing else 90.0
        self.puddles = clamp(self._t + delay, 0.0, 85.0)
        self.wetness = clamp(self._t * 5, 0.0, 100.0)
        self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
        self.fog = clamp(self._t - 10, 0.0, 30.0)
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

class Weather(object):
    def __init__(self, world, changing_weather_speed):
        self.world = world
        self.reset()
        self.weather = world.get_weather()
        self.changing_weather_speed = changing_weather_speed
        self._sun = Sun(self.weather.sun_azimuth_angle, self.weather.sun_altitude_angle)
        self._storm = Storm(self.weather.precipitation)

    def reset(self):
        weather_params = carla.WeatherParameters(sun_altitude_angle=90.)
        self.world.set_weather(weather_params)

    def tick(self):
        self._sun.tick(self.changing_weather_speed)
        self._storm.tick(self.changing_weather_speed)
        self.weather.cloudiness = self._storm.clouds
        self.weather.precipitation = self._storm.rain
        self.weather.precipitation_deposits = self._storm.puddles
        self.weather.wind_intensity = self._storm.wind
        self.weather.fog_density = self._storm.fog
        self.weather.wetness = self._storm.wetness
        self.weather.sun_azimuth_angle = self._sun.azimuth
        self.weather.sun_altitude_angle = self._sun.altitude
        self.world.set_weather(self.weather)

# environment class
class CarEnv:
    CAMERA = None
    WIDTH = 84
    HEIGHT = 84
    WEATHER_CHANGE_SPEED = 0.02

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(200.0)
        self.world = self.client.get_world()
        self.model = self.world.get_blueprint_library().find('vehicle.audi.a2')
        self.weather = Weather(self.world, self.WEATHER_CHANGE_SPEED)     
        self.vehicles_list = []
        self.spawn_vehicles()

    def reset(self):
        self.collisions = []
        self.actor_list = []

        # set up vehicle at random spawn point. might throw error if spawn point is not free
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model, self.transform)
        self.actor_list.append(self.vehicle)
        
        spectator = self.world.get_spectator()
        spectator.set_transform(self.vehicle.get_transform())

        # setup camera
        self.rgb = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb.set_attribute('image_size_x', f'{self.WIDTH}')
        self.rgb.set_attribute('image_size_y', f'{self.HEIGHT}')
        self.rgb.set_attribute('fov', '90')

        transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(yaw=0.0)) # carla.Transform(carla.Location(x=2.5, z=0.7))
        self.cam = self.world.spawn_actor(self.rgb, transform, attach_to=self.vehicle)

        self.actor_list.append(self.cam)
        self.cam.listen(lambda data: self.reshape_image(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        
        time.sleep(10)

        # setup collision detector
        col_sensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.col_sensor = self.world.spawn_actor(col_sensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.col_sensor)
        self.col_sensor.listen(lambda col: self.collision_data(col))
        
        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))

        return self.CAMERA

    def spawn_vehicles(self):
        traffic_manager = self.client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(2)
        traffic_manager.set_random_device_seed(0)

        blueprints = self.world.get_blueprint_library().filter('model3')

        num_vehicles = 20

        spawn_points = self.world.get_map().get_spawn_points()
        init_transforms = np.random.choice(spawn_points, num_vehicles)
        
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # spawn vehicles

        blueprint = blueprints[0]
        batch = []
        for tr in init_transforms:
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):        
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)

            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(SpawnActor(blueprint, tr)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in self.client.apply_batch_sync(batch, False):
            self.vehicles_list.append(response.actor_id)

        for response in self.client.apply_batch_sync(batch):
            if response.error:
                pass
            else:
                self.vehicles_list.append(response.actor_id)

    # log collisions
    def collision_data(self, event):
        self.collisions.append(event)

    def reshape_image(self, image):
        bgra = np.array(image.raw_data).reshape(self.HEIGHT, self.WIDTH, 4)
        bgr = bgra[:, :, :3]
        rgb = np.flip(bgr, axis=2)
        self.CAMERA = rgb/255.0

    def step(self, action):
        print(action)
        self.vehicle.apply_control(carla.VehicleControl(throttle=action[0], steer=action[1], brake=action[2]))

        # Weather evolves
        self.weather.tick()

        # dont really need reward for offline RL
        reward = 0
        
        done = False

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.CAMERA, reward, done, None
