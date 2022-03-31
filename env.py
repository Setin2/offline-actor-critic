import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math

#import carla egg
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

client = carla.Client("localhost", 2000)
SECONDS_PER_EPISODE = 100

# environment class
class CarEnv:
    CAMERA = None
    WIDTH = 48
    HEIGHT = 48
    STEER_AMT = 1.0

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(200.0)
        self.world = self.client.get_world()
        self.model = self.world.get_blueprint_library().filter('model3')[0]

    def reset(self):
        self.collisions = []
        self.actor_list = []

        # set up vehicle at random spawn point
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model, self.transform)
        self.actor_list.append(self.vehicle)

        # setup camera
        self.rgb = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb.set_attribute('image_size_x', f'{self.WIDTH}')
        self.rgb.set_attribute('image_size_y', f'{self.HEIGHT}')
        self.rgb.set_attribute('fov', '110')

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
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

    # log collisions
    def collision_data(self, event):
        self.collisions.append(event)

    # reshape image given by camera
    def reshape_image(self, image):
        bgra = np.array(image.raw_data).reshape(self.HEIGHT, self.WIDTH, 4)
        bgr = bgra[:, :, :3]
        rgb = np.flip(bgr, axis=2)
        self.CAMERA = rgb

    def step(self, action):
        action = action.tolist()[0]
        print(action)
        self.vehicle.apply_control(carla.VehicleControl(action[0], action[1], action[2]))

        # stop episode & penalize on collision
        if len(self.collisions) != 0:
            done = True
            reward = -200
        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.CAMERA, reward, done, None
