import gym
from gym import logger, spaces
from gym.utils import seeding
from gym_thegame.envs.server import Server
from gym_thegame.envs.client import Client
import configparser
import math
import multiprocessing
from time import sleep
import numpy as np


class ThegameEnv(gym.Env):
  """thegame environment
  """
  metadata = {'render.modes': ['human', 'rgb_array']}

  def __init__(self):
    config = configparser.ConfigParser()
    config.read('thegame.cfg')
    cfg = config['DEFAULT']
    self.server_bin = cfg.get('ServerBinary', './thegame-server')
    self.port = cfg.getint('Port', 50051)
    self.name = cfg.get('Name', 'gym')
    if 'Environment' in config:
      cfg = config['Environment']
    self.width = cfg.getint('Width', 160)
    self.height = cfg.getint('Height', 160)
    self.total_steps = cfg.getint('TotalSteps', 10000)
    self.acc_disc = cfg.getint('AccelerateDiscretize', 12)
    self.shoot_disc = cfg.getint('ShootDiscretize', 12)
    self.server, self.client = None, None
    # obs: (width, height, channel)
    self.observation_space = spaces.Box(
        shape=(self.width, self.height, 3), low=0, high=255)
    # actions: (accelerate, shoot, ability)
    self.action_space = spaces.MultiDiscrete(
        [self.acc_disc, self.shoot_disc, 8])
    self.viewer = None
    self.obv = np.zeros((self.width, self.height, 3), dtype=np.uint8)

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, actions):
    def convert_to_radians(actions):
      acc_dir, shoot_dir, ability_type = actions
      acc_dir = acc_dir / self.acc_disc * 2 * math.pi
      shoot_dir = shoot_dir / self.shoot_disc * 2 * math.pi
      return acc_dir, shoot_dir, ability_type

    actions = convert_to_radians(actions)
    self.pipe_actions[0].send(actions)
    self.server.sync()
    self.obv, reward = self.pipe_obv_reward[1].recv()
    self.counter += 1
    done = self.counter >= self.total_steps
    return self.obv, reward, done, {}

  def reset(self):
    self.close()
    # thegame server
    self.server = Server(self.server_bin, self.port)
    self.server.start()
    sleep(2)

    # thegame client
    self.pipe_obv_reward = multiprocessing.Pipe()
    self.pipe_actions = multiprocessing.Pipe()
    self.client = multiprocessing.Process(
        target=Client.main,
        args=(
            'localhost:{}'.format(self.port),
            self.name,
            self.width,
            self.height,
            self.pipe_actions[1],
            self.pipe_obv_reward[0],
        ))
    self.client.start()
    sleep(2)

    self.counter = 0
    self.server.sync()
    self.obv, _ = self.pipe_obv_reward[1].recv()
    return self.obv

  def render(self, mode='human'):
    img = np.array(self.obv, dtype=np.uint8)
    if mode == 'rgb_array':
      return img
    elif mode == 'human':
      from gym.envs.classic_control import rendering
      if self.viewer is None:
        self.viewer = rendering.SimpleImageViewer()
      self.viewer.imshow(img)
      return self.viewer.isopen
    logger.warn('mode `{}` is not supported'.format(mode))

  def close(self):
    if self.viewer:
      self.viewer.close()
    if self.client:
      self.client.terminate()
    if self.server:
      self.server.terminate()
