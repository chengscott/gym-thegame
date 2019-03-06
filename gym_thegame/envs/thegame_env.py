import gym
from gym import logger, spaces
from gym.utils import seeding
from gym_thegame.envs.server import Server
from gym_thegame.envs.client import Client
import math
import multiprocessing
from time import sleep


class ThegameEnv(gym.Env):
  """thegame environment
  """
  WIDTH, HEIGHT = 160, 160
  TOTAL_STEPS = 10000
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.res_w': WIDTH,
      'video.res_h': HEIGHT,
  }

  def __init__(self, server='./thegame-server', port=50051, name='gym'):
    self.server_bin, self.port = server, port
    self.name = name
    self.server, self.client = None, None
    self.observation_space = spaces.Box(
        shape=(ThegameEnv.WIDTH, ThegameEnv.HEIGHT, 3), low=0, high=255)
    # shoot, accelerate, ability
    self.action_space = spaces.MultiDiscrete([12, 12, 8])

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, actions):
    def convert_to_radians(actions):
      acc_dir, shoot_dir, ability_type = actions
      acc_dir = acc_dir / 12 * 2 * math.pi
      shoot_dir = shoot_dir / 12 * 2 * math.pi
      return acc_dir, shoot_dir, ability_type

    actions = convert_to_radians(actions)
    self.pipe_actions[0].send(actions)
    self.server.sync()
    obv, reward = self.pipe_obv_reward[1].recv()
    self.counter += 1
    done = self.counter >= ThegameEnv.TOTAL_STEPS
    return obv, reward, done, {}

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
            ThegameEnv.WIDTH,
            ThegameEnv.HEIGHT,
            self.pipe_actions[1],
            self.pipe_obv_reward[0],
        ))
    self.client.start()
    sleep(2)

    self.counter = 0
    self.server.sync()
    obv, _ = self.pipe_obv_reward[1].recv()
    return obv

  def render(self, mode='human', close=False):
    pass

  def close(self):
    if self.client:
      self.client.terminate()
    if self.server:
      self.server.terminate()
