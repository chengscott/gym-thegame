from gym_thegame.envs.server import Server
from gym_thegame.envs.client import Client
from gym_thegame.envs.utils import (LazyFrames, parse_args, get_obs_space,
                                    get_to_state_fn, convert_to_radians)
import gym
from gym import logger, spaces
from collections import deque
import math
import multiprocessing
from time import sleep
import numpy as np


class ThegameEnv(gym.Env):
  """thegame environment
  """
  metadata = {'render.modes': ['human', 'rgb_array']}

  def __init__(self):
    self.args = parse_args()
    self.viewer = None
    self.server, self.client = None, None
    # observation
    self.obv = deque([], maxlen=self.args.total_frame)
    self.to_state_fn = get_to_state_fn[self.args.obv_type]
    # observation space
    self.observation_space = get_obs_space(self.args)
    # action space
    if self.args.acc_disc:
      self.action_space = spaces.MultiDiscrete(
          [self.args.shoot_disc, self.args.acc_disc + 1])
    else:
      self.action_space = spaces.Discrete(self.args.shoot_disc)

  def step(self, actions=None, move_to=None):
    """
    environment step
    """

    def rescale(reward):
      return np.clip(reward / 40, -10, 10)

    actions = convert_to_radians(actions, self.args)

    self.pipe_actions[0].send((actions, move_to))
    self.server.sync()
    game_state, reward = self.pipe_obv_reward[1].recv()
    obv, self.img = self.to_state_fn(game_state, self.args)

    # update environment timestep and check episode end
    if move_to == None:
      self.counter += 1
    done = self.counter >= self.args.total_steps
    if reward != 0:
      print('timestep', self.counter, 'reward', reward)
    self.obv.append(obv)
    obv = [
        self.obv[i]
        for i in range(0, self.args.total_frame, self.args.skip_frame)
    ]
    return LazyFrames(obv), rescale(reward), done, {}

  def reset(self):
    # terminate server and client if exist
    if self.client:
      self.client.terminate()
    if self.server:
      self.server.terminate()

    # create thegame server
    self.server = Server(self.args.server_bin, self.args.port)
    self.server.start()
    sleep(0.5)

    # create thegame client
    self.pipe_obv_reward = multiprocessing.Pipe()
    self.pipe_actions = multiprocessing.Pipe()
    self.client = multiprocessing.Process(target=Client.main,
                                          args=(
                                              'localhost:{}'.format(
                                                  self.args.port),
                                              self.args,
                                              self.pipe_actions[1],
                                              self.pipe_obv_reward[0],
                                          ))
    self.client.start()
    sleep(0.5)

    # receive initial observation
    self.server.sync()
    game_state, _ = self.pipe_obv_reward[1].recv()
    obv, self.img = self.to_state_fn(game_state, self.args)
    for _ in range(self.args.total_frame):
      self.obv.append(obv)

    # random init
    self.counter = 0
    random_x, random_y = np.random.random_sample([2]) * 3000 + 500
    for _ in range(300):
      self.step(move_to=(random_x, random_y))

    return self.step(move_to=(random_x, random_y))[0]

  def render(self, mode='human'):
    img = self.img
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
