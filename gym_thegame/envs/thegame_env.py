from gym import logger, spaces
from gym.utils import seeding
from gym_thegame.envs.client import Client
from thegame.experimental.gymbase import SinglePlayerEnv, Controls
import configparser
import math
import numpy as np


class ThegameEnv(SinglePlayerEnv):
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
    total_steps = cfg.getint('TotalSteps', 10000)
    super().__init__(
      server_bin=self.server_bin,
      listen=':{}'.format(self.port),
      total_steps=total_steps,
      client_name=self.name)
    self.acc_disc = cfg.getint('AccelerateDiscretize', 12)
    self.shoot_disc = cfg.getint('ShootDiscretize', 12)
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

  def action_to_controls(self, actions):
    def convert_to_radians(actions):
      acc_dir, shoot_dir, ability_type = actions
      acc_dir = acc_dir / self.acc_disc * 2 * math.pi
      shoot_dir = shoot_dir / self.shoot_disc * 2 * math.pi
      return acc_dir, shoot_dir, ability_type

    actions = convert_to_radians(actions)

    return Controls(
      shoot=True,
      shoot_direction=actions[1],
      accelerate=True,
      acceleration_direction=actions[0],
      level_up=[actions[2]]
    )

  _to_state = Client._to_state

  def game_state_to_observation(self, game_state):
    return self._to_state(
      hero=game_state.hero,
      heroes=game_state.heroes,
      polygons=game_state.polygons,
      bullets=game_state.bullets)

  def get_reward(self, prev_state, current_state):
    return current_state.hero.score - prev_state.hero.score

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
