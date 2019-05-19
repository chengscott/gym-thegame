import gym
from gym import logger, spaces
from gym.utils import seeding
from gym_thegame.envs.server import Server
from gym_thegame.envs.client import Client
import configparser
from collections import deque
import math
import multiprocessing
from time import sleep
import numpy as np
import sys
from baselines.common.atari_wrappers import LazyFrames
from gym_thegame.envs.utils import get_to_state_fn


def parse_args(filename='thegame.cfg'):
  """
  SkipFrame: skipped frame between two stacked frames. (0 if no skip frame)
  SkackFrame: number of stacked frames
  """

  class Args:
    pass

  args = Args()
  # [Environment] section arguments
  env_args = {
      'width': ('Width', 80),
      'height': ('Height', 80),
      'total_steps': ('TotalSteps', 2048),
      'acc_disc': ('AccelerationDistcretize', 0),
      'shoot_disc': ('ShootDiscretize', 5),
      'skip_frame': ('SkipFrame', 0),
      'stack_frame': ('StackFrame', 1),
      'bg_color': ('BGColor', 0),
      'boundary_color': ('BoundaryColor', 0),
  }
  # parse cfg
  config = configparser.ConfigParser()
  config.read(filename)
  cfg = config['DEFAULT']
  args.server_bin = cfg.get('ServerBinary', './thegame-server')
  args.port = cfg.getint('Port', 50051)
  args.name = cfg.get('Name', 'gym')
  if 'Environment' in config:
    cfg = config['Environment']
  args.obv_type = cfg.get('ObvType', 'gray')
  for k, v in env_args.items():
    setattr(args, k, cfg.getint(*v))
  # postprocessing
  args.skip_frame += 1
  args.quantize = 1600 / args.width
  args.total_frame = args.stack_frame * args.skip_frame
  return args


class ThegameEnv(gym.Env):
  """thegame environment
  """
  metadata = {'render.modes': ['human', 'rgb_array']}

  def __init__(self):
    self.args = parse_args()

    ### training agent ###
    self.viewer = None
    self.obv = deque([], maxlen=self.args.total_frame)

    # observation space
    if self.args.obv_type == 'gray':
      self.observation_space = spaces.Box(shape=(self.args.width,
                                                 self.args.height,
                                                 self.args.stack_frame),
                                          low=0,
                                          high=1)
    else:
      self.observation_space = spaces.Box(shape=(self.args.width,
                                                 self.args.height,
                                                 3 * self.args.stack_frame),
                                          low=-1,
                                          high=1)

    # action space
    if self.args.acc_disc:
      self.action_space = spaces.MultiDiscrete(
          [self.args.acc_disc, self.args.shoot_disc])
    else:
      self.action_space = spaces.Discrete(self.args.shoot_disc)

    ### training environment ###
    self.server, self.client = None, None
    # to state function
    self.to_state_fn = get_to_state_fn[self.args.obv_type]

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, actions=None, move_to=None):
    """
    environment step
    """

    def convert_to_radians(actions):
      """
      convert directions to radian
      input actions: (shoot_dir, acc_dir, ability_type)
      note: `acc_dir` and `ability_type` can be None
      return either:
        (shoot_dir, None, None)
        (shoot_dir, acc_dir, None)
        (shoot_dir, acc_dir, ability_type)
      """
      if actions == None:
        return None, None, None
      shoot_dir, acc_dir, ability_type = actions, None, None
      if self.args.acc_disc:
        shoot_dir, acc_dir, ability_type, *_ = *actions, None, None
        acc_dir = acc_dir / self.args.acc_disc * 2 * math.pi
      shoot_dir = shoot_dir / self.args.shoot_disc * 2 * math.pi
      return shoot_dir, acc_dir, ability_type

    def rescale(reward):
      return np.clip(reward / 40, -10, 10)
      return np.sign(reward)

    actions = convert_to_radians(actions)

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
    ## without random init
    #return np.concatenate([obv] * self.args.stack_frame, axis=-1)

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
