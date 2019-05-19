import gym
from gym import error, spaces, utils
from gym.utils import seeding
import configparser
from collections import deque
import random
import math
import numpy as np
import sys
from baselines.common.atari_wrappers import LazyFrames
from gym_thegame.envs.utils import get_to_state_fn


class Obj:
  def __init__(self,
               position,
               radius,
               health=1000,
               max_health=1000,
               id=0,
               owner_id=0,
               edge=4,
               body_damage=40,
               reward=360,
               move=(0, 0)):
    self.id = id
    self.owner_id = owner_id
    self.position = position
    self.radius = radius
    self.health = health
    self.max_health = max_health
    self.edges = edge
    self.body_damage = body_damage
    self.rewarding_experience = reward
    self.cooldown = 0  # for hero
    self.duration = 120  # for bullet
    self.move = move  # for bullet


def parse_args(filename='thegame.cfg'):
  """
  TotalSteps: episode game length, -1 means game end if shot
  SkipFrame: skipped frame between two stacked frames. (0 if no skip frame)
  SkackFrame: number of stacked frames
  PolyGenDirs: polygons possible generate directions, -1 means uniform 0 ~ 2pi
  PolyDirs: polygons be generated directions per game
  PolyGenRange: polygons generate range divider. 8 means 2pi / 8
  PolyGenNum: polygons generate number
  PolyShootableNum: polygons generate number of must shootable
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
      'cool_down': ('CoolDown', 15),
      'bullet_speed': ('BulletSpeed', 30),
      'poly_gen_dirs': ('PolyGenDirs', -1),
      'poly_dirs': ('PolyDirs', 1),
      'poly_gen_range': ('PolyGenRange', 5),
      'poly_gen_num': ('PolyGenNum', 15),
      'poly_shootable_num': ('PolyShootableNum', 7)
  }
  # parse cfg
  config = configparser.ConfigParser()
  config.read(filename)
  cfg = config['DEFAULT']
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


class ThegameTrainEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array']}

  def __init__(self):
    self.args = parse_args()

    ### training agent ###
    self.obv = deque([], maxlen=self.args.total_frame)
    self.viewer = None

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
    if self.args.shoot_disc == -1:
      self.action_space = spaces.Box(shape=(1, ),
                                     low=-1,
                                     high=1,
                                     dtype=np.float32)
    else:
      self.action_space = spaces.Discrete(self.args.shoot_disc)

    ### training environment ###
    self.reset_counter = 0

    # internal state
    self.env_state = None, [], [], []
    self.to_state_fn = get_to_state_fn[self.args.obv_type]

    # reset jump angle: (counter * multiply) % gen_directions
    for m in range(self.args.poly_gen_dirs // 4, self.args.poly_gen_dirs):
      if math.gcd(m, self.args.poly_gen_dirs) == 1:
        self.multiply = m
        break

  def step(self, action):
    """
    env step function
    obv, reward, done, info = step(action)
    """
    if self.args.shoot_disc == -1:
      shoot_dir = action * np.pi
    else:
      shoot_dir = action / self.args.shoot_disc * 2 * np.pi

    def collide(obj1, obj2):
      """
      check if obj1 and obj2 collide with each other
      """
      x1, y1 = obj1.position
      x2, y2 = obj2.position
      r = max(obj1.radius, obj2.radius)
      if (x1 - x2)**2 + (y1 - y2)**2 <= r**2:
        return True
      return False

    # load env internal state
    hero, heros, polygons, bullets = self.env_state

    ### handle shooting bullet ###
    if hero.cooldown == 0:
      move = self.args.bullet_speed * math.cos(
          shoot_dir), self.args.bullet_speed * math.sin(shoot_dir)
      bullets.append(
          Obj(position=hero.position,
              health=40,
              max_health=40,
              radius=10,
              move=move,
              body_damage=12))
      hero.cooldown = self.args.cool_down
    else:
      hero.cooldown -= 1

    ### handle bullet shot target ###
    reward = 0
    for b in bullets:
      if b.duration == 0 or b.health <= 0:
        bullets.remove(b)
      else:
        b.duration -= 1
        x, y = b.position
        dx, dy = b.move
        b.position = x + dx, y + dy
        for p in polygons:
          if collide(p, b):
            p.health -= b.body_damage
            b.health -= p.body_damage
            if p.health <= 0:
              polygons.remove(p)
              reward += p.rewarding_experience / 2
            else:
              reward += p.rewarding_experience * b.body_damage / p.max_health * (
                  1.5 - p.health / p.max_health) / 2

    # save env internal state
    self.env_state = hero, heros, polygons, bullets
    obv, self.img = self.to_state_fn(self.env_state, self.args)

    # create stacked frames
    self.obv.append(obv)
    obv = [
        self.obv[i]
        for i in range(0, self.args.total_frame, self.args.skip_frame)
    ]

    ### handle game episode end ###
    if self.args.total_steps == -1:
      # game end if shot
      if reward > 0:
        reward = 40
        done = True
      else:
        reward = -20
        done = False
    else:
      # game end if reach total game steps
      self.counter += 1
      done = self.counter >= self.args.total_steps
      if reward != 0:
        print('timestep', self.counter, 'reward', reward)
        sys.stdout.flush()

    return LazyFrames(obv), np.clip(reward / 40, -10, 10), done, {}
    #return np.concatenate(obv, axis=-1), np.clip(reward / 40, -10, 10), done, {}

  def reset(self):
    """
    environment reset

    1. generate polygons
    2. initialize internal states
    3. return initial stack frames
    """

    # random init hero position
    hx, hy = random.randint(300, 4700), random.randint(300, 3700)
    polys = []
    thetas = []

    ### polygons generating directions ###
    # poly_gen_dirs: all possible generating directions number
    # poly_dirs:     number of directions polygons be genrated per episode
    if self.args.poly_gen_dirs == -1:
      # uniform random
      for _ in range(self.args.poly_dirs):
        thetas.append(random.uniform(0, 2 * np.pi))
    else:
      # first direction = (counter * multiply) % poly_gen_dirs
      theta = (int(self.reset_counter * self.multiply) %
               self.args.poly_gen_dirs) / self.args.poly_gen_dirs * 2 * np.pi
      self.reset_counter += 1
      thetas.append(theta)

      # uniform random for lefting poly_dirs
      for _ in range(1, self.args.poly_dirs):
        thetas.append(
            random.randint(0, self.args.poly_gen_dirs - 1) /
            self.args.poly_gen_dirs * 2 * np.pi)

    # generate polygons using polygons generating directions
    total_reward = 0
    for theta in thetas:
      for i in range(self.args.poly_gen_num):
        if i < self.args.poly_shootable_num:
          # must shootable for poly_gen_range >= 120
          dtheta = random.uniform(-1, 1) * 2 * math.pi / 120 / 2
        else:
          dtheta = random.uniform(
              -1, 1) * 2 * math.pi / self.args.poly_gen_range / 2
        R = random.uniform(100, 800)
        e = random.randint(3, 5)  # edge
        e2r = {3: 20, 4: 20, 5: 25}  # radius
        e2bd = {3: 10, 4: 20, 5: 40}  # body_damage
        e2re = {3: 10, 4: 60, 5: 360}  # reward
        e2h = {3: 100, 4: 300, 5: 1000}  # health
        total_reward += e2re[e]
        # position = ( R * cos(theta), R * sing(theta) )
        dx, dy = math.cos(theta + dtheta) * R, math.sin(theta + dtheta) * R
        if 0 < hx + dx < 5000 and 0 < hy + dy < 4000:
          polys.append(
              Obj(position=(hx + dx, hy + dy),
                  radius=e2r[e],
                  edge=e,
                  health=e2h[e],
                  max_health=e2h[e],
                  body_damage=e2bd[e],
                  reward=e2re[e]))

    print(f'reset with hero position ({hx}, {hy})')
    print(f'reward {total_reward / 40} for killing all polygons')
    print(f'thetas {thetas} for generating polygons')
    sys.stdout.flush()

    # initial hero and env internal state
    hero = Obj(position=(hx, hy), radius=30)
    heros = []
    polygons = polys
    bullets = []
    self.env_state = hero, heros, polygons, bullets
    obv, self.img = self.to_state_fn(self.env_state, self.args)
    self.counter = 0

    for _ in range(self.args.total_frame):
      self.obv.append(obv)

    return LazyFrames([obv] * self.args.stack_frame)
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
