from gym_thegame.envs.utils import (LazyFrames, parse_args, get_obs_space,
                                    get_to_state_fn, convert_to_radians)
from thegame.experimental.gymbase import SinglePlayerEnv, GameState, Controls
from gym import logger, spaces
from collections import deque
import math
import numpy as np


class ThegameEnvV1(SinglePlayerEnv):
  def __init__(self, listen='localhost:50051'):
    self.args = parse_args()
    super().__init__(
        server_bin=self.args.server_bin,
        listen=f'localhost:{self.args.port}',
        total_steps=self.args.total_steps,
    )
    self.viewer = None
    # preserve environment state
    self.prev_enemies = {}
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

  def reset(self):
    self.step_num = 0
    self.server.reset()
    self.server.tick()
    self.game_state = self.client.fetch_state()
    return self.game_state_to_observation(self.game_state, reset=True)

  def action_to_controls(self, action):
    shoot_dir, acc_dir, ability_type = convert_to_radians(action, self.args)

    return Controls(
        accelerate=acc_dir is not None and acc_dir != 0,
        acceleration_direction=acc_dir,
        shoot=shoot_dir is not None,
        shoot_direction=shoot_dir,
    )

  def game_state_to_observation(self, gs: GameState, reset=False):
    game_state = gs.hero, gs.heroes, gs.polygons, gs.bullets
    obv, self.img = self.to_state_fn(game_state, self.args)
    if reset:
      for _ in range(self.args.total_frame):
        self.obv.append(obv)
      return LazyFrames([obv] * self.args.stack_frame)
    # update frame buffer
    self.obv.append(obv)
    obv = [
        self.obv[i]
        for i in range(0, self.args.total_frame, self.args.skip_frame)
    ]
    return LazyFrames(obv)

  def get_reward(self, prev, curr):
    def reward_scaling(enemy_cur, enemy_prev):
      health, max_health, exp = enemy_cur
      health_prev, _, _ = enemy_prev
      if health >= health_prev:
        return 0
      discount_factor = (health_prev -
                         health) / max_health * (1.5 - health / max_health)
      return exp * discount_factor / 2

    enemies = {}
    reward = 0
    for e in [*curr.heroes, *curr.polygons]:
      enemies[e.id] = (e.health, e.max_health, e.rewarding_experience)
    for e_id in enemies.keys() & self.prev_enemies.keys():
      reward += reward_scaling(enemies[e_id], self.prev_enemies[e_id])
    self.prev_enemies = enemies
    reward += (curr.hero.score - prev.hero.score) / 2

    return np.clip(reward / 40, -10, 10)

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
