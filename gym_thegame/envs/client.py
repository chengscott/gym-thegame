from thegame import HeadlessClient
from gym import logger
import numpy as np


class Client(HeadlessClient):
  def __init__(self, name, width, height, *pipes):
    self.name = name
    self.width, self.height = width, height
    self.pipe_actions, self.pipe_obv_reward = pipes
    self.prev_score = 0
    self.reward = 0

  def action(self, hero, heroes, polygons, bullets):
    # send observation and reward
    self.observation = self._to_state(hero, heroes, polygons, bullets)
    self.reward = hero.score - self.prev_score
    self.prev_score = hero.score
    self.pipe_obv_reward.send((self.observation, self.reward))
    # receive actions
    actions = self.pipe_actions.recv()
    logger.info('recv actions', actions)
    # perform actions
    acc_dir, shoot_dir, ability_type = actions
    self.accelerate(acc_dir)
    self.shoot(shoot_dir)
    self.level_up(ability_type)

  def _to_state(self, hero, heroes, polygons, bullets):
    def interp(color_range, ratio):
      return (int(end - (end - start) * ratio)
              for start, end in zip(*color_range))

    def draw_boundary(state):
      hx, hy = hero.position
      hx, hy = hx / 10, hy / 10
      max_map_x, max_map_y = 5000 / 10, 4000 / 10
      up_bound = int(max(0, self.height / 2 - hy))
      left_bound = int(max(0, self.width / 2 - hx))
      down_bound = int(min(self.height, self.height / 2 + (max_map_y - hy)))
      right_bound = int(min(self.width, self.width / 2 + (max_map_x - hx)))
      state[0:left_bound, :, :] = 0
      state[:, 0:up_bound, :] = 0
      state[right_bound:self.width, :, :] = 0
      state[:, down_bound:self.height, :] = 0
      return state

    def draw(state, obj, color_range):
      """draw `obj` relative to current hero position"""
      color = interp(color_range, obj.health / obj.max_health)
      hx, hy = hero.position
      x, y = obj.position
      x, y, hx, hy = x / 10, y / 10, hx / 10, hy / 10
      radius = obj.radius / 10
      x += self.width / 2 - hx
      y += self.height / 2 - hy
      lx, rx = int(max(0, x - radius)), int(min(self.width, x + radius))
      ly, ry = int(max(0, y - radius)), int(min(self.height, y + radius))
      for channel, value in enumerate(color):
        for i in range(lx, rx):
          for j in range(ly, ry):
            if (i - x)**2 + (j - y)**2 <= radius**2:
              state[i, j, channel] = value
      return state

    #
    # color range: (start rgb, end rgb)
    #
    # hero, own bullets: green
    hero_color = ((16, 79, 15), (119, 226, 118))
    # polygons: blue
    polygon_color = {
        3: ((33, 47, 104), (96, 115, 196)),
        4: ((33, 23, 84), (112, 98, 188)),
        5: ((23, 63, 89), (89, 148, 186)),
    }
    # heroes, other bullets: red
    other_color = ((114, 11, 11), (239, 103, 103))

    state = np.full((self.width, self.height, 3), 255)
    draw_boundary(state)

    for polygon in polygons:
      draw(state, polygon, polygon_color[polygon.edges])
    for bullet in bullets:
      if bullet.owner_id == hero.id:
        draw(state, bullet, hero_color)
      else:
        draw(state, bullet, other_color)
    for hero_ in heroes:
      draw(state, hero_, other_color)
    draw(state, hero, hero_color)

    return state

  @classmethod
  def main(cls, remote, *args):
    self = cls(*args)
    self.remote = remote
    self.run()
