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

    def draw(state, obj, color_range):
      """draw `obj` relative to current hero position"""
      color = interp(color_range, obj.health / obj.max_health)
      hx, hy = hero.position
      x, y = obj.position
      x, y, hx, hy = x / 10, y / 10, hx / 10, hy / 10
      radius = obj.radius
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

    state = np.zeros((self.width, self.height, 3))
    draw(state, hero, hero_color)
    for polygon in polygons:
      draw(state, polygon, polygon_color[polygon.edges])
    for bullet in bullets:
      if bullet.id == hero.id:
        draw(state, bullet, hero_color)
      else:
        draw(state, bullet, other_color)
    for hero_ in heroes:
      draw(state, hero_, other_color)

    return state

  @classmethod
  def main(cls, remote, *args):
    self = cls(*args)
    self.remote = remote
    self.run()
