from thegame import HeadlessClient
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
    acc_dir, shoot_dir, ability_type = actions
    print("recv actions:", actions)
    # perform actions
    self.accelerate(acc_dir)
    self.shoot(shoot_dir)
    self.level_up(ability_type)

  def _to_state(self, hero, heroes, polygons, bullets):
    """
    green
    self hero, bullets:
    rgb(16, 79, 15) ~ rgb(119, 226, 118)

    blue
    3 Polygon:
    rgb(33, 47, 104) ~ rgb(96, 115, 196)
    4 Polygon:
    rgb(33, 23, 84) ~ rgb(112, 98, 188)
    5 Polygon:
    rgb(23, 63, 89) ~ rgb(89, 148, 186)

    red
    other heros, bullets:
    rgb(114, 11, 11) ~ rgb(239, 103, 103)
    """

    hero_color_range = [[16, 119], [79, 226], [15, 118]]
    polygon_color_range = [[], [], [], [[33, 96], [47, 115], [104, 196]],
                           [[33, 112], [23, 98], [84, 188]],
                           [[23, 89], [63, 148], [89, 186]]]
    other_color_range = [[[114, 239], [11, 103], [11, 103]]]

    def interpolate(color_range, health, max_health):
      min_value = color_range[0]
      max_value = color_range[1]
      return int(max_value - (max_value - min_value) * health / max_health)

    def draw_value(arr, value, center, radius, layer, hero_pos):
      x, y = center
      hx, hy = hero_pos
      x, y, hx, hy = x / 10, y / 10, hx / 10, hy / 10
      x += self.width / 2 - hx
      y += self.height / 2 - hy
      sx = int(max(0, x - radius))
      sy = int(max(0, y - radius))
      ex = int(min(self.width, x + radius))
      ey = int(min(self.height, y + radius))
      for i in range(sx, ex):
        for j in range(sy, ey):
          if (i - x)**2 + (j - y)**2 <= radius**2:
            arr[i, j, layer] = value
      return arr

    state = np.zeros((self.width, self.height, 3))
    for i in range(3):
      state = draw_value(
          state, interpolate(hero_color_range[i], hero.health,
                             hero.max_health), hero.position, hero.radius, i,
          hero.position)
      for p in polygons:
        state = draw_value(
            state,
            interpolate(polygon_color_range[p.edges][i], p.health,
                        p.max_health), p.position, p.radius, i, hero.position)
      for b in bullets:
        if b.id == hero.id:
          state = draw_value(
              state, interpolate(hero_color_range[i], b.health, b.max_health),
              b.position, b.radius, i, hero.position)
        else:
          state = draw_value(
              state, interpolate(other_color_range[i], b.health, b.max_health),
              b.position, b.radius, i, hero.position)
      for h in heroes:
        state = draw_value(
            state, interpolate(other_color_range[i], h.health, h.max_health),
            h.position, h.radius, i, hero.position)

    return state

  @classmethod
  def main(cls, remote, *args):
    self = cls(*args)
    self.remote = remote
    self.run()
