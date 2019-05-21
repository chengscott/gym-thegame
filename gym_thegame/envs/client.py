from thegame import HeadlessClient, Ability
from gym import logger
import numpy as np
import math
import sys


class Client(HeadlessClient):
  XMax, YMax = 5000, 4000

  def __init__(self, args, *pipes):
    self.name = args.name
    self.args = args
    self.pipe_actions, self.pipe_obv_reward = pipes

    # environment state preserve
    self.prev_score = 0
    self.prev_enemies = {}

  def action(self, hero, heroes, polygons, bullets):
    # reward calculate
    def reward_scaling(e_cur, e_prev):
      health, max_health, exp = e_cur
      health_prev, _, _ = e_prev
      if health >= health_prev:
        return 0
      discount_factor = (health_prev -
                         health) / max_health * (1.5 - health / max_health)
      return exp * discount_factor / 2

    enemies = {}
    reward = 0
    for e in [*heroes, *polygons]:
      enemies[e.id] = (e.health, e.max_health, e.rewarding_experience)
    for e_id in enemies.keys() & self.prev_enemies.keys():
      reward += reward_scaling(enemies[e_id], self.prev_enemies[e_id])

    reward += (hero.score - self.prev_score) / 2

    self.prev_score = hero.score
    self.prev_enemies = enemies
    self.pipe_obv_reward.send(((hero, heroes, polygons, bullets), reward))

    # receive actions
    actions, move_to = self.pipe_actions.recv()

    if move_to:
      self.accelerate_towards(*move_to)
      return

    logger.info('recv actions', actions)

    # perform actions
    shoot_dir, acc_dir, ability_type = actions

    if shoot_dir != None:
      self.shoot(shoot_dir)
    if acc_dir != None:
      self.accelerate(acc_dir)
    if ability_type != None:
      self.level_up(ability_type)

  @classmethod
  def main(cls, remote, *args):
    class Option:
      pass

    self = cls(*args)
    self.options = Option()
    self.options.remote = remote
    self.run()
