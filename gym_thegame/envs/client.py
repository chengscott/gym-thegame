from thegame import HeadlessClient

def to_state():
  pass

class Client(HeadlessClient):
  def init(self, name, *pipe):
    self.name = name
    self.pipe_actions, self.pipe_obv_reward = pipes
    self.prev_score = 0
    self.reward = 0

  def action(self, hero, heroes, polygons, bullets):
    # send observation and reward
    self.observation = to_state(hero, heroes, polygons, bullets)
    self.reward = hero.score - self.prev_score
    self.prev_score = hero.score
    self.pipe_obv_reward.send((self.observation, self.reward))
    # receive actions
    actions = self.pipe_actions.recv()
    acc_dir, shoot_dir, ability_type = actions[0]
    print("recv actions:", actions)
    # perform actions
    self.accelerate(acc_dir)
    self.shoot(shoot_dir)
    self.level_up(ability_type)

  @classmethod
  def main(cls, remote, *args):
    self = cls(*args)
    self.remote = remote
    self.run()
