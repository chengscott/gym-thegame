import gym
from gym import error, spaces
from gym.utils import seeding
from gym_thegame.envs import Server, Client

class ThegameEnv(gym.Env):
  WIDTH, HEIGHT = 200, 100
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second': 30,
      'video.res_w': WIDTH,
      'video.res_h': HEIGHT,
  }

  def __init__(self, server, port=50051, name='gym'):
    self.observation_space = spaces.Box(
        shape=(WIDTH, HEIGHT, 3), low=0, high=255)
    self.action_space = spaces.Dict({
        'shoot': spaces.Discrete(12),
        'accelerate': spaces.Discrete(12),
        'ability': spaces.Discrete(8),
    })

    # thegame server
    self.server = Server(server, port)
    self.server.start()
    sleep(0.1)

    # thegame client
    self.pipe_obv_reward = multiprocessing.Pipe()
    self.pipe_actions = multiprocessing.Pipe()
    self.client = multiprocessing.Process(
        target=Client.main,
        args=(
            ':{}'.format(port),
            name,
            self.pipe_actions[1],
            self.pipe_obv_reward[0],
        ))
    self.client.start()
    sleep(0.1)

  def step(self, actions):
    self.pipe_actions[0].send(actions)
    self.server.sync()
    obv, reward = self.pipe_obv_reward[1].recv()
    # TODO: done
    return obv, np.array([reward]), np.array([done]), []

  def reset(self):
    self.server.sync()
    obv, _ = self.pipe_obv_reward[1].recv()
    return obv

  def render(self, mode='human', close=False):
    pass

  def close(self):
    self.client.terminate()
    self.server.terminate()

  def seed(self):
    pass
