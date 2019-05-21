from gym.envs.registration import register

register(
    id='thegame-v0',
    entry_point='gym_thegame.envs:ThegameEnv',
)
register(
    id='thegame-v1',
    entry_point='gym_thegame.envs:ThegameEnvV1',
)
register(
    id='thegame-train-v0',
    entry_point='gym_thegame.envs:ThegameTrainEnv',
)
