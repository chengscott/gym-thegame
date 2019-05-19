from gym.envs.registration import register

register(
    id='thegame-v0',
    entry_point='gym_thegame.envs:ThegameEnv',
)
register(
    id='thegame-simple-v0',
    entry_point='gym_thegame.envs:ThegameSimpleEnv',
)
register(
    id='thegame-train-v0',
    entry_point='gym_thegame.envs:ThegameTrainEnv',
)
