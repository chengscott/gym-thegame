from gym.envs.registration import register

register(
    id='thegame-v0',
    entry_point='gym_thegame.envs:ThegameEnv',
)
register(
    id='thegame-simple-v0',
    entry_point='gym_thegame.envs:ThegameSimpleEnv',
)