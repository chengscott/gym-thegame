# gym-thegame

## Install with thegame

- use the [sync version](https://github.com/chengscott/thegame) of thegame server

```shell
pip install --upgrade git+https://github.com/openai/gym.git
pip install --upgrade git+https://github.com/openai/baselines.git
pip install --upgrade git+https://github.com/chengscott/gym-thegame.git
pip install --upgrade git+https://github.com/chengscott/thegame.git#subdirectory=client/python
```

## Run with baselines

```shell
python -m baselines.run --extra_import gym_thegame --env=thegame-v0 --alg=ppo2 --network=mlp --num_timesteps=2e7
```

- view as an audience

```shell
python -m thegame.gui.audience localhost:50051
```

