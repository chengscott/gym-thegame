# gym-thegame

## Install with thegame

```shell
go get -u github.com/afg984/thegame/server/go/thegame
pip install --upgrade git+https://github.com/openai/gym.git
pip install --upgrade git+https://github.com/openai/baselines.git
pip install --upgrade git+https://github.com/chengscott/gym-thegame.git
pip install --upgrade git+https://github.com/afg984/thegame.git#subdirectory=client/python
```

## Run with baselines

```shell
python -m baselines.run --extra_import gym_thegame --env=thegame-v0 --alg=ppo2 --network=mlp --num_timesteps=2e7
```

- view as an audience

```shell
python -m thegame.gui.spectator localhost:50051 --smooth
```
