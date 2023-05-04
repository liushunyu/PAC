from stable_baselines3.common.env_util import make_grid2op_env
from utils.wrappers import wrap_exp
from utils.rewards import LoadGenRatioReward
from utils.action import PlayableAction
from stable_baselines3.common.vec_env import DummyVecEnv

import os
from grid2op.MakeEnv.UserUtils import change_local_dir
change_local_dir(os.path.join(os.environ['CODE_HOME'], '/PAC/data_grid2op'))

try:
    from lightsim2grid.LightSimBackend import LightSimBackend
    backend = LightSimBackend()
except:
    from grid2op.Backend import PandaPowerBackend
    backend = PandaPowerBackend()

print(backend)


def get_env(args, seed):
    env = None
    n_envs = 8
    if args.algo == 'dqn' or args.algo == 'sdqn' or args.algo == 'ddqn':
        n_envs = 1
    if args.env_id[:5] == 'l2rpn':
        wrapper_class = wrap_exp(args.env_id, args.mode)
        env_kwargs = {'reward_class': LoadGenRatioReward, 'action_class': PlayableAction,
                      'backend': backend}
        wrapper_kwargs = {'hazard_ratio': args.hazard_ratio}
        env = make_grid2op_env(args.env_id, n_envs=n_envs, seed=seed, wrapper_class=wrapper_class,
                               wrapper_kwargs=wrapper_kwargs, env_kwargs=env_kwargs, vec_env_cls=DummyVecEnv)

    assert env, "env not exist"

    return env
