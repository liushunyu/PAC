import os
from datetime import datetime

from utils.default_env import get_env
from utils.default_model import get_model
from utils.callback import Grid2OpTensorboardCallback


def train(args, seed, log_dir):
    env = get_env(args, seed)

    model, total_timesteps = get_model(env, args, seed, log_dir)

    model.learn(total_timesteps=total_timesteps, callback=Grid2OpTensorboardCallback())

    model.save(os.path.join(log_dir, 'final_model'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='l2rpn_case14_sandbox')
    # parser.add_argument('--env_id', type=str, default='l2rpn_wcci_2020')
    parser.add_argument('--algo', type=str, default='dqn')
    parser.add_argument('--mode', type=str, default='gnn')
    parser.add_argument('--use_hazard', action="store_true", default=False)
    parser.add_argument('--hiera_use_low_feature', action="store_true", default=False)
    parser.add_argument('--hazard_ratio', '-hr', type=float, default=0.9)
    parser.add_argument('--seed', type=str, default='[365]')
    parser.add_argument('--exp', type=str, default='debug')
    parser.add_argument('--info', type=str, default='')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    seed = list(map(int, args.seed.strip('[]').split(',')))

    log_datetime = datetime.now()

    for i in range(len(seed)):
        log_exp_name = 'debug'
        if args.exp == 'run':
            log_exp_name = args.env_id + '_' + args.algo + '_' + args.mode

            if args.algo == 'htac' and args.use_hazard:
                log_exp_name += '_hr' + str(args.hazard_ratio)

            if args.algo == 'htac' and args.mode == 'gnn' and args.hiera_use_low_feature:
                log_exp_name += '_low'

            log_exp_name += '_s' + str(seed[i])

        if args.info != '':
            log_exp_name = log_exp_name + '_' + args.info
        log_date_dir = os.path.join('exp_v1', log_datetime.strftime('%Y-%m-%d_') + log_exp_name)
        log_dir = os.path.join(log_date_dir, log_datetime.strftime('%H-%M-%S'))

        train(args, seed[i], log_dir)

