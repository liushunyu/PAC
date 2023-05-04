from stable_baselines3 import PPO, A2C, DQN, HTAC
from stable_baselines3.common.utils import linear_schedule
import torch as th


def get_model(env, args, seed, log_dir):
    model = None
    total_timesteps = None

    if args.env_id[:5] == 'l2rpn':
        total_timesteps = int(1e5)
        if args.algo == 'htac':
            if args.mode == 'mlp':
                policy_kwargs = dict(
                    hiera_arch=[2, int((env.action_space.n - 1) / 2), 2],
                    hiera_switch_action_top_down=True,
                    hiera_use_low_feature=args.hiera_use_low_feature
                )
                model = HTAC('MlpPolicy', env, learning_rate=linear_schedule(2.5e-4), n_steps=128, batch_size=64, n_epochs=10,
                             gamma=0.99, gae_lambda=0.95, clip_range=linear_schedule(0.2), clip_range_vf=None, ent_coef=0.0,
                             vf_coef=0.5, max_grad_norm=0.5, target_kl=None, tensorboard_log=log_dir, create_eval_env=False,
                             policy_kwargs=policy_kwargs, verbose=1, seed=seed, use_hazard=args.use_hazard)
            if args.mode == 'gnn':
                policy_kwargs = dict(
                    hiera_arch=[2, int((env.action_space.n - 1) / 2), 2],
                    hiera_switch_action_top_down=True,
                    hiera_use_low_feature=args.hiera_use_low_feature
                )
                model = HTAC('GnnPolicy', env, learning_rate=linear_schedule(2.5e-4), n_steps=128, batch_size=64, n_epochs=10,
                             gamma=0.99, gae_lambda=0.95, clip_range=linear_schedule(0.2), clip_range_vf=None, ent_coef=0.0,
                             vf_coef=0.5, max_grad_norm=0.5, target_kl=None, tensorboard_log=log_dir, create_eval_env=False,
                             policy_kwargs=policy_kwargs, verbose=1, seed=seed, use_hazard=args.use_hazard)
        if args.algo == 'ppo':
            if args.mode == 'mlp':
                model = PPO('MlpPolicy', env, learning_rate=linear_schedule(2.5e-4), n_steps=128, batch_size=64, n_epochs=10,
                            gamma=0.99, gae_lambda=0.95, clip_range=linear_schedule(0.2), clip_range_vf=None, ent_coef=0.0,
                            vf_coef=0.5, max_grad_norm=0.5, target_kl=None, tensorboard_log=log_dir, create_eval_env=False,
                            policy_kwargs=None, verbose=1, seed=seed)
            if args.mode == 'gnn':
                model = PPO('GnnPolicy', env, learning_rate=linear_schedule(2.5e-4), n_steps=128, batch_size=64, n_epochs=10,
                            gamma=0.99, gae_lambda=0.95, clip_range=linear_schedule(0.2), clip_range_vf=None, ent_coef=0.0,
                            vf_coef=0.5, max_grad_norm=0.5, target_kl=None, tensorboard_log=log_dir, create_eval_env=False,
                            policy_kwargs=None, verbose=1, seed=seed)
        if args.algo == 'a2c':
            if args.mode == 'mlp':
                model = A2C('MlpPolicy', env, learning_rate=linear_schedule(7e-4), n_steps=5, gamma=0.99,
                            gae_lambda=1.0, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, rms_prop_eps=1e-05, use_rms_prop=True,
                            normalize_advantage=False, tensorboard_log=log_dir, create_eval_env=False,
                            policy_kwargs=None, verbose=1, seed=seed)
            if args.mode == 'gnn':
                model = A2C('GnnPolicy', env, learning_rate=linear_schedule(7e-4), n_steps=5, gamma=0.99,
                            gae_lambda=1.0, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, rms_prop_eps=1e-05, use_rms_prop=True,
                            normalize_advantage=False, tensorboard_log=log_dir, create_eval_env=False,
                            policy_kwargs=None, verbose=1, seed=seed)
        if args.algo == 'dqn':
            if args.mode == 'mlp':
                policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[300, 300, 300, 300], use_dueling=True)
                model = DQN('MlpPolicy', env, learning_rate=linear_schedule(1e-4), buffer_size=100000, batch_size=64, gamma=0.99, tau=0.01,
                            learning_starts=int(total_timesteps / 5), target_update_interval=256, train_freq=256, gradient_steps=1,
                            exploration_fraction=0.2, exploration_initial_eps=0.4, exploration_final_eps=1.0 / (7.0 * 288.0),
                            optimize_memory_usage=False, tensorboard_log=log_dir, create_eval_env=False,
                            policy_kwargs=policy_kwargs, verbose=1, seed=seed, use_double=True)
            if args.mode == 'gnn':
                policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[300, 300, 300, 300], use_dueling=True)
                model = DQN('GnnPolicy', env, learning_rate=linear_schedule(1e-4), buffer_size=100000, batch_size=64, gamma=0.99, tau=0.01,
                            learning_starts=int(total_timesteps / 5), target_update_interval=256, train_freq=256, gradient_steps=1,
                            exploration_fraction=0.2, exploration_initial_eps=0.4, exploration_final_eps=1.0 / (7.0 * 288.0),
                            optimize_memory_usage=False, tensorboard_log=log_dir, create_eval_env=False,
                            policy_kwargs=policy_kwargs, verbose=1, seed=seed, use_double=True)
        if args.algo == 'sdqn':
            if args.mode == 'mlp':
                policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[300, 300, 300, 300], use_dueling=False)
                model = DQN('MlpPolicy', env, learning_rate=linear_schedule(1e-4), buffer_size=100000, batch_size=64, gamma=0.99, tau=0.01,
                            learning_starts=int(total_timesteps / 5), target_update_interval=256, train_freq=256, gradient_steps=1,
                            exploration_fraction=0.2, exploration_initial_eps=0.4, exploration_final_eps=1.0 / (7.0 * 288.0),
                            optimize_memory_usage=False, tensorboard_log=log_dir, create_eval_env=False,
                            policy_kwargs=policy_kwargs, verbose=1, seed=seed, use_double=False)
            if args.mode == 'gnn':
                policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[300, 300, 300, 300], use_dueling=False)
                model = DQN('GnnPolicy', env, learning_rate=linear_schedule(1e-4), buffer_size=100000, batch_size=64, gamma=0.99, tau=0.01,
                            learning_starts=int(total_timesteps / 5), target_update_interval=256, train_freq=256, gradient_steps=1,
                            exploration_fraction=0.2, exploration_initial_eps=0.4, exploration_final_eps=1.0 / (7.0 * 288.0),
                            optimize_memory_usage=False, tensorboard_log=log_dir, create_eval_env=False,
                            policy_kwargs=policy_kwargs, verbose=1, seed=seed, use_double=False)
        if args.algo == 'ddqn':
            if args.mode == 'mlp':
                policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[300, 300, 300, 300], use_dueling=False)
                model = DQN('MlpPolicy', env, learning_rate=linear_schedule(1e-4), buffer_size=100000, batch_size=64, gamma=0.99, tau=0.01,
                            learning_starts=int(total_timesteps / 5), target_update_interval=256, train_freq=256, gradient_steps=1,
                            exploration_fraction=0.2, exploration_initial_eps=0.4, exploration_final_eps=1.0 / (7.0 * 288.0),
                            optimize_memory_usage=False, tensorboard_log=log_dir, create_eval_env=False,
                            policy_kwargs=policy_kwargs, verbose=1, seed=seed, use_double=True)
            if args.mode == 'gnn':
                policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[300, 300, 300, 300], use_dueling=False)
                model = DQN('GnnPolicy', env, learning_rate=linear_schedule(1e-4), buffer_size=100000, batch_size=64, gamma=0.99, tau=0.01,
                            learning_starts=int(total_timesteps / 5), target_update_interval=256, train_freq=256, gradient_steps=1,
                            exploration_fraction=0.2, exploration_initial_eps=0.4, exploration_final_eps=1.0 / (7.0 * 288.0),
                            optimize_memory_usage=False, tensorboard_log=log_dir, create_eval_env=False,
                            policy_kwargs=policy_kwargs, verbose=1, seed=seed, use_double=True)

    assert model, "model not exist"
    return model, total_timesteps
