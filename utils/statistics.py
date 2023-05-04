import grid2op
import numpy as np
import pickle
from utils.rewards import RedispReward, LoadGenRatioReward, NormalizationRedispReward
from utils.action import PlayableAction
try:
    from lightsim2grid.LightSimBackend import LightSimBackend
    backend = LightSimBackend()
except:
    from grid2op.Backend import PandaPowerBackend
    backend = PandaPowerBackend()


DATA_SPLIT = {
    'l2rpn_case14_sandbox': (list(range(0, 40 * 26, 40)), list(range(1, 100 * 10 + 1, 100))),
    'l2rpn_wcci_2020': ([17, 240, 494, 737, 990, 1452, 1717, 1942, 2204, 2403, 19, 242, 496, 739, 992, 1454, 1719, 1944, 2206, 2405, 230, 301, 704, 952, 1008, 1306, 1550, 1751, 2110, 2341, 2443, 2689], list(range(2880, 2890))),
}


def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()


def is_day_done(env, day_id):
    if (env.chronics_handler.data.current_index - day_id * 288) % 864 == 0:
        return True
    else:
        return False


def do_nothing(env, chronics, total_day, dn_action):

    obs_total = []
    dn_alive_total = []
    Redisp_return_total = []
    NormalizationRedisp_return_total = []
    LoadGenRatio_return_total = []

    for chronic_id in chronics:
        for day_id in range(total_day):
            env.set_id(chronic_id)
            env.reset()
            env.fast_forward_chronics(day_id * 288)
            obs = env.get_obs()
            obs_total.append(obs.to_vect())

            done = False
            alive_cnt = 0
            Redisp_return = 0
            NormalizationRedisp_return = 0
            LoadGenRatio_return = 0

            while not done:
                obs, reward, done, info = env.step(dn_action)
                obs_total.append(obs.to_vect())
                done = is_day_done(env, day_id) if done is False else done
                alive_cnt += 1

                Redisp_return += info['rewards']['RedispReward']
                NormalizationRedisp_return += info['rewards']['NormalizationRedispReward']
                LoadGenRatio_return += info['rewards']['LoadGenRatioReward']

            print("chronic_id: {} day_id: {} alive_cnt: {}".format(chronic_id, day_id, alive_cnt))
            print("Redisp_return: {} NormalizationRedisp_return: {} LoadGenRatio_return: {}".format(Redisp_return, NormalizationRedisp_return, LoadGenRatio_return))
            dn_alive_total.append(alive_cnt)
            Redisp_return_total.append(Redisp_return)
            NormalizationRedisp_return_total.append(NormalizationRedisp_return)
            LoadGenRatio_return_total.append(LoadGenRatio_return)

    obs_mean = np.mean(obs_total, axis=0)
    obs_std = np.std(obs_total, axis=0)

    print("mean dn_alive_total: {}".format(np.mean(dn_alive_total)))
    print("mean Redisp_return_total: {}".format(np.mean(Redisp_return_total)))
    print("mean NormalizationRedisp_return_total: {}".format(np.mean(NormalizationRedisp_return_total)))
    print("mean LoadGenRatio_return_total: {}".format(np.mean(LoadGenRatio_return_total)))

    save_variable(obs_mean, './pretrain/obs_mean.pt')
    save_variable(obs_std, './pretrain/obs_std.pt')
    save_variable(dn_alive_total, './pretrain/dn_alive_total.pt')
    save_variable(Redisp_return_total, './pretrain/Redisp_return_total.pt')
    save_variable(NormalizationRedisp_return_total, './pretrain/NormalizationRedisp_return_total.pt')
    save_variable(LoadGenRatio_return_total, './pretrain/LoadGenRatio_return_total.pt')


def statistics(env_id):
    env = grid2op.make(env_id, reward_class=LoadGenRatioReward, action_class=PlayableAction, backend=backend,
                       other_rewards={'RedispReward': RedispReward, 'NormalizationRedispReward': NormalizationRedispReward,
                                      'LoadGenRatioReward': LoadGenRatioReward})
    train_chronics, test_chronics = DATA_SPLIT[env_id]
    train_total_day = int(env.chronics_handler.data.max_iter / 288) - 2

    dn_action = env.action_space({})
    do_nothing(env, train_chronics, train_total_day, dn_action)

