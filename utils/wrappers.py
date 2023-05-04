import numpy as np
import gym
from grid2op.Converter import IdToAct
from utils.action import init_all_actions
import dgl
import torch
import pickle
import networkx as nx
from stable_baselines3.common.spaces import Graph
import os


DATA_SPLIT = {
    'l2rpn_case14_sandbox': (list(range(0, 40 * 26, 40)), list(range(45, 100 * 10 + 1, 100))),
    'l2rpn_wcci_2020': ([17, 240, 494, 737, 990, 1452, 1717, 1942, 2204, 2403, 19, 242, 496, 739, 992, 1454, 1719, 1944, 2206, 2405, 230, 301, 704, 952, 1008, 1306, 1550, 1751, 2110, 2341, 2443, 2689], list(range(2880, 2890))),
}

all_attr_name = ['year', 'month', 'day', 'hour_of_day', 'minute_of_hour', 'day_of_week',
                 'prod_p', 'prod_q', 'prod_v', 'load_p', 'load_q', 'load_v', 'topo_vect',
                 'p_or', 'q_or', 'v_or', 'a_or', 'p_ex', 'q_ex', 'v_ex', 'a_ex', 'rho', 'line_status',
                 'timestep_overflow', 'time_before_cooldown_line', 'time_before_cooldown_sub',
                 'actual_dispatch', 'target_dispatch', 'time_next_maintenance', 'duration_next_maintenance']

line_or_attr_name = ['p_or', 'q_or', 'v_or', 'a_or',
                     'rho', 'line_status', 'timestep_overflow', 'time_before_cooldown_line',
                     'time_next_maintenance', 'duration_next_maintenance', 'topo_vect', 'time_before_cooldown_sub']

line_ex_attr_name = ['p_ex', 'q_ex', 'v_ex', 'a_ex',
                     'rho', 'line_status', 'timestep_overflow', 'time_before_cooldown_line',
                     'time_next_maintenance', 'duration_next_maintenance', 'topo_vect', 'time_before_cooldown_sub']

gen_attr_name = ['prod_p', 'prod_q', 'prod_v', 'topo_vect', 'time_before_cooldown_sub', 'actual_dispatch', 'target_dispatch']

load_attr_name = ['load_p', 'load_q', 'load_v', 'topo_vect', 'time_before_cooldown_sub']


def get_obs_subset(observation_space, subset_attr_name):
    size = 0
    idx = np.zeros(0, dtype=np.uint)
    for attr_name in subset_attr_name:
        begin, end, _ = observation_space.get_indx_extract(attr_name)
        idx = np.concatenate((idx, np.arange(begin, end, dtype=np.uint)))
        size += end - begin
    return idx, size


def get_graph(observation):
    node_num = observation.dim_topo

    feature_p = np.zeros(node_num, dtype=np.float32)
    feature_p[observation.gen_pos_topo_vect] = observation.prod_p
    feature_p[observation.load_pos_topo_vect] = observation.load_p
    feature_p[observation.line_or_pos_topo_vect] = observation.p_or
    feature_p[observation.line_ex_pos_topo_vect] = observation.p_ex

    feature_q = np.zeros(node_num, dtype=np.float32)
    feature_q[observation.gen_pos_topo_vect] = observation.prod_q
    feature_q[observation.load_pos_topo_vect] = observation.load_q
    feature_q[observation.line_or_pos_topo_vect] = observation.q_or
    feature_q[observation.line_ex_pos_topo_vect] = observation.q_ex

    feature_v = np.zeros(node_num, dtype=np.float32)
    feature_v[observation.gen_pos_topo_vect] = observation.prod_v
    feature_v[observation.load_pos_topo_vect] = observation.load_v
    feature_v[observation.line_or_pos_topo_vect] = observation.v_or
    feature_v[observation.line_ex_pos_topo_vect] = observation.v_ex

    feature_topo_vect_onehot_0 = np.zeros(node_num, dtype=np.float32)
    feature_topo_vect_onehot_1 = np.zeros(node_num, dtype=np.float32)
    feature_topo_vect_onehot_0[:] = observation.topo_vect
    feature_topo_vect_onehot_1[:] = 1 - observation.topo_vect

    feature_rho = np.zeros(node_num, dtype=np.float32)
    feature_rho[observation.line_or_pos_topo_vect] = observation.rho
    feature_rho[observation.line_ex_pos_topo_vect] = observation.rho

    feature_line_status_onehot_0 = np.zeros(node_num, dtype=np.float32)
    feature_line_status_onehot_1 = np.zeros(node_num, dtype=np.float32)
    feature_line_status_onehot_0[observation.line_or_pos_topo_vect] = observation.line_status
    feature_line_status_onehot_0[observation.line_ex_pos_topo_vect] = observation.line_status
    feature_line_status_onehot_1[observation.line_or_pos_topo_vect] = 1 - observation.line_status
    feature_line_status_onehot_1[observation.line_ex_pos_topo_vect] = 1 - observation.line_status

    feature_timestep_overflow = np.zeros(node_num, dtype=np.float32)
    feature_timestep_overflow[observation.line_or_pos_topo_vect] = observation.timestep_overflow
    feature_timestep_overflow[observation.line_ex_pos_topo_vect] = observation.timestep_overflow

    feature_time_before_cooldown_line = np.zeros(node_num, dtype=np.float32)
    feature_time_before_cooldown_line[observation.line_or_pos_topo_vect] = observation.time_before_cooldown_line
    feature_time_before_cooldown_line[observation.line_ex_pos_topo_vect] = observation.time_before_cooldown_line

    feature_time_before_cooldown_sub = np.zeros(node_num, dtype=np.float32)
    feature_time_before_cooldown_sub[observation.gen_pos_topo_vect] = observation.time_before_cooldown_sub[observation.gen_to_subid]
    feature_time_before_cooldown_sub[observation.load_pos_topo_vect] = observation.time_before_cooldown_sub[observation.load_to_subid]
    feature_time_before_cooldown_sub[observation.line_or_pos_topo_vect] = observation.time_before_cooldown_sub[observation.line_or_to_subid]
    feature_time_before_cooldown_sub[observation.line_ex_pos_topo_vect] = observation.time_before_cooldown_sub[observation.line_ex_to_subid]

    feature_time_next_maintenance = np.zeros(node_num, dtype=np.float32)
    feature_time_next_maintenance[observation.line_or_pos_topo_vect] = observation.time_next_maintenance
    feature_time_next_maintenance[observation.line_ex_pos_topo_vect] = observation.time_next_maintenance

    feature_duration_next_maintenance = np.zeros(node_num, dtype=np.float32)
    feature_duration_next_maintenance[observation.line_or_pos_topo_vect] = observation.duration_next_maintenance
    feature_duration_next_maintenance[observation.line_ex_pos_topo_vect] = observation.duration_next_maintenance

    node_feature = np.vstack([feature_p, feature_q, feature_v,
                              feature_topo_vect_onehot_0, feature_topo_vect_onehot_1,
                              feature_rho, feature_line_status_onehot_0, feature_line_status_onehot_1,
                              feature_timestep_overflow,
                              feature_time_next_maintenance]).T

    return node_feature, observation.connectivity_matrix()


def get_is_hazard(rhos, thermal_limits, hazard_ratio):
    for rho, thermal_limit in zip(rhos, thermal_limits):
        if (thermal_limit < 400.0 and rho >= hazard_ratio - 0.05) or rho >= hazard_ratio:
            return True
    return False


def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


class Grid2OpConfig(gym.Wrapper):
    def __init__(self, env, hazard_ratio, use_test):
        super(Grid2OpConfig, self).__init__(env)
        self.env.deactivate_forecast()
        self.action_space_converter = IdToAct(self.action_space)
        self.action_space_converter.init_converter(all_actions=init_all_actions(self.env))
        self.action_space = gym.spaces.Discrete(n=self.action_space_converter.n)

        self.hazard_ratio = hazard_ratio
        self.last_obs_is_hazard = False
        self.use_test = use_test

        train_chronics, test_chronics = DATA_SPLIT[self.env.env_name]
        if self.use_test:
            self.chronics = test_chronics
        else:
            self.chronics = train_chronics

        self.obs_mean = load_variavle(os.environ['CODE_HOME'] + '/PAC/data_grid2op/' + self.env.env_name + '/obs_mean.pt')
        self.obs_std = load_variavle(os.environ['CODE_HOME'] + '/PAC/data_grid2op/' + self.env.env_name + '/obs_std.pt')
        self.obs_std[self.obs_std < 1e-5] = 1
        self.obs_mean[sum(self.env.observation_space.shape[:20]):] = 0
        self.obs_std[sum(self.env.observation_space.shape[:20]):] = 1

        self.env.parameters.NO_OVERFLOW_DISCONNECTION = False
        self.env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED = 6
        self.env.parameters.NB_TIMESTEP_COOLDOWN_SUB = 0
        self.env.parameters.NB_TIMESTEP_COOLDOWN_LINE = 0
        self.env.parameters.HARD_OVERFLOW_THRESHOLD = 3
        self.env.parameters.NB_TIMESTEP_RECONNECTION = 1
        self.env.parameters.IGNORE_MIN_UP_DOWN_TIME = True
        self.env.parameters.ALLOW_DISPATCH_GEN_SWITCH_OFF = True
        self.env.parameters.ENV_DC = False
        self.env.parameters.FORECAST_DC = False
        self.env.parameters.MAX_SUB_CHANGED = 1
        self.env.parameters.MAX_LINE_STATUS_CHANGED = 1

        self.chronic_day_id = None

        if not self.use_test:
            self.total_day = int(self.env.chronics_handler.data.max_iter / 288 - 2)
            dn_alive_total = load_variavle(os.environ['CODE_HOME'] + '/PAC/data_grid2op/' + self.env.env_name + '/dn_alive_total.pt')
            self.dn_alive_priority = 1 - np.sqrt(np.array(dn_alive_total) / 864.)
            self.chronic_day_priority = (1 - np.sqrt(1. / 864.)) + 0.1 * self.dn_alive_priority
        else:
            self.total_day = 2

    def update_priority(self):
        chronic_day_id = self.chronic_id * self.total_day + self.day_id
        current_priority = 1 - np.sqrt((self.env.chronics_handler.data.current_index - self.day_id * 288) / 864.)
        self.chronic_day_priority[chronic_day_id] = 0.1 * self.dn_alive_priority[chronic_day_id] + current_priority

    def is_day_done(self):
        if (self.env.chronics_handler.data.current_index - self.day_id * 288) % 864 == 0:
            return True
        else:
            return False

    def reset(self, **kwargs):
        if self.use_test:
            if self.chronic_day_id is None:
                self.chronic_day_id = 0
            else:
                self.chronic_day_id = (self.chronic_day_id + 1) % (len(self.chronics) * self.total_day)
        else:
            chronic_day_dist = torch.distributions.categorical.Categorical(logits=torch.Tensor(self.chronic_day_priority))
            self.chronic_day_id = chronic_day_dist.sample().item()

        self.chronic_id = int(np.floor(self.chronic_day_id / self.total_day))
        self.day_id = self.chronic_day_id % self.total_day

        self.env.set_id(self.chronics[self.chronic_id])
        self.env.reset(**kwargs)
        self.env.fast_forward_chronics(self.day_id * 288)
        observation = self.env.get_obs()
        self.last_obs_is_hazard = get_is_hazard(observation.rho, self.env.get_thermal_limit(), self.hazard_ratio)
        return self.convert_observation(observation)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action) if action.size == 1 else action
        action = self.action_space_converter.convert_act(action)
        observation, reward, done, info = self.env.step(action)
        done = self.is_day_done() if done is False else done

        if not done and False in observation.line_status:
            change_line_status = self.env.action_space.get_change_line_status_vect()
            disconnect_lines = np.where(observation.line_status == False)[0]
            for line_id in disconnect_lines:
                if observation.time_next_maintenance[line_id] != 0:
                    if observation.time_before_cooldown_line[line_id] == 0:
                        change_line_status[line_id] = True
            if True in change_line_status:
                action_change_line_status = self.env.action_space({'change_line_status': change_line_status})
                observation, reward, done, info = self.env.step(action_change_line_status)
                done = self.is_day_done() if done is False else done

        if done and not self.use_test:
            self.update_priority()

        self.last_obs_is_hazard = get_is_hazard(observation.rho, self.env.get_thermal_limit(), self.hazard_ratio)
        info['last_obs_is_hazard'] = self.last_obs_is_hazard
        info['sum_load_p'] = self.env.backend.loads_info()[0].sum()
        info['sum_gen_p'] = self.env.backend.generators_info()[0].sum()

        return self.convert_observation(observation), reward, done, info


class Grid2OpMlpConfig(Grid2OpConfig):
    def __init__(self, env, hazard_ratio, use_test):
        super(Grid2OpMlpConfig, self).__init__(env, hazard_ratio, use_test)
        self.subset_attr_name = ['prod_p', 'prod_q', 'prod_v', 'load_p', 'load_q', 'load_v', 'topo_vect',
                                 'p_or', 'q_or', 'v_or', 'p_ex', 'q_ex', 'v_ex', 'rho', 'line_status',
                                 'timestep_overflow',
                                 'time_next_maintenance']
        self.observation_idx, self.observation_size = get_obs_subset(env.observation_space, self.subset_attr_name)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.observation_size,), dtype=np.float32)
        self.obs_dim = self.observation_size

    def convert_observation(self, observation):
        observation_vec = (observation.to_vect() - self.obs_mean) / self.obs_std

        return observation_vec[self.observation_idx]


class Grid2OpGnnConfig(Grid2OpConfig):
    def __init__(self, env, hazard_ratio, use_test):
        super(Grid2OpGnnConfig, self).__init__(env, hazard_ratio, use_test)
        self.observation_space = Graph(node_num=self.env.dim_topo, feature_dim=10)

    def convert_observation(self, observation):
        observation.from_vect((observation.to_vect() - self.obs_mean) / self.obs_std, check_legit=False)

        node_feature, adj_matrix = get_graph(observation)
        g_nx = nx.from_numpy_matrix(adj_matrix)
        g_dgl = dgl.add_self_loop(dgl.from_networkx(g_nx))
        g_dgl.ndata['x'] = torch.tensor(node_feature, dtype=torch.float32)
        return g_dgl


def grid2op_mlp_wrapper(env, hazard_ratio, use_test=False):
    env = Grid2OpMlpConfig(env, hazard_ratio, use_test)
    return env


def grid2op_gnn_wrapper(env, hazard_ratio, use_test=False):
    env = Grid2OpGnnConfig(env, hazard_ratio, use_test)
    return env


def wrap_exp(env_id, mode):
    if env_id[:5] == 'l2rpn':
        if mode == 'mlp':
            return grid2op_mlp_wrapper
        if mode == 'gnn':
            return grid2op_gnn_wrapper

    assert False, "wrapper not exist"
