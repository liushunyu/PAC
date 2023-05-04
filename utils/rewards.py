from grid2op.Reward import BaseReward, RedispReward


class NormalizationRedispReward(RedispReward):
    def __init__(self):
        super(NormalizationRedispReward, self).__init__()

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        res = super().__call__(action, env, has_error, is_done, is_illegal, is_ambiguous)
        if has_error or is_illegal or is_ambiguous:
            res = res / self.reward_min
        else:
            res = res / self.reward_max
        return res


class LoadGenRatioReward(BaseReward):
    def __init__(self):
        BaseReward.__init__(self)

    def initialize(self, env):
        self.reward_min = -1.0
        self.reward_max = 1.0

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            # previous action was bad
            reward = self.reward_min
        elif is_done:
            # really strong reward if an episode is over without game over
            reward = self.reward_max
        else:
            gen_p, *_ = env.backend.generators_info()
            load_p, *_ = env.backend.loads_info()
            reward = load_p.sum() / gen_p.sum()
        return reward
