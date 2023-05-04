import torch
from stable_baselines3.common.callbacks import BaseCallback


class Grid2OpTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(Grid2OpTensorboardCallback, self).__init__(verbose)

        self.action_distribution = []
        self.cnt_do_nothing = 0
        self.cnt_is_illegal = 0
        self.cnt_is_ambiguous = 0

    def _on_rollout_start(self) -> None:
        self.action_distribution = []
        self.cnt_do_nothing = 0
        self.cnt_is_illegal = 0
        self.cnt_is_ambiguous = 0

    def _on_rollout_end(self) -> None:
        self.logger.record('action/action_distribution', torch.as_tensor(self.action_distribution), exclude=("stdout", "log", "json", "csv"))
        self.logger.record_mean('action/cnt_do_nothing', self.cnt_do_nothing)
        self.logger.record_mean('action/is_illegal', self.cnt_is_illegal)
        self.logger.record_mean('action/is_ambiguous', self.cnt_is_ambiguous)

    def _on_step(self) -> bool:
        try:
            actions = self.locals['actions']
        except:
            actions = self.locals['action']

        for i in range(self.locals['env'].num_envs):
            self.action_distribution.append(actions[i])
            if actions[i] == 0:
                self.cnt_do_nothing += 1
            if self.locals['infos'][i]['is_illegal']:
                self.cnt_is_illegal += 1
            if self.locals['infos'][i]['is_ambiguous']:
                self.cnt_is_illegal += 1

        return True


if __name__ == '__main__':
    callback = Grid2OpTensorboardCallback()
