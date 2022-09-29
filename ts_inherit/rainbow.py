import itertools
from math import pi
import numpy as np
from tianshou.policy import RainbowPolicy

from rainbow.config import cfg


class MyRainbow(RainbowPolicy):

    def add_iter(self):
        self._iter += 1

    def make_action_library(self, cfgs):
        steer_prime_choices = cfgs.steer_prime_choices
        acc_prime_choice = cfgs.acc_prime_choice
        self.action_library = list(itertools.product(steer_prime_choices, acc_prime_choice))

        # mesh = [np.linspace(lo, hi, a) for lo, hi, a in zip(cfgs.action_low, cfgs.action_high, cfgs.action_per_dim)]
        # self.action_library = list(itertools.product(*mesh))

    def map_action(self, data):
        """Map raw network output to action range in gym's env.action_space.

        This function is called in :meth:`~tianshou.data.Collector.collect` and only
        affects action sending to env. Remapped action will not be stored in buffer
        and thus can be viewed as a part of env (a black box action transformation).

        Action mapping includes 2 standard procedures: bounding and scaling. Bounding
        procedure expects original action range is (-inf, inf) and maps it to [-1, 1],
        while scaling procedure expects original action range is (-1, 1) and maps it
        to [action_space.low, action_space.high]. Bounding procedure is applied first.

        :param act: a data batch or numpy.ndarray which is the action taken by
            policy.forward.

        :return: action in the same form of input "act" but remap to the target action
            space.
        """
        assert hasattr(self, 'action_library')
        assert len(data.act.shape) <= 2, f"Unknown action format with shape {data.act.shape}."

        obs = data.obs['ego_obs']['data']
        act = data.act

        action = list()
        for _idx in range(len(act)):
            _act = act[_idx]
            _steer_prime, _acc_prime = self.action_library[_act]
            _steer = np.clip(_steer_prime, -pi/60.0, pi/60.0)
            _acc = np.clip(obs[_idx][-1][0][7] + _acc_prime, -2.0, 2.0)
            action.append([_steer, _acc])
        return np.array(action, dtype=np.float32)

    def sync_weight(self) -> None:
        """Synchronize the weight for the target network."""
        self.model_old.load_state_dict(self.model.state_dict())
        self.logger.info(f'Policy Parameters Updated!')

    def set_logger(self, logger):
        if not hasattr(self, 'logger'):
            setattr(self, 'logger', logger)


if __name__ == '__main__':
    a = np.linspace(-1, 1, 3)
    print(a)


