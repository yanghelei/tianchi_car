import gym
import numpy as np
from tianshou.data import Batch
from tianshou.policy import RainbowPolicy
import itertools


class MyRainbow(RainbowPolicy):

    def make_action_library(self, cfgs):
        mesh = [np.linspace(lo, hi, a) for lo, hi, a in zip(cfgs.action_low, cfgs.action_high, cfgs.action_per_dim)]
        self.action_library = list(itertools.product(*mesh))

    def map_action(self, act):
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
        assert len(act.shape) <= 2, f"Unknown action format with shape {act.shape}."
        if len(act.shape) == 1:
            return np.array([self.action_library[a] for a in act])
        return np.array([[self.action_library[a] for a in a_] for a_ in act])

