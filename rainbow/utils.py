import gym
import torch
from math import pi
import numpy as np
from tianshou.data import Batch
from rainbow.config import cfg
import itertools


def get_observation(observation):
    curr_xy = (observation["player"]["status"][0], observation["player"]["status"][1])
    npc_info_dict = {}

    for npc_info in observation["npcs"]:
        if int(npc_info[0]) == 0:
            continue
        npc_info_dict[np.sqrt((npc_info[2] - curr_xy[0]) ** 2 + (npc_info[3] - curr_xy[1]) ** 2)] = [
            npc_info[2] - curr_xy[0],
            npc_info[3] - curr_xy[1],
            npc_info[4],
            np.sqrt(npc_info[5] ** 2 + npc_info[6] ** 2),
            np.sqrt(npc_info[7] ** 2 + npc_info[8] ** 2),
            npc_info[9],
            npc_info[10],
        ]
    sorted_npc_info_dict = dict(sorted(npc_info_dict.items(), key=lambda x: x[0]))
    surr_obs_list = list(sorted_npc_info_dict.values())
    for _ in range(cfg.surr_number - len(surr_obs_list)):
        surr_obs_list.append(list(np.zeros(cfg.surr_vec_length)))

    curr_surr_obs = np.array(surr_obs_list).reshape(-1)[: cfg.surr_number * 7]

    return curr_surr_obs


def compute_reward(obs):
    return 1


def preprocess_fn(**kwargs):
    if 'rew' in kwargs:
        obs_next = list()
        rew = list()
        for _id in kwargs['env_id']:
            obs_next.append(get_observation(kwargs['obs_next'][_id]))
            rew.append(compute_reward(kwargs['obs_next'][_id]))
            # kwargs['obs_next'][_id] = get_observation(kwargs['obs_next'][_id])
            # kwargs['rew'][_id] = compute_reward(kwargs['obs'][_id], kwargs['obs_next'][_id])
        obs_next = np.array(obs_next)
        rew = np.array(rew)
        # kwargs['obs_next'] = np.array(kwargs['obs_next'])
        # kwargs['rew'] = np.array(kwargs['rew'])
        return Batch(obs_next=obs_next, rew=rew, done=kwargs['done'], info=kwargs['info'], policy=kwargs['policy'], env_id=kwargs['env_id'])
    else:
        obs = list()
        for _id in kwargs['env_id']:
            obs.append(get_observation(kwargs['obs'][_id]))
            # kwargs['obs'][_id] = get_observation(kwargs['obs'][_id])
        obs = np.array(obs)
        # kwargs['obs'] = np.array(kwargs['obs'])
        return Batch(obs=obs, env_id=kwargs['env_id'])


def set_seed(seed=1):
    # seed
    np.random.seed(seed)
    torch.manual_seed(seed)


class ContinuousToDiscrete:
    def __init__(self, action_space, action_per_dim):
        assert isinstance(action_space, gym.spaces.Box)

        low, high = action_space.low, action_space.high

        if isinstance(action_per_dim, int):
            action_per_dim = [action_per_dim] * action_space.shape[0]
        assert len(action_per_dim) == action_space.shape[0]
        self.action_space = gym.spaces.MultiDiscrete(action_per_dim)
        self.mesh = np.array([np.linspace(lo, hi, a) for lo, hi, a in zip(low, high, action_per_dim)], dtype=object)

    def action(self, act) -> np.ndarray:
        assert len(act.shape) <= 2, f"Unknown action format with shape {act.shape}."
        if len(act.shape) == 1:
            return np.array([self.mesh[i][a] for i, a in enumerate(act)])
        return np.array([[self.mesh[i][a] for i, a in enumerate(a_)] for a_ in act])


class ActionProjection:
    def __init__(self, low, high, action_per_dim):
        mesh = [np.linspace(lo, hi, a) for lo, hi, a in zip(low, high, action_per_dim)]
        self.action_library = list(itertools.product(*mesh))

    def get_action(self, act):
        assert len(act.shape) <= 2, f"Unknown action format with shape {act.shape}."
        if len(act.shape) == 1:
            return np.array([self.action_library[a] for a in act])
        return np.array([[self.action_library[a] for a in a_] for a_ in act])


if __name__ == '__main__':
    demo = ActionProjection(low=np.array([-pi/4.0, -6, -1]), high=np.array([pi/4.0, 2, 3]), action_per_dim=[5, 7, 3])
    action = np.array([0, 1])
    print(demo.get_action(action))
    action = np.array([[10], [3]])
    print(demo.get_action(action))
    pass
