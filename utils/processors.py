import gym
import torch
import itertools
import numpy as np
import collections
from math import pi
from tianshou.data import Batch
from utils.math import compute_distance


class Processor:
    def __init__(self, cfgs, model, mode='train'):

        self.mode = mode

        self.n_env = cfgs.training_num if mode == 'train' else 15 - cfgs.training_num

        self.envs_id = [i for i in range(self.n_env)]
        self.env_last_obs = [None] * self.n_env
        self.env_last_distance = [None] * self.n_env
        self.model = model

        self.max_consider_nps = cfgs.max_consider_nps
        self.sur_dim = cfgs.network.sur_dim
        self.ego_dim = cfgs.network.ego_dim
        # self.surr_vec_deque = collections.deque(maxlen=cfgs.history_length)

    # def assemble_surr_vec_obs(self, obs, sur_norm_layer, ego_norm_layer):
    #
    #     curr_surr_obs = get_observation(obs)
    #     # 加入到末尾帧
    #     self.surr_vec_deque.append(curr_surr_obs)
    #     surr_vec_obs = np.array(list(self.surr_vec_deque))
    #     sur_norm_layer.update(surr_vec_obs[-1])
    #     sur_state = torch.Tensor(surr_vec_obs).float().unsqueeze(0)  # 这里还是未norm的raw_data

    # return sur_state

    def get_observation(self, observation):
        curr_xy = (observation["player"]["status"][0], observation["player"]["status"][1])  # 车辆后轴中心位置
        npc_info_dict = {}

        for npc_info in observation["npcs"]:
            if int(npc_info[0]) == 0:
                continue
            npc_info_dict[np.sqrt((npc_info[2] - curr_xy[0]) ** 2 + (npc_info[3] - curr_xy[1]) ** 2)] = [
                npc_info[2] - curr_xy[0],  # dx
                npc_info[3] - curr_xy[1],  # dy
                npc_info[4],  # 障碍物朝向
                np.sqrt(npc_info[5] ** 2 + npc_info[6] ** 2),  # 障碍物速度大小（标量）
                np.sqrt(npc_info[7] ** 2 + npc_info[8] ** 2),  # 障碍物加速度大小（标量）
                npc_info[9],  # 障碍物宽度
                npc_info[10],  # 障碍物长度
            ]
        if len(npc_info_dict) == 0:
            sur_obs_list = np.zeros((self.max_consider_nps, self.sur_dim))
            n_sur = 1
        else:
            # 按距离由近至远排列
            sorted_npc_info_dict = dict(sorted(npc_info_dict.items(), key=lambda x: x[0]))
            sur_obs_list = list(sorted_npc_info_dict.values())
            n_sur = len(sur_obs_list)
            for _ in range(self.max_consider_nps - n_sur):
                sur_obs_list.append(list(np.zeros(self.sur_dim)))
            sur_obs_list = np.array(sur_obs_list)[:self.max_consider_nps, :]

        # sur_obs_list = list(sorted_npc_info_dict.values())
        # # 若数量不足 surr_number 则补齐
        # for _ in range(cfg.surr_number - len(sur_obs_list)):
        #     sur_obs_list.append(list(np.zeros(cfg.surr_vec_length)))
        # # 截断
        # curr_surr_obs = np.array(sur_obs_list).reshape(-1)[: cfg.surr_number * 7]

        target_xy = (
            (observation["player"]["target"][0] + observation["player"]["target"][4]) / 2,  # 目标区域中心位置x
            (observation["player"]["target"][1] + observation["player"]["target"][5]) / 2,  # 目标区域中心位置y
        )
        curr_xy = (observation["player"]["status"][0], observation["player"]["status"][1])  # 当前车辆位置
        delta_xy = (target_xy[0] - curr_xy[0], target_xy[1] - curr_xy[1])  # 目标区域与当前位置的绝对偏差
        curr_yaw = observation["player"]["status"][2]  # 当前朝向
        curr_velocity = observation["player"]["status"][3]  # 当前车辆后轴中心纵向速度
        prev_steer = observation["player"]["status"][7]  # 上一个前轮转角命令
        prev_acc = observation["player"]["status"][8]  # 上一个加速度命令
        lane_list = []

        for lane_info in observation["map"].lanes:
            lane_list.append(lane_info.lane_id)

        current_lane_index = lane_list.index(observation["map"].lane_id)

        current_offset = observation["map"].lane_offset

        ego_obs = np.array(
            [[
                delta_xy[0],  # 目标区域与当前位置的偏差x
                delta_xy[1],  # 目标区域与当前位置的偏差y
                curr_yaw,  # 当前车辆的朝向角
                curr_velocity,  # 车辆后轴当前纵向速度
                prev_steer,  # 上一个前轮转角命令
                prev_acc,  # 上一个加速度命令
                current_lane_index,  # 当前所处车道的id
                current_offset,  # 车道的偏移量
            ]]
        )

        obs = dict(
            sur_obs=dict(
                n=n_sur,
                data=sur_obs_list
            ),
            ego_obs=dict(
                n=1,
                data=ego_obs
            )
        )
        self.model.sur_norm.update(sur_obs_list)
        self.model.ego_norm.update(ego_obs)
        return obs

    def compute_reward(self, env_id, next_obs, info):
        target_xy = (
            (next_obs["player"]["target"][0] + next_obs["player"]["target"][4]) / 2,
            (next_obs["player"]["target"][1] + next_obs["player"]["target"][5]) / 2,
        )
        curr_xy = (next_obs["player"]["status"][0], next_obs["player"]["status"][1])
        distance_with_target = compute_distance(target_xy, curr_xy)

        assert self.env_last_distance[env_id] is not None

        # distance_reward = (self.env_last_distance[env_id] - distance_with_target) / (self.target_speed * self.dt)
        distance_reward = (self.env_last_distance[env_id] - distance_with_target) / (1 * 1)

        self.env_last_distance[env_id] = distance_with_target

        step_reward = -1

        if info["collided"]:
            end_reward = -2000
        elif info["reached_stoparea"]:
            end_reward = 2000
        elif info["timeout"]:
            end_reward = -2000
        else:
            end_reward = 0.0
        return distance_reward + end_reward + step_reward

    def update_distance_to_target(self, env_id):
        obs = self.env_last_obs[env_id]
        assert obs is not None
        target_xy = (
            (obs["player"]["target"][0] + obs["player"]["target"][4]) / 2,
            (obs["player"]["target"][1] + obs["player"]["target"][5]) / 2,
        )
        curr_xy = (obs["player"]["status"][0], obs["player"]["status"][1])
        self.env_last_distance[env_id] = compute_distance(target_xy, curr_xy)

    def preprocess_fn(self, **kwargs):
        # assert len(kwargs['env_id']) == len(self.envs_id)
        if 'rew' in kwargs:
            obs_next = [None] * len(kwargs['env_id'])
            rew = [0] * len(kwargs['env_id'])
            for _idx, _id in enumerate(kwargs['env_id']):
                obs_next[_idx] = self.get_observation(kwargs['obs_next'][_idx])
                rew[_idx] = self.compute_reward(_id, kwargs['obs_next'][_idx], kwargs['info'][_idx])
                self.env_last_obs[_id] = kwargs['obs_next'][_idx]
            obs_next = np.array(obs_next)
            rew = np.array(rew)
            return Batch(obs_next=obs_next, rew=rew, done=kwargs['done'], info=kwargs['info'], policy=kwargs['policy'], env_id=kwargs['env_id'])
        else:
            obs = [None] * len(kwargs['env_id'])
            for _idx, _id in enumerate(kwargs['env_id']):
                obs[_idx] = self.get_observation(kwargs['obs'][_idx])
                self.env_last_obs[_id] = kwargs['obs'][_idx]
                self.update_distance_to_target(_id)
            obs = np.array(obs)
            return Batch(obs=obs, env_id=kwargs['env_id'])


def set_seed(seed=1):
    # seed
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    demo = ActionProjection(low=np.array([-pi / 4.0, -6, -1]), high=np.array([pi / 4.0, 2, 3]), action_per_dim=[5, 7, 3])
    action = np.array([0, 1])
    print(demo.get_action(action))
    action = np.array([[10], [3]])
    print(demo.get_action(action))
    pass
