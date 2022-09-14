# ******************************************************************************
# * Copyright (C) Alibaba-inc - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# *****************************************************************************

import collections
from algo_ts.data.batch import Batch
import numpy as np
from train_ts.utils import cal_distance, cal_relative_rotation_angle, ActionProjection


class Processor:
    def __init__(self, cfgs, n_env, logger=None, norms=None, update_norm=True):
        self.cfgs = cfgs

        self.n_env = n_env

        self.envs_id = [i for i in range(self.n_env)]
        self.env_last_distance = [None] * self.n_env

        self.sur_norm = norms[0]
        self.ego_norm = norms[1]

        self.max_consider_nps = cfgs.max_consider_nps
        self.sur_dim = cfgs.sur_in
        self.ego_dim = cfgs.ego_in

        self.logger = logger

        self.update_norm = update_norm

        self.action_library = ActionProjection(cfgs.low, cfgs.high, cfgs.action_per_dim)

        self.n_ego_vec_deque = [collections.deque(maxlen=self.cfgs.history_length)] * self.n_env
        self.n_sur_vec_deque = [collections.deque(maxlen=self.cfgs.history_length)] * self.n_env

        self.reset_deque(self.envs_id)

    def reset_deque(self, env_ids):
        for env_id in env_ids:
            for _idx in range(self.cfgs.history_length):
                self.n_sur_vec_deque[env_id].append(np.zeros(self.max_consider_nps * self.sur_dim))
                self.n_ego_vec_deque[env_id].append(np.zeros(1 * self.ego_dim))

    def get_observation(self, observation, env_id=None):
        curr_xy = (observation["player"]["status"][0], observation["player"]["status"][1])  # 当前车辆位置

        target_xy = (
            (observation["player"]["target"][0] + observation["player"]["target"][4]) / 2,  # 目标区域中心位置x
            (observation["player"]["target"][1] + observation["player"]["target"][5]) / 2,  # 目标区域中心位置y
        )

        curr_yaw = observation["player"]["status"][2]  # 当前朝向

        distance_to_target = np.sqrt((target_xy[0] - curr_xy[0]) ** 2 + (target_xy[1] - curr_xy[1]) ** 2)
        rotation_angle = cal_relative_rotation_angle(np.array(list(curr_xy)),
                                                     np.array(list(target_xy)), curr_yaw)

        curr_velocity = observation["player"]["status"][3]  # 当前车辆后轴中心纵向速度
        curr_acc = observation["player"]["status"][4]  # 当前车辆后轴中心纵向加速度
        curr_lateral_acc = observation["player"]["status"][5]  # 当前车辆后轴中心横向加速度
        curr_steer = observation["player"]["status"][6]  # 当前前轮转角
        prev_steer = observation["player"]["status"][7]  # 上一个前轮转角命令
        prev_acc = observation["player"]["status"][8]  # 上一个加速度命令

        if prev_acc != curr_acc:
            self.logger.info(f'Now forward acc is {curr_acc}, last action acc is {prev_acc}!')

        lane_list = []

        if observation["map"] is not None:
            speed_limit = 0.0
            for lane_info in observation["map"].lanes:
                lane_list.append(lane_info.lane_id)
                if observation["map"].lane_id == lane_info.lane_id:
                    speed_limit = lane_info.speed_limit
            current_lane_index = lane_list.index(observation["map"].lane_id)
            current_offset = observation["map"].lane_offset
        else:  # 按照主办方的说法，车开到道路外有可能出现 none 的情况
            current_lane_index = -1.0
            current_offset = 0.0
            speed_limit = 0.0

        ego_obs = np.array(
            [
                distance_to_target,  # 目标区域与当前位置的偏差x
                rotation_angle,  # 目标区域与当前位置的偏差y
                curr_yaw,  # 当前车辆的朝向角
                curr_velocity,  # 车辆后轴当前纵向速度
                curr_lateral_acc,  # 车辆当前后轴横向加速度
                curr_steer,  # 车辆当前前轮转角
                prev_steer,  # 上一个前轮转角命令
                prev_acc,  # 上一个加速度命令(车辆当前后轴纵向加速度)
                current_lane_index,  # 当前所处车道的id
                speed_limit - curr_velocity,  # 当前车道速度上限与当前车速的差值
                current_offset,  # 车道的偏移量
            ]
        )

        npc_info_dict = {}

        for npc_info in observation["npcs"]:
            if int(npc_info[0]) == 0:
                continue

            distance_to_npc = np.sqrt((npc_info[2] - curr_xy[0]) ** 2 + (npc_info[3] - curr_xy[1]) ** 2)
            rotation_angle = cal_relative_rotation_angle(np.array(list(curr_xy)),
                                                         np.array([npc_info[2], npc_info[3]]), curr_yaw)

            npc_info_dict[distance_to_npc] = [
                distance_to_target,  # dx
                rotation_angle,  # dy
                npc_info[4],  # 障碍物朝向
                npc_info[5],  # vx
                npc_info[6],  # vy
                npc_info[7],  # ax
                npc_info[8],  # ay
                npc_info[9],  # 障碍物宽度
                npc_info[10],  # 障碍物长度
            ]

        # 按距离由近至远排列
        sorted_npc_info_dict = dict(sorted(npc_info_dict.items(), key=lambda x: x[0]))
        sur_obs_list = list(sorted_npc_info_dict.values())

        for _ in range(self.max_consider_nps - len(sur_obs_list)):
            sur_obs_list.append(list(np.zeros(self.sur_dim)))

        sur_obs_list = np.array(sur_obs_list)[:self.max_consider_nps, :].reshape(-1)  # 6 * 9 = 54

        if self.update_norm:
            self.sur_norm.update(sur_obs_list)
            self.ego_norm.update(ego_obs)

        self.n_sur_vec_deque[env_id].append(sur_obs_list)
        self.n_ego_vec_deque[env_id].append(ego_obs)

        sur_obs_list = np.array(list(self.n_sur_vec_deque[env_id]))  # 3 * 54 = 162
        ego_obs = np.array(list(self.n_ego_vec_deque[env_id]))  # 3 * 11 = 33

        obs = dict(
            sur_obs=sur_obs_list,
            ego_obs=ego_obs
        )

        return obs

    def compute_reward(self, env_id, next_obs, info):
        param1 = -0.1
        param2 = -0.05
        param3 = -0.02

        curr_xy = (next_obs["player"]["status"][0], next_obs["player"]["status"][1])
        target_xy = (
            (next_obs["player"]["target"][0] + next_obs["player"]["target"][4]) / 2,
            (next_obs["player"]["target"][1] + next_obs["player"]["target"][5]) / 2,
        )

        distance_with_target = cal_distance(curr_xy, target_xy)

        # 基于势能的奖励设计
        if self.env_last_distance[env_id] is None:
            self.env_last_distance[env_id] = distance_with_target

        distance_reward = param1 * (self.env_last_distance[env_id] - distance_with_target)

        # 更新智能体距离终点的距离
        self.env_last_distance[env_id] = distance_with_target

        # 期望智能体避免过于靠近其他npc
        npc_distance_dict = {}
        for npc_info in next_obs["npcs"]:
            if int(npc_info[0]) == 0:
                continue

            distance_to_npc = np.sqrt((npc_info[2] - curr_xy[0]) ** 2 + (npc_info[3] - curr_xy[1]) ** 2)

            # dict(distance: npc_id)
            npc_distance_dict[distance_to_npc] = npc_info[0]

        min_npc_distance = min(npc_distance_dict.keys())

        escape_collision_reward = -param2 * (min_npc_distance <= 3.0)

        # 期望智能体尽快到达目标区域
        step_reward = -param3

        # 一次性奖励惩罚
        if info["collided"]:
            end_reward = -10
        elif info["reached_stoparea"]:
            end_reward = 20
        elif info["timeout"]:
            end_reward = -10
        else:
            end_reward = 0.0
        reward = distance_reward + end_reward + step_reward + escape_collision_reward

        return reward

    def preprocess_fn(self, **kwargs):
        if 'rew' in kwargs:
            obs_next = [None] * len(kwargs['env_id'])
            rew = [0] * len(kwargs['env_id'])
            for _idx, _id in enumerate(kwargs['env_id']):
                obs_next[_idx] = self.get_observation(kwargs['obs_next'][_idx], env_id=_id)
                rew[_idx] = self.compute_reward(_id, kwargs['obs_next'][_idx], kwargs['info'][_idx])
            obs_next = np.array(obs_next)
            rew = np.array(rew)
            return Batch(obs_next=obs_next, rew=rew, done=kwargs['done'], policy=kwargs['policy'])
        else:
            self.reset_deque(kwargs['env_id'])
            obs = [None] * len(kwargs['env_id'])
            for _idx, _id in enumerate(kwargs['env_id']):
                obs[_idx] = self.get_observation(kwargs['obs'][_idx], env_id=_id)
            obs = np.array(obs)
            return Batch(obs=obs)
