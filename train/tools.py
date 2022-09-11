# ******************************************************************************
# * Copyright (C) Alibaba-inc - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# *****************************************************************************

import collections
from copy import deepcopy
import math
import os
from typing import Dict
import traceback
import numpy
import torch
import torch as ch
import torch.nn as nn
from train.config import PolicyParam
from train.np_image import NPImage
import numpy as np 
from geek.env.logger import Logger

logger = Logger.get_logger(__name__)
STD = 2 ** 0.5


def file_name(file_dir, file_type):
    L = []
    for dirpath, dirnames, filenames in os.walk(file_dir):
        for file in filenames:
            if os.path.splitext(file)[1] == file_type:
                L.append(os.path.join(dirpath, file))
    return L

class EnvPostProcsser:
    def __init__(self) -> None:
        self.args = PolicyParam

        self.target_speed = self.args.target_speed
        self.dt = self.args.dt
        self.history_length = self.args.history_length
        self.img_width = self.args.img_width
        self.img_length = self.args.img_length
        self.vec_length = self.args.ego_vec_length
        self.surr_vec_length = self.args.surr_vec_length
        self.surr_number = self.args.surr_agent_number
        self.obs_type = self.args.obs_type

        self.prev_distance = None
        # self.surr_cnn_normalize = Normalization(shape=(self.img_width, self.img_length, 3))
        # self.reward_scale = RewardScaling(shape=1, gamma=self.args.gamma) # not used
        # self.surr_img_deque = collections.deque(maxlen=self.history_length)
        # running mean std 
        self.surr_vec_deque = collections.deque(maxlen=self.history_length)
        self.vec_deque = collections.deque(maxlen=self.history_length)
        
        for i in range(self.history_length):
            # self.surr_img_deque.append(numpy.zeros((self.img_width, self.img_length, 3)))
            self.surr_vec_deque.append(numpy.zeros((1, self.surr_number * self.surr_vec_length)))
            self.vec_deque.append(numpy.zeros(self.vec_length))
        # self.tianchi_cnn = TianchiCNN()  # not used

    def process_surr_vec_obs(self, observation) -> np.array :

        curr_xy = (observation["player"]["status"][0],  # 车辆后轴中心位置 x
                   observation["player"]["status"][1])  # 车辆后轴中心位置 y 
        npc_info_dict = {}
        
        for npc_info in observation["npcs"]:
            if int(npc_info[0]) == 0:
                continue
            npc_info_dict[
                numpy.sqrt((npc_info[2] - curr_xy[0]) ** 2 + (npc_info[3] - curr_xy[1]) ** 2)
            ] = [
                npc_info[2] - curr_xy[0], # dx
                npc_info[3] - curr_xy[1], # dy
                npc_info[4], # 障碍物朝向
                numpy.sqrt(npc_info[5] ** 2 + npc_info[6] ** 2), # 障碍物速度大小（标量）
                numpy.sqrt(npc_info[7] ** 2 + npc_info[8] ** 2), # 障碍物加速度大小（标量）
                npc_info[9], # 障碍物宽度
                npc_info[10], # 障碍物长度
            ]
         # 按距离由近至远排序
        sorted_npc_info_dict = dict(sorted(npc_info_dict.items(), key=lambda x: x[0]))
        surr_obs_list = list(sorted_npc_info_dict.values())
         # 若数量不足surr_number则补齐
        for _ in range(self.surr_number - len(surr_obs_list)):
            surr_obs_list.append(list(numpy.zeros(self.surr_vec_length)))
        # 截断 
        curr_surr_obs = numpy.array(surr_obs_list).reshape(-1)[: self.surr_number * 7]
        curr_surr_obs = numpy.array(surr_obs_list)[: self.surr_number]

        return curr_surr_obs

    def assemble_surr_vec_obs(self, obs) -> torch.tensor:
        
        obs = deepcopy(obs)
        curr_surr_obs = self.process_surr_vec_obs(obs)
        # 加入到末尾帧
        self.surr_vec_deque.append(curr_surr_obs)
        surr_vec_obs = np.array(list(self.surr_vec_deque))
        sur_state = torch.Tensor(surr_vec_obs).float().unsqueeze(0)
        return sur_state

    def process_ego_vec_obs(self, observation) -> np.array:
        target_xy = (
            (observation["player"]["target"][0] + observation["player"]["target"][4]) / 2, # 目标区域中心位置x
            (observation["player"]["target"][1] + observation["player"]["target"][5]) / 2, # 目标区域中心位置y
        )
        curr_xy = (observation["player"]["status"][0], observation["player"]["status"][1]) # 当前车辆位置
        delta_xy = (target_xy[0] - curr_xy[0], target_xy[1] - curr_xy[1]) # 目标区域与当前位置的绝对偏差
        curr_yaw = observation["player"]["status"][2] # 当前朝向
        curr_velocity = observation["player"]["status"][3] # 当前车辆后轴中心纵向速度
        prev_steer = observation["player"]["status"][7] # 上一个前轮转角命令
        prev_acc = observation["player"]["status"][8] # 上一个加速度命令
        lane_list = []
                    
        if observation['map'] is not None:
            for lane_info in observation["map"].lanes:
                lane_list.append(lane_info.lane_id)
            current_lane_index = lane_list.index(observation["map"].lane_id)
            current_offset = observation["map"].lane_offset
            self.pre_lane_index = current_lane_index
            self.pre_offset = current_offset
        else:
            logger.info('map in obs is None!')
            current_lane_index = self.pre_lane_index
            current_offset = self.pre_offset

        vec_obs = numpy.array(
            [
                delta_xy[0],  # 目标区域与当前位置的偏差x
                delta_xy[1],  # 目标区域与当前位置的偏差y
                curr_yaw,  # 当前车辆的朝向角
                curr_velocity,  # 车辆后轴当前纵向速度
                prev_steer,  # 上一个前轮转角命令
                prev_acc,  # 上一个加速度命令
                current_lane_index,  # 当前所处车道的id
                current_offset,  # 车道的偏移量
            ]
        )
        self.pre_vec_obs = vec_obs
        return vec_obs

    def assemble_ego_vec_obs(self, obs) -> torch.tensor:

        observation = deepcopy(obs)
        vec_obs = self.process_ego_vec_obs(observation)
        # 添加到末尾帧
        self.vec_deque.append(vec_obs)
        mlp_obs = numpy.array(list(self.vec_deque))
        ego_state = torch.Tensor(mlp_obs).float().unsqueeze(0)
        
        return ego_state

    def assemble_reward(self, observation: Dict, info: Dict) -> float:
        try:
            target_xy = (
                (observation["player"]["target"][0] + observation["player"]["target"][4]) / 2,
                (observation["player"]["target"][1] + observation["player"]["target"][5]) / 2,
            )
            curr_xy = (observation["player"]["status"][0], observation["player"]["status"][1])
            distance_with_target = numpy.sqrt(
                (target_xy[0] - curr_xy[0]) ** 2 + (target_xy[1] - curr_xy[1]) ** 2
            )
        except KeyError:
            print(info)
            logger.error(f"exception: {traceback.print_exc()}")
            distance_with_target = self.prev_distance

        if self.prev_distance is None:
            self.prev_distance = distance_with_target
        
        distance_reward = (self.prev_distance - distance_with_target) * 0.5
        self.prev_distance = distance_with_target
        step_reward = -0.1

        # if info["collided"]:
        #     end_reward = -200
        if info["reached_stoparea"]:
            end_reward = 200
        elif info["timeout"]:
            end_reward = -200
        else:
            end_reward = 0.0
        
        if observation['map'] is not None:
            current_lane_index = observation["map"].lane_id
            current_offset = observation["map"].lane_offset
        else:
            logger.info('map in obs is None!')
            current_lane_index = -1
            current_offset = 0

        # add penalty when reaching close to other cars (same lane)
        length = observation["player"]['property'][1] # 车辆长度
        width = observation["player"]['property'][0] # 车辆宽度
        npc_info = observation['npcs'] 
        same_lane_npcs = [] # 同车道 npc
        for npc in npc_info:
            if npc[0] == 0:
                break
            dy = npc[3] - observation["player"]['status'][1]
            if np.abs(dy) < width:
                same_lane_npcs.append(npc)
        collide_reward = 0
        for npc in same_lane_npcs:
            safe_distance = (npc[-1] + length)/2 + length
            dx = npc[2] - observation["player"]['status'][0]
            if dx < 0:
                if np.abs(dx) < safe_distance:
                    penalty = -0.1-(safe_distance - np.abs(dx))*0.1
                    collide_reward = min(collide_reward, penalty)
        
        if info['collided']:
            collide_reward -= 10 

        return distance_reward + end_reward + step_reward + collide_reward

    def reset(self, initial_obs):
        self.prev_distance = None
        self.pre_vec_obs = None
        self.pre_lane_index = 1
        self.pre_offset = 0
        # self.reward_scale.reset()
        # self.img_deque = collections.deque(maxlen=5)
        # 填充初始帧
        ego_vec_state = self.process_ego_vec_obs(initial_obs)
        sur_vec_state = self.process_surr_vec_obs(initial_obs)
        self.vec_deque = collections.deque(maxlen=5)
        self.surr_vec_deque = collections.deque(maxlen=5)
        for i in range(self.history_length):
            self.vec_deque.append(ego_vec_state)
            self.surr_vec_deque.append(sur_vec_state)
        ego_vec_state = torch.from_numpy(numpy.array(list(self.vec_deque))).unsqueeze(0) # [1,5,8]
        sur_vec_state = torch.from_numpy(numpy.array(list(self.surr_vec_deque))).unsqueeze(0) # [1,5,8]
        return ego_vec_state, sur_vec_state
