# ******************************************************************************
# * Copyright (C) Alibaba-inc - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# *****************************************************************************

import collections
from copy import deepcopy
import math
import os
from typing import Dict

import numpy
import torch
import torch as ch
import torch.nn as nn
from train.config import PolicyParam
from train.np_image import NPImage
import numpy as np 

STD = 2 ** 0.5


def file_name(file_dir, file_type):
    L = []
    for dirpath, dirnames, filenames in os.walk(file_dir):
        for file in filenames:
            if os.path.splitext(file)[1] == file_type:
                L.append(os.path.join(dirpath, file))
    return L

def initialize_weights(mod, initialization_type, scale=STD):
    """
    Weight initializer for the models.
    Inputs: A model, Returns: none, initializes the parameters
    """
    for p in mod.parameters():
        if initialization_type == "normal":
            p.data.normal_(0.01)
        elif initialization_type == "xavier":
            if len(p.data.shape) >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                p.data.zero_()
        elif initialization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
            else:
                p.data.zero_()
        else:
            raise ValueError("Need a valid initialization key")


def orthogonal_init(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    u, s, v = ch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with ch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


class TianchiCNN(object):
    def __init__(
        self,
    ):
        self.base_image_size = 500
        self.crop_size = 224
        self.resize_size = 224
        self.base_image_reso = 1.0
        self.draw_count = 0

    def draw_from_obs(self, obs, center_dict=None):

        undrivable_img = NPImage(
            meter_per_pixel=self.base_image_reso,
            width=self.base_image_size,
            height=self.base_image_size,
        )

        drivable_img = NPImage(
            meter_per_pixel=self.base_image_reso,
            width=self.base_image_size,
            height=self.base_image_size,
        )

        obs_img = NPImage(
            meter_per_pixel=self.base_image_reso,
            width=self.base_image_size,
            height=self.base_image_size,
        )

        center_pose_x = obs["player"]["status"][0]
        center_pose_y = obs["player"]["status"][1]
        center_pose_h = obs["player"]["status"][2]

        ego_width = obs["player"]["property"][0]
        ego_length = obs["player"]["property"][1]

        undrivable_img.set_center_pose((center_pose_x, center_pose_y))
        drivable_img.set_center_pose((center_pose_x, center_pose_y))
        obs_img.set_center_pose((center_pose_x, center_pose_y))

        obs_img.draw_rect((center_pose_x, center_pose_y, center_pose_h), ego_length, ego_width, 255)

        for npc in obs["npcs"]:
            obs_img.draw_rect((npc[2], npc[3], npc[4]), npc[10], npc[9], 128)

        if center_dict:
            for centers in center_dict.values():
                drivable_img.draw_polyline(centers, 128)

        merge_img = undrivable_img.merge([drivable_img, obs_img])

        heading_angle = 90 - center_pose_h * 180.0 / math.pi

        merge_img.resize(self.resize_size).rotate(
            (self.resize_size / 2, self.resize_size / 2), heading_angle, self.resize_size
        ).flip()

        return merge_img.img_data


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
        else:
            current_lane_index = -1
            current_offset = 0
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
        if observation is None:
            observation = self.pre_vec_obs
            print(info) 
        target_xy = (
            (observation["player"]["target"][0] + observation["player"]["target"][4]) / 2,
            (observation["player"]["target"][1] + observation["player"]["target"][5]) / 2,
        )
        curr_xy = (observation["player"]["status"][0], observation["player"]["status"][1])
        distance_with_target = numpy.sqrt(
            (target_xy[0] - curr_xy[0]) ** 2 + (target_xy[1] - curr_xy[1]) ** 2
        )
        if self.prev_distance is None:
            self.prev_distance = distance_with_target
        distance_reward = (self.prev_distance - distance_with_target) / (
            self.target_speed * self.dt
        )
        self.prev_distance = distance_with_target
        step_reward = -0.5

        if info["collided"]:
            end_reward = -200
        elif info["reached_stoparea"]:
            end_reward = 200
        elif info["timeout"]:
            end_reward = -200
        else:
            end_reward = 0.0
        
        # add penalty when reaching close to other cars
            
        return distance_reward + end_reward + step_reward

    def reset(self, initial_obs):
        self.prev_distance = None
        self.pre_vec_obs = None
        # self.reward_scale.reset()
        # self.img_deque = collections.deque(maxlen=5)
        # 填充初始帧
        ego_vec_state = self.process_ego_vec_obs(initial_obs)
        sur_vec_state = self.process_surr_vec_obs(initial_obs)
        ego_vec_state = np.clip(ego_vec_state, -5, 5)
        sur_vec_state = np.clip(sur_vec_state, -5, 5)
        self.vec_deque = collections.deque(maxlen=5)
        self.surr_vec_deque = collections.deque(maxlen=5)
        for i in range(self.history_length):
            self.vec_deque.append(ego_vec_state.flatten())
            self.surr_vec_deque.append(sur_vec_state.flatten())
        ego_vec_state = torch.from_numpy(numpy.array(list(self.vec_deque))).unsqueeze(0) # [1,5,8]
        sur_vec_state = torch.from_numpy(numpy.array(list(self.surr_vec_deque))).unsqueeze(0) # [1,5,8]
        return ego_vec_state, sur_vec_state
