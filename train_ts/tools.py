# ******************************************************************************
# * Copyright (C) Alibaba-inc - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# *****************************************************************************

import collections
import math
import os
from typing import Dict

import numpy
import numpy as np
import torch
import torch as ch
import torch.nn as nn

from train_ts.utils import cal_relative_rotation_angle
from train_ts.np_image import NPImage

STD = 2 ** 0.5


def file_name(file_dir, file_type):
    L = []
    for dirpath, dirnames, filenames in os.walk(file_dir):
        for file in filenames:
            if os.path.splitext(file)[1] == file_type:
                L.append(os.path.join(dirpath, file))
    return L


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = numpy.zeros(shape)
        self.S = numpy.zeros(shape)
        self.std = numpy.sqrt(self.S)

    def update(self, x):
        x = numpy.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = numpy.sqrt(self.S / self.n)


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = numpy.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = numpy.zeros(self.shape)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


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


def orthogonal_init(tensor, gain=1.0):
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
    def __init__(self, args) -> None:
        self.args = args

        self.target_speed = args.target_speed
        self.dt = args.dt
        self.history_length = args.history_length
        self.img_width = args.img_width
        self.img_length = args.img_length
        self.vec_length = args.ego_vec_length
        self.surr_vec_length = args.surr_vec_length
        self.surr_number = args.surr_agent_number
        self.obs_type = args.obs_type

        self.prev_distance = [None for _ in range(args.training_num)]
        self.surr_cnn_normalize = [Normalization(shape=(self.img_width, self.img_length, 3)) for _ in range(args.training_num)]
        self.vec_normalize = [Normalization(shape=self.vec_length) for _ in range(args.training_num)]
        self.surr_vec_normalize = [Normalization(shape=(1, self.surr_number * 7)) for _ in range(args.training_num)]
        self.reward_scale = [RewardScaling(shape=1, gamma=args.gamma) for _ in range(args.training_num)]
        self.surr_img_deque = [collections.deque(maxlen=5) for _ in range(args.training_num)]
        self.surr_vec_deque = [collections.deque(maxlen=5) for _ in range(args.training_num)]
        self.vec_deque = [collections.deque(maxlen=5) for _ in range(args.training_num)]
        
        for i in range(args.training_num):
            for j in range(self.history_length):
                self.surr_img_deque[i].append(numpy.zeros((self.img_width, self.img_length, 3)))
                self.surr_vec_deque[i].append(numpy.zeros((1, self.surr_number * 7)))
                self.vec_deque[i].append(numpy.zeros(self.vec_length))
            self.tianchi_cnn = TianchiCNN()  # not used

    def assemble_surr_cnn_obs(self, observations, env):
        """
        not used
        """
        env_state_list = []
        for env_id, observation in enumerate(observations):
            centers = {}
            for lane in observation["map"].lanes:
                lane_id = lane.lane_id
                centers[lane_id] = env[env_id].centers_by_lane_id(lane_id)

            img_obs = self.tianchi_cnn.draw_from_obs(observation, centers).astype(numpy.float32)
            img_obs = self.surr_cnn_normalize[env_id](img_obs)
            self.surr_img_deque[env_id].append(img_obs)
            cnn_obs = numpy.concatenate(list(self.img_deque[env_id]), axis=2)
            env_state = torch.Tensor(cnn_obs).float().unsqueeze(0).permute(0, 3, 2, 1)
            env_state_list.append(env_state)
        env_states = torch.cat(env_state_list, dim=0)
        return env_states

    def assemble_surr_vec_obs(self, observations):
        env_state_list = []
        multi_npc_info_dict = {}
        for env_id, observation in enumerate(observations):
            curr_xy = (observation["player"]["status"][0], observation["player"]["status"][1])
            curr_yaw = observation['player']['status'][2]
            npc_info_dict = {}

            for npc_info in observation["npcs"]:
                if int(npc_info[0]) == 0:
                    continue
                distance_to_npc = numpy.sqrt((npc_info[2] - curr_xy[0]) ** 2 + (npc_info[3] - curr_xy[1]) ** 2)
                rotation_angle = cal_relative_rotation_angle(np.array(list(curr_xy)),
                                                             np.array([npc_info[2], npc_info[3]]), curr_yaw)
                npc_info_dict[
                    distance_to_npc
                ] = [
                    distance_to_npc,
                    rotation_angle,
                    npc_info[4],
                    numpy.sqrt(npc_info[5] ** 2 + npc_info[6] ** 2),
                    numpy.sqrt(npc_info[7] ** 2 + npc_info[8] ** 2),
                    npc_info[9],
                    npc_info[10],
                ]
            sorted_npc_info_dict = dict(sorted(npc_info_dict.items(), key=lambda x: x[0]))
            surr_obs_list = list(sorted_npc_info_dict.values())
            for _ in range(self.surr_number - len(surr_obs_list)):
                surr_obs_list.append(list(numpy.zeros(self.surr_vec_length)))

            curr_surr_obs = numpy.array(surr_obs_list).reshape(-1)[: self.surr_number * 7]
            curr_surr_obs = self.surr_vec_normalize[env_id](curr_surr_obs)
            self.surr_vec_deque[env_id].append(curr_surr_obs.reshape(1, -1))
            surr_vec_obs = numpy.concatenate(list(self.surr_vec_deque[env_id]), axis=0)[numpy.newaxis, :, :]
            env_state = torch.Tensor(surr_vec_obs).float().unsqueeze(0)
            env_state_list.append(env_state)
            multi_npc_info_dict[env_id] = sorted_npc_info_dict
        env_states = torch.cat(env_state_list, dim=0)
        return env_states, multi_npc_info_dict

    def assemble_ego_vec_obs(self, observations):
        ego_vec_obs_list = []
        for env_id, observation in enumerate(observations):
            target_xy = (
                (observation["player"]["target"][0] + observation["player"]["target"][4]) / 2,
                (observation["player"]["target"][1] + observation["player"]["target"][5]) / 2,
            )
            curr_xy = (observation["player"]["status"][0], observation["player"]["status"][1])
            curr_yaw = observation["player"]["status"][2]
            distance_to_target = numpy.sqrt((target_xy[0] - curr_xy[0]) ** 2 + (target_xy[1] - curr_xy[1]) ** 2)
            rotation_angle = cal_relative_rotation_angle(np.array(list(curr_xy)),
                                                         np.array(list(target_xy)), curr_yaw)
            curr_velocity = observation["player"]["status"][3]
            prev_steer = observation["player"]["status"][7]
            prev_acc = observation["player"]["status"][8]
            lane_list = []

            for lane_info in observation["map"].lanes:
                lane_list.append(lane_info.lane_id)
            current_lane_index = lane_list.index(observation["map"].lane_id)
            current_offset = observation["map"].lane_offset
            vec_obs = numpy.array(
                [
                    distance_to_target,
                    rotation_angle,
                    curr_yaw,
                    curr_velocity,
                    prev_steer,
                    prev_acc,
                    current_lane_index,
                    current_offset,
                ]
            )
            vec_obs = self.vec_normalize[env_id](vec_obs)
            self.vec_deque[env_id].append(vec_obs)
            mlp_obs = numpy.concatenate(list(self.vec_deque[env_id]), axis=0)
            vec_state = torch.Tensor(mlp_obs).float().unsqueeze(0)
            ego_vec_obs_list.append(vec_state)
        vec_states = torch.stack(ego_vec_obs_list, dim=0)
        return vec_states

    def assemble_reward(self, observations: Dict, all_info: Dict, multi_npc_info_dict: Dict) -> np.array:
        reward_list = []
        mu = 0.05
        alpha = 0.05
        sigma = 0.01
        for env_id, observation in enumerate(observations):
            target_xy = (
                (observation["player"]["target"][0] + observation["player"]["target"][4]) / 2,
                (observation["player"]["target"][1] + observation["player"]["target"][5]) / 2,
            )
            sorted_npc_info_dict = multi_npc_info_dict[env_id]
            curr_xy = (observation["player"]["status"][0], observation["player"]["status"][1])
            distance_with_target = numpy.sqrt(
                (target_xy[0] - curr_xy[0]) ** 2 + (target_xy[1] - curr_xy[1]) ** 2
            )
            if self.prev_distance[env_id] is None:
                self.prev_distance[env_id] = distance_with_target

            # 基于势能的奖励设计
            distance_reward = mu * (self.prev_distance[env_id] - 0.999 * distance_with_target)

            self.prev_distance[env_id] = distance_with_target

            # 躲避障碍
            min_distance = min(sorted_npc_info_dict.keys())

            escape_collision_reward = -alpha * (min_distance < 2.0)

            step_reward = -sigma

            if all_info[env_id]["collided"]:
                end_reward = -10
            elif all_info[env_id]["reached_stoparea"]:
                end_reward = 20
            elif all_info[env_id]["timeout"]:
                end_reward = -10
            else:
                end_reward = 0.0
            reward = distance_reward + end_reward + step_reward + escape_collision_reward
            reward_list.append(reward)
        reward = np.array(reward_list, dtype=np.float64)
        return reward

    def assemble_surr_obs(self, observations, env=None):
        if self.obs_type == "cnn":
            return self.assemble_surr_cnn_obs(observations=observations, env=env)
        elif self.obs_type == "vec":
            return self.assemble_surr_vec_obs(observations=observations)
        else:
            raise Exception("error observation type")

    def reset(self):
        self.prev_distance = [None for _ in range(self.args.training_num)]
        self.img_deque = [collections.deque(maxlen=5) for _ in range(self.args['training_num'])]
        self.vec_deque = [collections.deque(maxlen=5) for _ in range(self.args['training_num'])]

        for env_id in range(self.args.training_num):
            self.reward_scale[env_id].reset()
            for _ in range(self.history_length):
                self.img_deque[env_id].append(numpy.zeros((self.img_width, self.img_length, 3)))
                self.vec_deque[env_id].append(numpy.zeros(self.vec_length))
