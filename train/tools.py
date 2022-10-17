# ******************************************************************************
# * Copyright (C) Alibaba-inc - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# *****************************************************************************

import collections
from copy import deepcopy
import os
from typing import Dict
import traceback
import numpy
import torch
from train.config import PolicyParam
import numpy as np
from geek.env.logger import Logger
from utils.geometry import get_polygon
from pid import PIDController

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
    def __init__(self, stage, actions_map, pid_params=None) -> None:
        self.args = PolicyParam

        self.actions_map = actions_map

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
        self.stage = stage
        for i in range(self.history_length):
            # self.surr_img_deque.append(numpy.zeros((self.img_width, self.img_length, 3)))
            self.surr_vec_deque.append(numpy.zeros((1, self.surr_number * self.surr_vec_length)))
            self.vec_deque.append(numpy.zeros(self.vec_length))
        # self.tianchi_cnn = TianchiCNN()  # not used
        if pid_params is not None :
            self.pid_controller = PIDController(pid_params)
        else: 
            self.pid_controller = None

    def process_surr_vec_obs(self, observation) -> np.array:

        curr_xy = (observation["player"]["status"][0],  # 车辆后轴中心位置 x
                   observation["player"]["status"][1])  # 车辆后轴中心位置 y 

        npc_info_dict = {}

        for npc_info in observation["npcs"]:
            if int(npc_info[0]) == 0:
                continue

            npc_info_dict[
                numpy.sqrt((npc_info[2] - curr_xy[0]) ** 2 + (npc_info[3] - curr_xy[1]) ** 2)
            ] = [
                npc_info[2] - curr_xy[0],  # dx
                npc_info[3] - curr_xy[1],  # dy
                npc_info[4],  # 障碍物朝向
                numpy.sqrt(npc_info[5] ** 2 + npc_info[6] ** 2),  # 障碍物速度大小（标量）
                numpy.sqrt(npc_info[7] ** 2 + npc_info[8] ** 2),  # 障碍物加速度大小（标量）
                npc_info[9],  # 障碍物宽度
                npc_info[10],  # 障碍物长度
            ]
        # 按距离由近至远排序
        sorted_npc_info_dict = dict(sorted(npc_info_dict.items(), key=lambda x: x[0]))
        surr_obs_list = list(sorted_npc_info_dict.values())
        # 若数量不足surr_number则补齐
        for _ in range(self.surr_number - len(surr_obs_list)):
            surr_obs_list.append(list(numpy.zeros(self.surr_vec_length)))
        # 截断 
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
            (observation["player"]["target"][0] + observation["player"]["target"][4]) / 2,  # 目标区域中心位置x
            (observation["player"]["target"][1] + observation["player"]["target"][5]) / 2,  # 目标区域中心位置y
        )
        curr_xy = (observation["player"]["status"][0], observation["player"]["status"][1])  # 当前车辆位置
        delta_xy = (target_xy[0] - curr_xy[0], target_xy[1] - curr_xy[1])  # 目标区域与当前位置的绝对偏差
        curr_yaw = observation["player"]["status"][2]  # 当前朝向
        curr_velocity = observation["player"]["status"][3]  # 当前车辆后轴中心纵向速度
        curr_velocity_prime = observation["player"]["status"][4]  # 当前车辆后轴中心纵向加速度
        curr_lateral_acc = observation["player"]["status"][5]  # 当前车辆横向加速度
        curr_wheel_angle = observation["player"]["status"][6]  # 当前车辆前轮转角
        prev_steer = observation["player"]["status"][7]  # 上一个前轮转角命令
        prev_acc = observation["player"]["status"][8]  # 上一个加速度命令
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
                # np.sqrt(delta_xy[0]**2+delta_xy[1]**2),
                delta_xy[0],  # 目标区域与当前位置的偏差x
                delta_xy[1],  # 目标区域与当前位置的偏差y
                curr_yaw,  # 当前车辆的朝向角
                curr_velocity,  # 车辆后轴当前纵向速度
                curr_velocity_prime,
                curr_lateral_acc,
                curr_wheel_angle,
                prev_steer,  # 上一个前轮转角命令
                prev_acc,  # 上一个加速度命令
                current_lane_index,  # 当前所处车道的id
                current_offset,  # 车道的偏移量
            ]
        )
        self.pre_vec_obs = vec_obs

        if self.pid_controller is not None:
            self.pid_controller.update(curr_yaw)

        return vec_obs

    def assemble_ego_vec_obs(self, obs) -> torch.tensor:

        observation = deepcopy(obs)
        vec_obs = self.process_ego_vec_obs(observation)
        # 添加到末尾帧
        self.vec_deque.append(vec_obs)
        mlp_obs = numpy.array(list(self.vec_deque))
        ego_state = torch.Tensor(mlp_obs).float().unsqueeze(0)

        return ego_state

    def assemble_reward(self, observation: Dict, info: Dict, balance=None) -> float:
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

        if info["collided"]:
            end_reward = -200
        elif info["reached_stoparea"]:
            end_reward = 200
        elif info["timeout"]:
            end_reward = -200
        else:
            end_reward = 0.0

        if observation['map'] is not None:
            lane_list = []
            for lane_info in observation["map"].lanes:
                lane_list.append(lane_info.lane_id)
            current_lane_index = lane_list.index(observation["map"].lane_id)
            current_offset = observation["map"].lane_offset
            speed_limit = observation["map"].lanes[current_lane_index].speed_limit
        else:
            logger.info('map in obs is None!')
            current_lane_index = -1
            current_offset = 0
            speed_limit = 27.78

        # add penalty when reaching close to other cars (same lane)
        length = observation["player"]['property'][1]  # 车辆长度
        width = observation["player"]['property'][0]  # 车辆宽度
        npc_info = observation['npcs']
        same_lane_npcs = []  # 同车道 npc
        neighbor_lane_npcs = []  # 不同车道 npc
        for npc in npc_info:
            if npc[0] == 0:
                break
            dy = npc[3] - observation["player"]['status'][1]
            dx = npc[2] - observation["player"]['status'][0]
            if np.abs(dy) < (width + npc[-2]) / 2:
                same_lane_npcs.append(npc)
            if np.abs(dx) < (length + npc[-1]) / 2:  # (车长 + npc车长)/2
                neighbor_lane_npcs.append(npc)

        collide_reward_1 = 0
        for npc in same_lane_npcs:
            safe_distance = (npc[-1] + length) / 2 + length
            dx = npc[2] - observation["player"]['status'][0]
            if dx < 0:
                if np.abs(dx) < safe_distance:
                    penalty = -0.1 - (safe_distance - np.abs(dx)) * 0.1
                    collide_reward_1 = min(collide_reward_1, penalty)

        # add penalty when reaching close to other cars (different lane)
        collide_reward_2 = 0
        for npc in neighbor_lane_npcs:
            safe_distance = ((npc[-2]) + width) / 2 + 0.5
            dy = npc[3] - observation["player"]['status'][1]
            if np.abs(dy) < safe_distance:
                penalty = -0.1 - (safe_distance - np.abs(dy)) * 1
                collide_reward_2 = min(collide_reward_2, penalty)

        collide_reward = collide_reward_1 + collide_reward_2
        if info['collided']:
            collide_reward -= 10

        speed = observation["player"]["status"][3]
        acc_y = observation["player"]["status"][5]
        acc_x = observation["player"]["status"][4]

        last_acc_y = self.last_obs["player"]["status"][5]
        last_acc_x = self.last_obs["player"]['status'][4]

        acc_y_dealta = (acc_y - last_acc_y) / self.args.dt
        acc_x_dealta = (acc_x - last_acc_x) / self.args.dt

        acc_prime_reward = 0
        acc_reward = 0
        speed_reward = 0
        offset_reward = 0

        # add penalty when make big turn 
        if np.abs(acc_y) > 4:
            acc_reward += -1
        elif np.abs(acc_y) > 2:
            acc_reward += -0.4 - (np.abs(acc_y) - 2) * 0.3
        if np.abs(acc_y_dealta) > 0.9:
            acc_prime_reward += -1
        elif np.abs(acc_y_dealta) > 0.7:
            acc_prime_reward += -0.4 - (np.abs(acc_y_dealta) - 0.7) * 3

        # add penalty when make big jerk (x axis)
        if np.abs(acc_x_dealta) > 0.9:
            acc_prime_reward += -1
        elif np.abs(acc_x_dealta) > 0.7:
            acc_prime_reward += -0.4 - (np.abs(acc_x_dealta) - 0.7) * 3

        # add penalty when speed over limit
        if speed > speed_limit:
            speed_reward += -(speed - speed_limit) * 0.1

        # add penalty when offset is too large (lane width 3.75)
        # if current_lane_index == 2:
        if np.abs(current_offset) > 0.5:
            offset_reward += -(np.abs(current_offset) - 0.5) * 1

        rule_reward = speed_reward + acc_reward + acc_prime_reward + offset_reward

        self.last_obs = observation
        base_reward = distance_reward + end_reward + step_reward
        # balance different reward 
        collide_reward_balance = 0
        rule_reward_balance = 0
        if self.stage == 2:
            rule_reward_balance = 1
        # test 
        if balance is not None:
            rule_reward_balance = balance
        total_reward = base_reward + collide_reward_balance * collide_reward + rule_reward_balance * rule_reward
        info = dict(base_reward=base_reward, collide_reward=collide_reward, rule_reward=rule_reward)

        return total_reward, info
    
    def pid_control(self):
        
        if self.pid_controller.step == 0 : 
            steer = self.pid_controller.absolute_pid_calculate()
        else:
            steer = self.pid_controller.incre_pid_calculate()

        steer = min(steer, np.pi/360)
        steer = max(steer, -np.pi/360)
        return steer

    def get_available_actions(self, observation):

        actions = np.array(list(self.actions_map.values()))

        curr_xy = (observation["player"]["status"][0], observation["player"]["status"][1])  # 当前车辆位置

        car_polygon = get_polygon(
            center=curr_xy,
            length=observation["player"]['property'][1],  # 车辆长度
            width=observation["player"]['property'][0],  # 车辆宽度
            theta=observation["player"]["status"][2]  # 车辆朝向角
        )

        car_x, car_y = car_polygon.exterior.xy

        steer_masks = [True, True]

        for npc_info in observation["npcs"]:
            if int(npc_info[0]) == 0:
                continue
            npc_center = (npc_info[2], npc_info[3])
            npc_width = npc_info[9]
            npc_length = npc_info[10]
            npc_theta = npc_info[4]

            npc_polygon = get_polygon(
                center=npc_center,
                length=npc_length,
                width=npc_width,
                theta=npc_theta
            )

            safe_distance = car_polygon.distance(npc_polygon)

            npc_x, npc_y = npc_polygon.exterior.xy
            block_x_range = (min(npc_x), max(npc_x))
            block_y_range = (min(npc_y), max(npc_y))

            if block_x_range[0] < min(car_x) < block_x_range[1] or block_x_range[0] < max(car_x) < block_x_range[
                1]:  # 如果 car 和 npc 并排前行
                if min(car_y) < block_y_range[0]:  # car 在 npc 的左侧
                    if safe_distance < 0.82 and observation["player"]["status"][2] > 0:  # 小于安全距离并且车头仍朝右
                        steer_masks[1] = False  # 右转屏蔽
                elif max(car_y) > block_y_range[1]:  # car 在 npc 的右侧
                    if safe_distance < 0.82 and observation["player"]["status"][2] < 0:  # 小于安全距离并且车头仍朝左
                        steer_masks[0] = False  # 左转屏蔽

        if min(car_y) < 1:  # 车辆压左线
            steer_line_mask = actions[:, 0] < 0
        elif max(car_y) > 1 + 3.75 * 3:  # 车辆压右线
            steer_line_mask = actions[:, 0] > 0
        else:
            steer_line_mask = np.ones((len(self.actions_map),), dtype=np.bool_)

        if not steer_masks[0] and steer_masks[1]:  # 左侧有障碍物，车头仍朝左
            lateral_steer_mask = actions[:, 0] < 0
        elif steer_masks[0] and not steer_masks[1]:  # 右侧有障碍物，车头仍朝右
            lateral_steer_mask = actions[:, 0] > 0
        else:
            lateral_steer_mask = np.ones((len(self.actions_map),), dtype=np.bool_)

        mask = steer_line_mask & lateral_steer_mask

        return torch.from_numpy(mask).unsqueeze(0)

    def reset(self, initial_obs):
        self.prev_distance = None
        self.pre_vec_obs = None
        self.pre_lane_index = 1
        self.pre_offset = 0
        self.last_obs = initial_obs
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
        ego_vec_state = torch.from_numpy(numpy.array(list(self.vec_deque))).unsqueeze(0)  # [1,5,8]
        sur_vec_state = torch.from_numpy(numpy.array(list(self.surr_vec_deque))).unsqueeze(0)  # [1,5,8]
        available_actions = self.get_available_actions(initial_obs)
        if self.pid_controller is not None:
            self.pid_controller.initial()
        return ego_vec_state, sur_vec_state, available_actions

    @staticmethod
    def judge_for_adjust_steer(observation):
        """
        在环境根据observation获得下次执行的动作之前，判断该观测是否满足执行pid调节角度的条件
        1、当前车道位于第一车道
        2、当前车辆所在车道其安全距离内无其他车辆
        3、进入当前车道时，车辆的航向角与道路中心线夹角大于某个阈值
        满足条件1、2、3切换至pid调节转向角，至车辆航向角稳定至0附近某个很小区间
        """
        # 获取当前所在车道
        if observation['map'] is not None:
            lane_list = []
            for lane_info in observation["map"].lanes:
                lane_list.append(lane_info.lane_id)
            current_lane_index = lane_list.index(observation["map"].lane_id)
            current_offset = observation["map"].lane_offset

        else:
            logger.info('map in obs is None!')
            current_lane_index = -1

        if current_lane_index != 2:
            return False
        else:
            # 判断当前车辆的航向角是否与车道线夹角大于阈值
            if abs(abs(observation['player']['status'][2]) - np.pi) < np.pi / 720:
                return False
            if abs(abs(observation['player']['status'][2]) - np.pi) > np.pi / 18:
                return False
            if current_offset > 0.5:
                return False

            # 处理当前车辆与npc之间的关系
            curr_xy = (observation["player"]["status"][0],  # 车辆后轴中心位置 x
                       observation["player"]["status"][1])  # 车辆后轴中心位置 y

            width = observation["player"]['property'][0]  # 车辆宽度
            same_lane_npcs = []  # 同车道 npc

            # 获得同车道npc的位置信息
            for npc_info in observation["npcs"]:
                if int(npc_info[0]) == 0:
                    break

                dy = npc_info[3] - observation["player"]['status'][1]
                if np.abs(dy) < (width + npc_info[-2]) / 2:
                    same_lane_npcs.append(npc_info)

            min_distance_from_forward = np.inf
            for npc_info in same_lane_npcs:
                # 处理同车道npc之间的距离关系
                npc_x = npc_info[2]

                # 忽略后方车辆
                if npc_x >= curr_xy[0]:
                    continue

                # 观测范围内前方车辆的距离
                if abs(npc_x - curr_xy[0]) < min_distance_from_forward:
                    min_distance_from_forward = abs(npc_x - curr_xy[0])

            if min_distance_from_forward <= 15:
                # 紧急情况，不需要调整，依靠学习策略
                return False
            return True
