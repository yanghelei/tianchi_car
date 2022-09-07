import gym
import torch
import itertools
import numpy as np
import collections
from math import pi, inf
from tianshou.data import Batch
from utils.math import compute_distance


class Processor:
    def __init__(self, cfgs, logger, n_env, models=None, update_norm=True):

        if models is None:
            models = list()

        self.cfgs = cfgs

        self.n_env = n_env

        self.envs_id = [i for i in range(self.n_env)]
        self.env_last_obs = [None] * self.n_env
        self.env_last_distance = [0] * self.n_env

        self.phases_distances = [[]] * self.n_env

        self.models = models

        self.max_consider_nps = cfgs.max_consider_nps
        self.sur_dim = cfgs.network.sur_in
        self.ego_dim = cfgs.network.ego_in

        self.logger = logger

        self.dt = cfgs.dt

        self.update_norm = update_norm

        steer_prime_choices = cfgs.steer_prime_choices
        acc_prime_choice = cfgs.acc_prime_choice
        self.action_library = np.array(list(itertools.product(steer_prime_choices, acc_prime_choice)))

        self.n_ego_vec_deque = [collections.deque(maxlen=self.cfgs.history_length)] * self.n_env
        self.n_ego_n_deque = [collections.deque(maxlen=self.cfgs.history_length)] * self.n_env
        self.n_sur_vec_deque = [collections.deque(maxlen=self.cfgs.history_length)] * self.n_env
        self.n_sur_n_deque = [collections.deque(maxlen=self.cfgs.history_length)] * self.n_env

        self.reset_deque(self.envs_id)

    def reset_deque(self, env_ids):
        for env_id in env_ids:
            for _idx in range(self.cfgs.history_length):
                self.n_sur_n_deque[env_id].append(1)
                self.n_sur_vec_deque[env_id].append(np.zeros((self.max_consider_nps, self.sur_dim)))
                self.n_ego_n_deque[env_id].append(1)
                self.n_ego_vec_deque[env_id].append(np.zeros((1, self.ego_dim)))

    def get_observation(self, observation, env_id=None):
        curr_xy = (observation["player"]["status"][0], observation["player"]["status"][1])  # 当前车辆位置

        target_xy = (
            (observation["player"]["target"][0] + observation["player"]["target"][4]) / 2,  # 目标区域中心位置x
            (observation["player"]["target"][1] + observation["player"]["target"][5]) / 2,  # 目标区域中心位置y
        )

        delta_xy = (target_xy[0] - curr_xy[0], target_xy[1] - curr_xy[1])  # 目标区域与当前位置的绝对偏差
        curr_yaw = observation["player"]["status"][2]  # 当前朝向
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
            [[
                delta_xy[0],  # 目标区域与当前位置的偏差x
                delta_xy[1],  # 目标区域与当前位置的偏差y
                curr_yaw,  # 当前车辆的朝向角
                curr_velocity,  # 车辆后轴当前纵向速度
                curr_lateral_acc,  # 车辆当前后轴横向加速度
                curr_steer,  # 车辆当前前轮转角
                prev_steer,  # 上一个前轮转角命令
                prev_acc,  # 上一个加速度命令(车辆当前后轴纵向加速度)
                current_lane_index,  # 当前所处车道的id
                speed_limit - curr_velocity,  # 当前车道速度上限与当前车速的差值
                current_offset,  # 车道的偏移量
            ]]
        )

        curr_xy = (observation["player"]["status"][0], observation["player"]["status"][1])  # 车辆后轴中心位置
        npc_info_dict = {}

        for npc_info in observation["npcs"]:
            if int(npc_info[0]) == 0:
                continue
            npc_info_dict[np.sqrt((npc_info[2] - curr_xy[0]) ** 2 + (npc_info[3] - curr_xy[1]) ** 2)] = [
                npc_info[2] - curr_xy[0],  # dx
                npc_info[3] - curr_xy[1],  # dy
                npc_info[4],  # 障碍物朝向
                npc_info[5],  # vx
                npc_info[6],  # vy
                # np.sqrt(npc_info[5] ** 2 + npc_info[6] ** 2) - observation["player"]["status"][3],  # 障碍物速度大小（标量） - 当前车速度大小
                npc_info[7],  # ax
                npc_info[8],  # ay
                # np.sqrt(npc_info[7] ** 2 + npc_info[8] ** 2),  # 障碍物加速度大小（标量）
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

        if self.update_norm:
            for _model in self.models:
                _model.sur_norm.update(sur_obs_list)
                _model.ego_norm.update(ego_obs)

        self.n_sur_n_deque[env_id].append(n_sur)
        self.n_sur_vec_deque[env_id].append(sur_obs_list)
        self.n_ego_n_deque[env_id].append(1)
        self.n_ego_vec_deque[env_id].append(ego_obs)

        n_sur = np.array(list(self.n_sur_n_deque[env_id]))
        sur_obs_list = np.array(list(self.n_sur_vec_deque[env_id]))
        n_ego = np.array(list(self.n_ego_n_deque[env_id]))
        ego_obs = np.array(list(self.n_ego_vec_deque[env_id]))

        # action mask module
        if curr_velocity > speed_limit and prev_acc > 0:
            # 如果【当前速度大于该条车道的限速】，并且【当前加速度大于零（车辆仍在加速状态）】
            acc_prime_mask = self.action_library[:, 1] < 0  # 速度太快，屏蔽继续加速的动作
        else:
            acc_prime_mask = np.ones((len(self.action_library),), dtype=np.bool_)
        if curr_steer < -pi / 36:  # 前轮左转大于5°，屏蔽继续左转的动作
            steer_prime_mask = self.action_library[:, 0] > 0
        elif curr_steer > pi / 36:  # 前轮右转大于5°，屏蔽继续右转的动作
            steer_prime_mask = self.action_library[:, 0] < 0
        else:
            steer_prime_mask = np.ones((len(self.action_library),), dtype=np.bool_)
        mask = acc_prime_mask & steer_prime_mask

        obs = dict(
            sur_obs=dict(
                n=n_sur,
                data=sur_obs_list
            ),
            ego_obs=dict(
                n=n_ego,
                data=ego_obs
            ),
            mask=mask
        )

        return obs

    def compute_reward(self, env_id, next_obs, info):
        target_xy = (
            (next_obs["player"]["target"][0] + next_obs["player"]["target"][4]) / 2,
            (next_obs["player"]["target"][1] + next_obs["player"]["target"][5]) / 2,
        )
        curr_xy = (next_obs["player"]["status"][0], next_obs["player"]["status"][1])
        distance_with_target = compute_distance(target_xy, curr_xy)

        assert self.env_last_distance[env_id] is not None

        distance_close = self.env_last_distance[env_id] - distance_with_target

        self.env_last_distance[env_id] = distance_with_target

        if distance_with_target < self.phases_distances[env_id][-1]:
            phase_reward = 50
            self.phases_distances[env_id].pop()
        else:
            phase_reward = 0

        step_reward = -1

        car_status = next_obs['player']['status']
        last_car_status = self.env_last_obs[env_id]['player']['status']

        fastly_brake = False
        """
            急刹
            判据: 1.纵向加速度绝对值大于2；
                 2.纵向jerk绝对值大于0.9；
            扣分: 10分每次，最高30；
        """
        car_forward_acc = car_status[4]

        if car_forward_acc != car_status[8]:
            self.logger.info(f'Now forward acc is {car_forward_acc}, last action acc is {car_status[8]}!')

        last_car_forward_acc = last_car_status[4]
        if abs(car_forward_acc) > 2 or abs((last_car_forward_acc - car_forward_acc) / self.dt) > 0.9:
            fastly_brake = True
            brake_reward = -10
        else:
            brake_reward = 0

        big_turn = False
        """
            大转向
            判据: 1.横向加速度绝对值大于4；
                 2.横向jerk绝对值大于0.9；
            扣分: 10分每次，最高30；
        """
        car_lateral_acc = car_status[5]
        last_car_lateral_acc = last_car_status[5]
        if abs(car_lateral_acc) > 4 or abs((car_lateral_acc - last_car_lateral_acc) / self.dt) > 0.9:
            big_turn = True
            turn_reward = -10
        else:
            turn_reward = 0

        # 压线 TODO: 压线的判据

        over_speed = False
        """
            超速
            判据: 1.车速超过当前车道上限的20%；
                 2.或全程平均车速超过120km/h；
            扣分: 15分每次，最高30
            TODO: 全程平均车速尚未考虑
        """
        lane_list = []
        if next_obs["map"] is not None:
            speed_limit = None
            for lane_info in next_obs["map"].lanes:
                lane_list.append(lane_info.lane_id)
                if next_obs["map"].lane_id == lane_info.lane_id:
                    speed_limit = lane_info.speed_limit
                    break
            current_lane_index = lane_list.index(next_obs["map"].lane_id)
            current_offset = next_obs["map"].lane_offset
            if speed_limit is None:
                speed_limit = 0.0
                # self.logger.info('Env:' + str(env_id) + 'Not find current lane\'s speed limit!!!\tUse inf to keep running!')
        else:  # 按照主办方的说法，车开到道路外有可能出现 none 的情况
            current_lane_index = -1.0
            current_offset = 0.0
            speed_limit = 0.0
            # self.logger.info('Env:' + str(env_id) + 'next_obs[\'map\'] is None!!!\tUse inf as speed limit to keep running!')
        car_speed = car_status[3]  # 当前车速
        if car_speed > speed_limit or car_speed < speed_limit * 0.6:  # 当前车速超过车道限速或者小于车道限速的60%
            over_speed = True
            high_speed_reward = -15
        else:
            high_speed_reward = 0

        if current_lane_index != -1:
            offset_reward = 0.3 / (abs(current_offset) + 1)
        else:
            offset_reward = 0

        if fastly_brake or big_turn or over_speed:
            rule_reward = brake_reward + turn_reward + high_speed_reward
        else:
            rule_reward = distance_close + offset_reward

        if info["collided"]:  # 碰撞
            end_reward = -1000
        elif info["reached_stoparea"]:  #
            end_reward = 1000
        # elif info["timeout"]:  # 超时未完成
        #     end_reward = -100
        else:
            end_reward = 0.0  # 尚未 terminate

        return end_reward + step_reward + rule_reward + phase_reward

    def update_distance_to_target(self, env_id):
        obs = self.env_last_obs[env_id]
        assert obs is not None
        target_xy = (
            (obs["player"]["target"][0] + obs["player"]["target"][4]) / 2,
            (obs["player"]["target"][1] + obs["player"]["target"][5]) / 2,
        )
        curr_xy = (obs["player"]["status"][0], obs["player"]["status"][1])
        self.env_last_distance[env_id] = compute_distance(target_xy, curr_xy)
        self.get_phases(env_id)

    def get_phases(self, env_id):
        phases = list()
        distance = 0
        while distance < self.env_last_distance[env_id]:
            phases.append(distance)
            distance += 100
        self.phases_distances[env_id] = phases

    def preprocess_fn(self, **kwargs):
        # assert len(kwargs['env_id']) == len(self.envs_id)
        if 'rew' in kwargs:
            obs_next = [None] * len(kwargs['env_id'])
            rew = [0] * len(kwargs['env_id'])
            for _idx, _id in enumerate(kwargs['env_id']):
                obs_next[_idx] = self.get_observation(kwargs['obs_next'][_idx], env_id=_id)
                rew[_idx] = self.compute_reward(_id, kwargs['obs_next'][_idx], kwargs['info'][_idx])
                self.env_last_obs[_id] = kwargs['obs_next'][_idx]
            obs_next = np.array(obs_next)
            rew = np.array(rew)
            return Batch(obs_next=obs_next, rew=rew, done=kwargs['done'], policy=kwargs['policy'])
        else:
            self.reset_deque(kwargs['env_id'])
            obs = [None] * len(kwargs['env_id'])
            for _idx, _id in enumerate(kwargs['env_id']):
                obs[_idx] = self.get_observation(kwargs['obs'][_idx], env_id=_id)
                self.env_last_obs[_id] = kwargs['obs'][_idx]
                self.update_distance_to_target(_id)
            obs = np.array(obs)
            return Batch(obs=obs)


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
