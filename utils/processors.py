import torch
import itertools
import numpy as np
import collections
from math import pi
from tianshou.data import Batch
from utils.math import compute_distance, get_polygon
from shapely.geometry import Polygon, LineString


class EvalProcessor:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.max_consider_nps = cfgs.max_consider_nps
        self.sur_dim = cfgs.network.sur_in
        self.ego_dim = cfgs.network.ego_in

        self.dt = cfgs.dt

        steer_prime_choices = cfgs.steer_prime_choices
        acc_prime_choice = cfgs.acc_prime_choice
        self.action_library = np.array(list(itertools.product(steer_prime_choices, acc_prime_choice)))

        self.n_ego_vec_deque = collections.deque(maxlen=self.cfgs.history_length)
        self.n_ego_n_deque = collections.deque(maxlen=self.cfgs.history_length)
        self.n_sur_vec_deque = collections.deque(maxlen=self.cfgs.history_length)
        self.n_sur_n_deque = collections.deque(maxlen=self.cfgs.history_length)

        self.reset()

    def reset(self):
        for _idx in range(self.cfgs.history_length):
            self.n_sur_n_deque.append(1)
            self.n_sur_vec_deque.append(np.zeros((self.max_consider_nps, self.sur_dim)))
            self.n_ego_n_deque.append(1)
            self.n_ego_vec_deque.append(np.zeros((1, self.ego_dim)))

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

        self.n_sur_n_deque.append(n_sur)
        self.n_sur_vec_deque.append(sur_obs_list)
        self.n_ego_n_deque.append(1)
        self.n_ego_vec_deque.append(ego_obs)

        n_sur = np.array(list(self.n_sur_n_deque))
        sur_obs_list = np.array(list(self.n_sur_vec_deque))
        n_ego = np.array(list(self.n_ego_n_deque))
        ego_obs = np.array(list(self.n_ego_vec_deque))

        # action mask module
        if (curr_velocity > speed_limit and prev_acc > 0) or (curr_velocity + prev_acc * 1 > speed_limit):
            # 如果【当前速度大于该条车道的限速】，并且【当前加速度大于零（车辆仍在加速状态）】
            # 或者【按当前的加速度加速一秒后将会超过当前车道的限速，屏蔽继续加速的动作】
            acc_prime_mask = self.action_library[:, 1] < 0  # 速度太快，屏蔽继续加速的动作
        elif (curr_velocity < speed_limit * 0.6 and prev_acc < 0) or (curr_velocity + prev_acc * 1 < speed_limit * 0.6):
            # 如果【当前速度小于该条车道的限速的60%】，并且【当前加速度小于零（车辆仍在减速状态）】
            # 或者【按当前的加速度加速一秒后将会低于当前车道的限速的60%，屏蔽继续减速的动作】
            acc_prime_mask = self.action_library[:, 1] > 0  # 速度太慢，屏蔽继续减速的动作
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

        obs = Batch(obs=np.array([obs]), info={})

        return obs


class Processor:
    def __init__(self, cfgs, logger, n_env, models=None, update_norm=True):

        if models is None:
            models = list()

        self.cfgs = cfgs

        self.n_env = n_env

        self.envs_id = [i for i in range(self.n_env)]
        self.env_last_obs = [None] * self.n_env

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

        car_polygon = get_polygon(
            center=curr_xy,
            length=self.cfgs.car.length,
            width=self.cfgs.car.width,
            theta=curr_yaw
        )

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
            current_offset = 10.0
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

        npc_info_dict = {}
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

            npc_info_dict[safe_distance] = [
                safe_distance,
                npc_info[2] - curr_xy[0],  # dx
                npc_info[3] - curr_xy[1],  # dy
                npc_info[4],  # 障碍物朝向
                np.sqrt(npc_info[5] ** 2 + npc_info[6] ** 2) - observation["player"]["status"][3],  # 障碍物速度大小（标量） - 当前车速度大小
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
        if (curr_velocity > speed_limit and prev_acc > 0) or (curr_velocity + prev_acc * 1 > speed_limit):
            # 如果【当前速度大于该条车道的限速】，并且【当前加速度大于零（车辆仍在加速状态）】
            # 或者【按当前的加速度加速一秒后将会超过当前车道的限速，屏蔽继续加速的动作】
            acc_prime_mask = self.action_library[:, 1] < 0  # 速度太快，屏蔽继续加速的动作
        elif (curr_velocity < speed_limit * 0.6 and prev_acc < 0) or (curr_velocity + prev_acc * 1 < speed_limit * 0.6):
            # 如果【当前速度小于该条车道的限速的60%】，并且【当前加速度小于零（车辆仍在减速状态）】
            # 或者【按当前的加速度加速一秒后将会低于当前车道的限速的60%，屏蔽继续减速的动作】
            acc_prime_mask = self.action_library[:, 1] > 0  # 速度太慢，屏蔽继续减速的动作
        else:
            acc_prime_mask = np.ones((len(self.action_library),), dtype=np.bool_)
        # if curr_steer < -pi / 36:  # 前轮左转大于5°，屏蔽继续左转的动作
        #     steer_prime_mask = self.action_library[:, 0] > 0
        # elif curr_steer > pi / 36:  # 前轮右转大于5°，屏蔽继续右转的动作
        #     steer_prime_mask = self.action_library[:, 0] < 0
        # else:
        #     steer_prime_mask = np.ones((len(self.action_library),), dtype=np.bool_)
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
        curr_xy = (next_obs["player"]["status"][0], next_obs["player"]["status"][1])
        last_xy = (self.env_last_obs[env_id]["player"]["status"][0], self.env_last_obs[env_id]["player"]["status"][1])

        target_xy = (
            (next_obs["player"]["target"][0] + next_obs["player"]["target"][4]) / 2,
            (next_obs["player"]["target"][1] + next_obs["player"]["target"][5]) / 2,
        )

        curr_distance_with_target = compute_distance(target_xy, curr_xy)
        last_distance_with_target = compute_distance(target_xy, last_xy)
        distance_close = last_distance_with_target - curr_distance_with_target

        step_reward = -1

        car_status = next_obs['player']['status']
        last_car_status = self.env_last_obs[env_id]['player']['status']

        fastly_brake = False
        car_forward_acc = car_status[4]
        last_car_forward_acc = last_car_status[4]
        if abs(car_forward_acc) > 2 or abs((last_car_forward_acc - car_forward_acc) / self.dt) > 0.9:
            fastly_brake = True

        big_turn = False
        car_lateral_acc = car_status[5]
        last_car_lateral_acc = last_car_status[5]
        if abs(car_lateral_acc) > 4 or abs((car_lateral_acc - last_car_lateral_acc) / self.dt) > 0.9:
            big_turn = True

        speed_accept_ratio = 1.0
        lane_list = []
        if next_obs["map"] is not None:
            speed_limit = 0.0
            for lane_info in next_obs["map"].lanes:
                lane_list.append(lane_info.lane_id)
                if next_obs["map"].lane_id == lane_info.lane_id:
                    speed_limit = lane_info.speed_limit
                    break
            current_offset = next_obs["map"].lane_offset
        else:  # 按照主办方的说法，车开到道路外有可能出现 none 的情况
            current_offset = 10
            speed_limit = 1.0

        car_speed = car_status[3]  # 当前车速
        if car_speed > speed_limit:  # 当前车速超过车道限速
            over_speed_ratio = (car_speed - speed_limit) / speed_limit  # 超过规定限速的百分比
            speed_accept_ratio = max((0.2 - over_speed_ratio) / 0.2, 0)

        keep_line_center_ratio = max((0.5 - abs(current_offset)) / 0.5, 0)

        if fastly_brake or big_turn:
            rule_reward = -10
        else:
            rule_reward = distance_close * (1 + keep_line_center_ratio) * speed_accept_ratio

        if info["collided"]:  # 碰撞
            end_reward = -1000
        elif info["reached_stoparea"]:  #
            end_reward = 1000
        else:
            end_reward = 0.0  # 尚未 terminate

        car_polygon = get_polygon(
            center=curr_xy,
            length=self.cfgs.car.length,
            width=self.cfgs.car.width,
            theta=car_status[2]
        )

        npc_reward = self.get_npc_rewards(curr_xy, car_polygon, next_obs["npcs"])

        return end_reward + step_reward + rule_reward + npc_reward

    def get_npc_rewards(self, car_center, car_polygon, npc_infos):
        reward = 0
        for npc_info in npc_infos:
            if int(npc_info[0]) == 0:
                continue
            npc_center = (npc_info[2], npc_info[3])
            if compute_distance(pos_0=car_center, pos_1=npc_center) > 20:
                continue
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
            if safe_distance < self.cfgs.dangerous_distance:
                reward -= 10
        return reward

    def preprocess_fn(self, **kwargs):
        # assert len(kwargs['env_id']) == len(self.envs_id)
        if 'rew' in kwargs:
            n_ready_env = len(kwargs['env_id'])
            obs_next = [None] * n_ready_env
            rew = [0] * n_ready_env
            for idx in range(n_ready_env):
                global_env_id = kwargs['env_id'][idx]
                global_end_obs_next = kwargs['obs_next'][idx]
                global_env_info = kwargs['info'][idx]

                obs_next[idx] = self.get_observation(global_end_obs_next, env_id=global_env_id)
                rew[idx] = self.compute_reward(global_env_id, global_end_obs_next, global_env_info)

                self.env_last_obs[global_env_id] = global_end_obs_next
            obs_next = np.array(obs_next)
            rew = np.array(rew)
            return Batch(obs_next=obs_next, rew=rew, done=kwargs['done'], policy=kwargs['policy'])
        else:
            self.reset_deque(kwargs['env_id'])
            n_ready_env = len(kwargs['env_id'])
            obs = [None] * n_ready_env
            for idx in range(n_ready_env):
                global_env_id = kwargs['env_id'][idx]
                global_end_obs = kwargs['obs'][idx]
                obs[idx] = self.get_observation(global_end_obs, env_id=global_env_id)
                self.env_last_obs[global_env_id] = global_end_obs
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
