# ******************************************************************************
# * Copyright (C) Alibaba-inc - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# *****************************************************************************

import multiprocessing as mp
import os
import time
import traceback
from collections import namedtuple
from typing import List

import gym
import numpy as np
import torch
from train.config import CommonConfig
from geek.env.logger import Logger
from geek.env.matrix_env import Scenarios, DoneReason
from train.tools import EnvPostProcsser
from train.config import PolicyParam

Transition = namedtuple(
    "Transition",
    ("sur_obs", "vec_obs", "value", "action", 'gaussian_action', "logproba", "mask", "reward", "base_reward", "collide_reward", "rule_reward", "info",),
)
Get_Enough_Batch = mp.Value("i", 0) # 标志位：是否采集了足够的样本数

logger = Logger.get_logger(__name__)


def make_env(render_id: str):
    env = gym.make("MatrixEnv-v1", scenarios=Scenarios.TRAINING, render_id=str(render_id))
    return env


class Episode(object):
    def __init__(self):
        self.episode = []

    def push(self, *args):
        self.episode.append(Transition(*args))

    def __len__(self):
        return len(self.episode)


class Memory(object):
    def __init__(self):
        self.memory = []
        self.num_episode = 0
        self.arrive_goal_num = 0

    def push(self, epi: Episode):
        self.memory += epi.episode
        self.num_episode += 1
        if epi.episode[-1][-1]["reached_stoparea"]:
            self.arrive_goal_num += 1

    def sample(self):
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


class EnvWorker(mp.Process):
    def __init__(self, remote, queue, lock, seed, worker_index, stage):
        super(EnvWorker, self).__init__()
        self.worker_index = worker_index
        self.remote = remote
        self.queue = queue
        self.lock = lock
        self.gaussian = PolicyParam.gaussian
        self.env_post_processer = EnvPostProcsser(stage)
        self.env_action_space = CommonConfig.env_action_space
        self.action_num =  CommonConfig.action_num
        self.action_repeat = PolicyParam.action_repeat
        if not self.gaussian:
            self.actions_map = self._set_actions_map(self.action_num)
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    @staticmethod
    def lmap(v: float, x: List, y: List) -> float:
        return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    @staticmethod
    def _set_actions_map(action_num):
        #dicretise action space
        forces = np.linspace(-0.7, 0.7, num=11, endpoint=True)
        thetas = np.linspace(-0.13, 0.13, num=11, endpoint=True) # 7度
        actions = [[force, theta] for force in forces for theta in thetas]
        actions_map = {i:actions[i] for i in range(action_num)}
        return actions_map 

    def action_transform(self, action, gaussian=True) -> np.array :
        if gaussian:
            low_action = self.env_action_space.low
            high_action = self.env_action_space.high
            steer = self.lmap(action[0],[-1.0, 1.0],[low_action[0], high_action[0]],)
            acc = self.lmap(action[1], [-1.0, 1.0], [low_action[1], high_action[1]])
            env_action = np.array([steer, acc])
        else:
            env_action = np.array(self.actions_map[action])

        return env_action

    def run(self):
        self.env = make_env(self.worker_index)
        env_pid = -1
        while True:
            command, policy, balance = self.remote.recv()
            if command == "sample":
                while Get_Enough_Batch.value == 0:
                    try:
                        episode = Episode()
                        obs = self.env.reset()
                        vec_state, env_state = self.env_post_processer.reset(obs)
                        while Get_Enough_Batch.value == 0:
                            with torch.no_grad():
                                action, gaussian_action, logproba, value = policy.select_action(env_state, vec_state)
                                env_state = env_state.data.cpu().numpy()[0]
                                vec_state = vec_state.data.cpu().numpy()[0]
                            # 映射到环境动作
                            if self.gaussian:
                                env_action = self.action_transform(gaussian_action)
                            else:
                                env_action = self.action_transform(action, False)
                            for _ in range(self.action_repeat):
                                obs, reward, done, info = self.env.step(env_action)
                                if done:
                                    break
                            is_runtime_error = info.get(DoneReason.Runtime_ERROR, False)
                            is_infer_error = info.get(DoneReason.INFERENCE_DONE, False)
                            # 出现error则放弃本帧数据
                            if is_infer_error or is_runtime_error:
                                logger.error(f"worker {self.worker_index}:env error!!")
                                logger.info(env_action)
                                break
                            if not done:
                                new_env_state = self.env_post_processer.assemble_surr_vec_obs(obs)
                                new_vec_state = self.env_post_processer.assemble_ego_vec_obs(obs)
                            reward, reward_dict = self.env_post_processer.assemble_reward(obs, info, balance)
                            base_reward = reward_dict['base_reward']
                            collide_reward = reward_dict['collide_reward']
                            rule_reward = reward_dict['rule_reward']
                            mask = 0 if done else 1
                            episode.push(
                                env_state, vec_state, value, action, gaussian_action, logproba, mask, reward, base_reward, collide_reward, rule_reward, info,
                            )
                            if done:
                                with self.lock:
                                    self.queue.put(episode)
                                break
                            env_state = new_env_state
                            vec_state = new_vec_state
                    except Exception as e:
                        logger.error(f"exception: {traceback.print_exc()}")

            elif command == 'eval':

                while Get_Enough_Batch.value == 0:
                    try:
                        episode = Episode()
                        obs = self.env.reset()
                        vec_state, env_state = self.env_post_processer.reset(obs)
                        while Get_Enough_Batch.value == 0:
                            with torch.no_grad():
                                action, gaussian_action, logproba, value = policy.select_action(env_state, vec_state, False)
                                env_state = env_state.data.cpu().numpy()[0]
                                vec_state = vec_state.data.cpu().numpy()[0]
                            # 映射到环境动作
                            if self.gaussian:
                                env_action = self.action_transform(gaussian_action)
                            else:
                                env_action = self.action_transform(action, False)
                            for _ in range(self.action_repeat):
                                obs, reward, done, info = self.env.step(env_action)
                                if done:
                                    break
                            # 出现error则则放弃本回合的数据
                            is_runtime_error = info.get(DoneReason.Runtime_ERROR, False)
                            is_infer_error = info.get(DoneReason.INFERENCE_DONE, False)
                            # 出现error则放弃本帧数据
                            if is_infer_error or is_runtime_error:
                                logger.error(f"worker {self.worker_index}:env error!!")
                                break
                            if not done:
                                new_env_state = self.env_post_processer.assemble_surr_vec_obs(obs)
                                new_vec_state = self.env_post_processer.assemble_ego_vec_obs(obs)
                            reward, reward_dict = self.env_post_processer.assemble_reward(obs, info)
                            base_reward = reward_dict['base_reward']
                            collide_reward = reward_dict['collide_reward']
                            rule_reward = reward_dict['rule_reward']
                            mask = 0 if done else 1
                            episode.push(
                                env_state, vec_state, value, action, gaussian_action, logproba, mask, reward, base_reward, collide_reward, rule_reward, info,
                            )
                            if done:
                                with self.lock:
                                    self.queue.put(episode)
                                break
                            env_state = new_env_state
                            vec_state = new_vec_state
                    except Exception as e:
                        logger.error(f"exception: {traceback.print_exc()}")

            elif command == "close":
                self.remote.close()
                self.env.close()
                break
            else:
                raise NotImplementedError()


class MemorySampler(object):
    def __init__(self, args, logger, stage):
        self.logger = logger
        self.args = args
        self.num_workers = args.num_workers
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.device = args.device
        self.obs_type = args.obs_type
        self.stage = stage
        self.queue = mp.Queue()
        self.lock = mp.Lock()

        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_workers)])
        self.workers = [
            EnvWorker(remote, self.queue, self.lock, args.seed + index, index, stage)
            for index, remote in enumerate(self.work_remotes)
        ]

        for worker in self.workers:
            worker.daemon = True
            worker.start()

        for remote in self.work_remotes:
            remote.close()

    def sample(self, policy, balance):
        policy.to("cpu")
        memory = Memory()
        Get_Enough_Batch.value = 0
        for remote in self.remotes:
            remote.send(("sample", policy, balance))

        while len(memory) < self.batch_size:
            episode = self.queue.get(True)
            memory.push(episode)

        Get_Enough_Batch.value = 1

        while self.queue.qsize() > 0:
            self.queue.get()

        policy.to(self.device)
        return memory

    def eval(self, policy, eval_episode, balance=0):
        policy.to('cpu')
        memory =  Memory()
        Get_Enough_Batch.value = 0
        for remote in self.remotes:
            remote.send(("eval", policy, balance))
            
        while memory.num_episode < eval_episode:
            episode = self.queue.get(True)
            memory.push(episode)

        Get_Enough_Batch.value = 1

        while self.queue.qsize() > 0:
            self.queue.get()

        policy.to(self.device)
        return memory
        
    def close(self):
        Get_Enough_Batch.value = 1
        for remote in self.remotes:
            remote.send(("close", None, None))
        for worker in self.workers:
            worker.join()
