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

from geek.env.logger import Logger
from geek.env.matrix_env import Scenarios, DoneReason
from train.tools import EnvPostProcsser

Transition = namedtuple(
    "Transition",
    ("sur_obs", "vec_obs", "value", "action", 'gaussian_action', "logproba", "mask", "reward", "info",),
)
Get_Enough_Batch = mp.Value("i", 0) # 标志位：是否采集了足够的样本数

logger = Logger.get_logger(__name__)


def make_env(render_id: str):
    env = gym.make("MatrixEnv-v1", scenarios=Scenarios.TRAINING)
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
    def __init__(self, remote, queue, lock, seed, worker_index):
        super(EnvWorker, self).__init__()
        self.worker_index = worker_index
        self.remote = remote
        self.queue = queue
        self.lock = lock

        self.env_post_processer = EnvPostProcsser()
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    @staticmethod
    def lmap(v: float, x: List, y: List) -> float:
        return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])

    def run(self):
        self.env = make_env(self.worker_index)
        env_pid = -1
        while True:
            command, policy = self.remote.recv()
            if command == "sample":
                while Get_Enough_Batch.value == 0:
                    try:
                        episode = Episode()
                        obs = self.env.reset()
                        vec_state, env_state = self.env_post_processer.reset(obs)
                        while Get_Enough_Batch.value == 0:
                            with torch.no_grad():
                                action, gaussian_action, logproba, value = policy.select_action(env_state, vec_state)
                                action = action.data.cpu().numpy()[0]
                                logproba = logproba.data.cpu().numpy()[0]
                                value = value.data.cpu().numpy()[0][0]
                                gaussian_action = gaussian_action.data.cpu().numpy()[0]
                                env_state = env_state.data.cpu().numpy()[0]
                                vec_state = vec_state.data.cpu().numpy()[0]
                            # 线性映射到环境区间
                            steer = self.lmap(action[0],[-1.0, 1.0],[-0.3925, 0.3925],)
                            acc = self.lmap(action[1], [-1.0, 1.0], [-6.0, 2.0])
                            obs, reward, done, info = self.env.step(np.array([steer, acc]))
                            # 出现error则则放弃本帧的数据
                            if DoneReason.INFERENCE_DONE == info.get("DoneReason", ""):
                                if len(episode) > 0:
                                    with self.lock:
                                        self.queue.put(episode)
                                break
                            elif DoneReason.RUNTIME_ERROR == info.get("DoneReason", ""):
                                if len(episode) > 0:
                                    with self.lock:
                                        self.queue.put(episode)
                                break
                            try:
                                if not done:
                                    new_env_state = self.env_post_processer.assemble_surr_vec_obs(obs)
                                    new_vec_state = self.env_post_processer.assemble_ego_vec_obs(obs)
                                reward = self.env_post_processer.assemble_reward(obs, info)
                            except KeyError:
                                print(obs)
                                print(obs.keys())
                            mask = 0 if done else 1
                            episode.push(
                                env_state, vec_state, value, action, gaussian_action, logproba, mask, reward, info,
                            )
                            if done:
                                self.env.reset()
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
    def __init__(self, args, logger):
        self.logger = logger
        self.args = args
        self.num_workers = args.num_workers
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.device = args.device
        self.obs_type = args.obs_type

        self.queue = mp.Queue()
        self.lock = mp.Lock()

        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_workers)])
        self.workers = [
            EnvWorker(remote, self.queue, self.lock, args.seed + index, index)
            for index, remote in enumerate(self.work_remotes)
        ]

        for worker in self.workers:
            worker.daemon = True
            worker.start()

        for remote in self.work_remotes:
            remote.close()

    def sample(self, policy):
        policy.to("cpu")
        memory = Memory()
        Get_Enough_Batch.value = 0
        for remote in self.remotes:
            remote.send(("sample", policy))

        while len(memory) < self.batch_size:
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
            remote.send(("close", None))
        for worker in self.workers:
            worker.join()

    def save(self, path):
        torch.save(self.workers[0].env_post_processer.ego_vec_normalize.state_dict(), path+'/ego_norm.pth')
        torch.save(self.workers[0].env_post_processer.surr_vec_normalize.state_dict(), path+'/sur_norm.pth')
