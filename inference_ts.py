# -*- coding : utf-8 -*-
# @Time     : 2022/8/26 下午6:45
# @Author   : gepeng
# @ FileName: inference_ts.py
# @ Software: Pycharm

import os

import gym
import numpy as np
from multiprocessing import Pool

import sys

sys.path.append(os.path.join(os.getcwd(), 'tianchi_car'))
sys.path.append(os.path.join(os.getcwd(), 'tianchi_car', 'train_ts'))

from geek.env.logger import Logger
from geek.env.matrix_env import DoneReason, Scenarios
from train_ts.networks import MyActor, MyCritic
from algo_ts.policy import DiscreteSACPolicy
from algo_ts.data.batch import Batch
import torch
import torch.nn as nn
from train_ts.config import Config
from networks_tools.norm import Normalization

logger = Logger.get_logger(__name__)


def load_model(config):
    policy_path, sur_norm_path, ego_norm_path = get_model_path()

    # 设置norm
    sur_norm = Normalization(input_shape=config.max_consider_nps * config.sur_in)
    ego_norm = Normalization(input_shape=1 * config.ego_in)

    sur_norm.load_model(sur_norm_path, device='cpu')
    ego_norm.load_model(ego_norm_path, device='cpu')

    # model
    # MLP网络作为前置网络
    actor = MyActor(
        config=config,
        sur_norm=sur_norm,
        ego_norm=ego_norm,
        norm_layer=nn.LayerNorm,
        device=config.device,
    ).to(config.device)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    critic1 = MyCritic(config,
                       sur_norm=sur_norm,
                       ego_norm=ego_norm,
                       norm_layer=nn.LayerNorm,
                       device=config.device).to(config.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=config.critic_lr)

    critic2 = MyCritic(config,
                       sur_norm=sur_norm,
                       ego_norm=ego_norm,
                       norm_layer=nn.LayerNorm,
                       device=config.device).to(config.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=config.critic_lr)

    # 离散不建议使用自动
    if config.auto_alpha:
        target_entropy = 0.98 * np.log(np.prod(config.action_shape))
        log_alpha = torch.zeros(1, requires_grad=True, device=config.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=config.alpha_lr)
        config.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = DiscreteSACPolicy(
        config,
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=config.tau,
        gamma=config.gamma,
        alpha=config.alpha,
        reward_normalization=config.rew_norm,
    )
    # 加载模型
    policy.load_state_dict(torch.load(policy_path))

    return policy


def get_model_path():
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    policy_path = os.path.join(project_path, 'tianchi_car', 'train_ts', 'log', 'MatrixEnv-v1', 'sac', 'policy.pth')
    sur_norm_path = os.path.join(project_path, 'tianchi_car', 'train_ts', 'log', 'MatrixEnv-v1', 'sac', 'sur_norm.pth')
    ego_norm_path = os.path.join(project_path, 'tianchi_car', 'train_ts', 'log', 'MatrixEnv-v1', 'sac', 'ego_norm.pth')

    return policy_path, sur_norm_path, ego_norm_path


def run(worker_index):
    try:
        config = Config()

        env = gym.make("MatrixEnv-v1", scenarios=Scenarios.INFERENCE)
        if hasattr(env, 'action_space'):
            setattr(env, 'action_space', config.action_space)

        obs = env.reset()

        policy = load_model(config=config)
        # 装载初始数据
        data = Batch(
            obs={}, act={}, rew={}, done={}, obs_next={}, info={}, policy={}
        )
        data.obs = obs

        while True:
            result = policy(data, None)
            data.update(act=result.act)
            act = policy.map_action(data)[0]
            obs, reward, done, info = env.step(act)
            infer_done = DoneReason.INFERENCE_DONE == info.get("DoneReason", "")
            if done and not infer_done:
                obs = env.reset()
            elif infer_done:
                break
    except Exception as e:
        logger.info(f"{worker_index}, error: {str(e)}")


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    #
    # num_workers = 12
    #
    # pool = Pool(num_workers)
    # pool_result = pool.map_async(run, list(range(num_workers)))
    # pool_result.wait(timeout=3000)
    #
    # logger.info("inference done.")
    run(worker_index=0)
