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


from geek.env.logger import Logger
from geek.env.matrix_env import DoneReason, Scenarios
from networks import MyActor, MyCritic
from algo.policy import DiscreteSACPolicy
from algo.data.batch import Batch
from wrapper import Processor
import torch
from config import Config, CommonConfig
from norm import Normalization

logger = Logger.get_logger(__name__)

model_dir = CommonConfig.remote_path


def load_model(config):
    # 本地
    # policy_path = get_model_path()
    # 云端
    policy_path = model_dir + f'/policy.pth'

    # 设置norm
    sur_norm = Normalization(input_shape=config.max_consider_nps * config.sur_in)
    ego_norm = Normalization(input_shape=1 * config.ego_in)

    # model
    # MLP网络作为前置网络
    actor = MyActor(
        config=config,
        sur_norm=sur_norm,
        ego_norm=ego_norm,
    )
    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    critic1 = MyCritic(config,
                       sur_norm=sur_norm,
                       ego_norm=ego_norm,
                       )
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=config.critic_lr)

    critic2 = MyCritic(config,
                       sur_norm=sur_norm,
                       ego_norm=ego_norm,
                       )
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
    ).to(device=config.device)

    # 加载模型
    policy.load_state_dict(torch.load(policy_path))
    logger.info('model has been successfully loaded!')

    return policy, sur_norm, ego_norm


def get_model_path():
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    policy_path = os.path.join(project_path, 'tianchi_car', 'train', 'log', 'MatrixEnv-v1', 'sac', 'policy.pth')

    return policy_path


def run(worker_index):
    try:
        reach_goal = 0
        episode = 0
        logger.info(f'worker_{worker_index} starting')

        config = Config()

        env = gym.make("MatrixEnv-v1", scenarios=Scenarios.INFERENCE)
        if hasattr(env, 'action_space'):
            setattr(env, 'action_space', config.action_space)

        obs = env.reset()

        policy, sur_norm, ego_norm = load_model(config=config)
        policy.train(mode=False)
        processor = Processor(config, n_env=1, norms=[sur_norm, ego_norm], update_norm=False)

        while True:
            obs = processor.get_observation(obs, env_id=0)
            data = Batch(obs=np.array([obs]), info={})

            result = policy(data)
            data.update(act=result.act)
            act = policy.map_action(data.act)[0]
            next_obs, reward, done, info = env.step(act)
            obs = next_obs

            is_runtime_error = info.get(DoneReason.Runtime_ERROR, False)
            is_infer_error = info.get(DoneReason.INFERENCE_DONE, False)
            # 出现error则放弃本帧数据
            if is_infer_error or is_runtime_error:
                break

            if done:
                obs = env.reset()
                episode += 1

                processor.reset_processor(env_ids=[0])
                if info['reached_stoparea']:
                    logger.info('succeed !')
                    reach_goal += 1

                logger.info(f'{worker_index}, reach_goal_rate: {reach_goal}/{episode}')
                logger.info(f"worker_{worker_index}: done!")

    except Exception as e:
        logger.info(f"{worker_index}, error: {str(e)}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    num_workers = 12

    pool = Pool(num_workers)
    pool_result = pool.map_async(run, list(range(num_workers)))
    pool_result.wait(timeout=3000)

    logger.info("inference done.")
