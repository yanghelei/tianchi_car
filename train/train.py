# -*- coding : utf-8 -*-
# @Time     : 2022/8/25 下午1:14
# @Author   : gepeng
# @ FileName: main.py.py
# @ Software: Pycharm


import os
import numpy as np
import torch
import gym
from torch.utils.tensorboard import SummaryWriter
import sys

sys.path.append(os.path.join(os.getcwd(), 'tianchi_car'))

from algo.data import Collector, VectorReplayBuffer
from algo.env import SubprocVectorEnv
from algo.policy import DiscreteSACPolicy
from algo.utils import TensorboardLogger

from networks import MyActor, MyCritic

from geek.env.matrix_env import Scenarios
from wrapper import Processor
from config import Config, CommonConfig
from norm import Normalization
from geek.env.logger import Logger

from mytrainer import my_trainer

logger_i = Logger.get_logger(__name__)


class DiscreteSACPolicyTrainer:
    def __init__(self, config):
        super(DiscreteSACPolicyTrainer, self).__init__()
        self.config = config

    def make_envs(self, mode='train', render_id=None):
        if mode == 'train':
            if render_id is not None:
                env = gym.make(self.config.task, scenarios=Scenarios.TRAINING, render_id=str(render_id))
            else:
                env = gym.make(self.config.task, scenarios=Scenarios.TRAINING)
        else:
            env = gym.make(self.config.task, scenarios=Scenarios.TRAINING)

        if not hasattr(env, 'action_space'):
            setattr(env, 'action_space', self.config.action_space)

        if env.action_space is None:
            setattr(env, 'action_space', self.config.action_space)

        return env

    def train(self):
        # 线上运行补充参数，render_id=i
        train_envs = SubprocVectorEnv(
            [lambda i=_i: self.make_envs(mode='train') for _i in range(self.config.training_num)]
        )

        # seed
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        sur_norm = Normalization(input_shape=self.config.max_consider_nps * self.config.sur_in)
        ego_norm = Normalization(input_shape=1 * self.config.ego_in)

        # model
        # MLP网络作为前置网络
        actor = MyActor(
            config=self.config,
            sur_norm=sur_norm,
            ego_norm=ego_norm,
        )

        actor_optim = torch.optim.Adam(actor.parameters(), lr=self.config.actor_lr)

        critic1 = MyCritic(
            self.config,
            sur_norm=sur_norm,
            ego_norm=ego_norm
        )
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=self.config.critic_lr)

        critic2 = MyCritic(
            self.config,
            sur_norm=sur_norm,
            ego_norm=ego_norm,
        )
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=self.config.critic_lr)

        # 离散不建议使用自动
        if self.config.auto_alpha:
            target_entropy = 0.98 * np.log(np.prod(self.config.action_shape))
            log_alpha = torch.zeros(1, requires_grad=True, device=self.config.device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=self.config.alpha_lr)
            self.config.alpha = (target_entropy, log_alpha, alpha_optim)

        policy = DiscreteSACPolicy(
            self.config,
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            tau=self.config.tau,
            gamma=self.config.gamma,
            alpha=self.config.alpha,
            reward_normalization=self.config.rew_norm,
        ).to(device=self.config.device)

        # load a previous policy
        if self.config.resume_path:
            policy.load_state_dict(torch.load(self.config.resume_path, map_location=self.config.device))
            print("Loaded agent from: ", self.config.resume_path)

        train_processor = Processor(self.config, n_env=self.config.training_num, norms=[sur_norm, ego_norm])

        # collector
        train_collector = Collector(
            policy,
            train_envs,
            VectorReplayBuffer(self.config.buffer_size, len(train_envs)),
            exploration_noise=True,
            preprocess_fn=train_processor.preprocess_fn
        )

        # 进行一定数量的随机漫步，初始化norm的标准差和方差
        # if not self.config.resume_path:
        #     train_collector.collect(n_step=self.config.batch_size * self.config.training_num, random=True)
        #     logger_i.info('finished random step!')

        # log
        log_path = os.path.join(self.config.logdir, self.config.task, 'sac')
        writer = SummaryWriter(log_path)
        logger = TensorboardLogger(writer)

        def save_best_fn(policy):
            # 存储到云端
            torch.save(
                policy.state_dict(), remote_path + f"/policy.pth"
            )
            logger_i.info(f'model has been successfully saved : {remote_path}/policy.pth')

        # trainer
        my_trainer(
            policy,
            train_collector,
            None,
            self.config.epoch,
            self.config.step_per_epoch,
            self.config.step_per_collect,
            self.config.test_num,
            self.config.batch_size,
            update_per_step=self.config.update_per_step,
            save_best_fn=save_best_fn,
            logger=logger,
            test_in_train=False  # 不使用训练中测试
        )


if __name__ == '__main__':
    s = DiscreteSACPolicyTrainer(Config)
    remote_path = CommonConfig.remote_path
    os.makedirs(remote_path, exist_ok=True)
    s.train()
