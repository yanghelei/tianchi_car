# -*- coding : utf-8 -*-
# @Time     : 2022/8/31 下午9:58
# @Author   : gepeng
# @ FileName: config.py
# @ Software: Pycharm

from dataclasses import dataclass
import gym.spaces as spaces
import numpy as np
import torch.cuda
import os
from pathlib import Path


@dataclass
class Config:
    task = 'MatrixEnv-v1'
    seed = 4366
    reward_threshold = np.inf
    buffer_size = 100000
    actor_lr = 3e-4
    critic_lr = 1e-3
    alpha_lr = 3e-4
    gamma = 0.9
    tau = 0.005
    alpha = 0.05
    auto_alpha = 0
    start_timesteps = 10000
    epoch = 50
    step_per_epoch = 30000
    step_per_collect = 3000
    update_per_step = 1
    batch_size = 256
    hidden_sizes = [256]
    training_num = 15
    test_num = 15
    logdir = 'log'
    render = False
    resume_path = None
    resume_id = None
    rew_norm = False
    n_step = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    max_consider_nps = 6

    history_length = 3
    sur_in = 9
    ego_in = 11

    low = [-np.pi / 4, -2]
    high = [np.pi / 4, 2]
    action_per_dim = (3, 3)

    # 离散
    action_space = spaces.Discrete(n=int(np.prod(action_per_dim)))
    action_shape = action_space.n


class CommonConfig:
    remote_path = str(Path(os.path.dirname(__file__)).resolve().parent.parent.parent / 'myspace' / 'gp' / 'sac_v0')
