# ******************************************************************************
# * Copyright (C) Alibaba-inc - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# *****************************************************************************

from dataclasses import dataclass
from typing import List

import torch
import os 
from pathlib import Path
from gym.spaces import Box
import numpy as np 

@dataclass
class PolicyParam():
    seed: int = 1234

    debug = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gaussian = False # 动作是否连续
    action_repeat: int = 1 

    if debug:
        use_eval = True 
        num_workers = 1
        random_episode = 0
        num_episode = 20
        batch_size = 600
        minibatch_size = 600
        num_epoch = 3
        save_num_episode = 1
        eval_episode: int = 1
        eval_interval: int = 1
        log_num_episode: int = 1
    else:
        use_eval = True
        num_workers: int = 15
        warmup_episode: int = 10
        num_episode: int = 1200
        batch_size: int = 5120
        minibatch_size: int = 512
        num_epoch: int = 3
        save_num_episode = 100
        eval_episode: int = 200
        eval_interval: int = 100
        log_num_episode: int = 10

    # trick use    
    use_target_kl: bool = False
    use_advantage_norm: bool = True
    use_clipped_value_loss: bool = True
    use_value_norm: bool = True
    use_clip_grad: bool = True
    share: bool = True
    independent_std: bool = True

    # params
    gamma: float = 0.99
    lamda: float = 0.95
    loss_coeff_value: float = 0.5
    max_grad_norm: float = 0.5
    vf_clip_param: float = 0.5
    target_kl = 0.03 
    # schedule
    learning_rate_schedule={
            "initial": 1e-4,
            "final": 1e-5,
            "power": 1.0,
            "max_decay_steps": 10000
            }
    beta_schedule={
            "initial": 0.01,
            "final": 0.001,
            "power": 1.0,
            "max_decay_steps": 10000
            }
    clip_range_schedule={
        "initial": 0.2,
        "final": 0.1,
        "power": 1.0,
        "max_decay_steps": 10000
        }
    # testing yet
    balance_schedule={
        "initial": 0.1,
        "final": 1,
        "power": 1.0,
        "max_decay_steps": 10000
        }
    EPS: float = 1e-10

    reload = False
    if gaussian:
        model_path: str = str(Path(os.path.dirname(__file__)).resolve().parent.parent / 'myspace' / 'ryd'/ 'gaussian_ppo')
    else:
        model_path: str = str(Path(os.path.dirname(__file__)).resolve().parent.parent / 'myspace' / 'ryd'/ 'categorical_ppo')
    obs_type: str = "vec"
    img_width: int = 224
    img_length: int = 224
    ego_vec_length: int = 11
    surr_vec_length: int = 7
    surr_agent_number: int = 10
    history_length: int = 5
    state_norm: bool = False

    target_speed = 30
    dt = 0.1

class CommonConfig:

    if PolicyParam.gaussian:
        remote_path: str = str(Path(os.path.dirname(__file__)).resolve().parent.parent / 'myspace' / 'ryd'/ 'gaussian_ppo')
    else:
        remote_path = str(Path(os.path.dirname(__file__)).resolve().parent.parent / 'myspace' / 'ryd'/ 'categorical_ppo')
    env_action_space = Box(low=np.array([-0.13, -0.7]), high=np.array([0.13, 0.7]), dtype=np.float32)
    action_num = 121