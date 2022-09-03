# ******************************************************************************
# * Copyright (C) Alibaba-inc - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# *****************************************************************************

from dataclasses import dataclass, field
from typing import List

import torch
import os 
from pathlib import Path
from gym.spaces import Box
import numpy as np 

@dataclass
class PolicyParam:
    seed: int = 1234

    debug = False
    use_eval = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    if debug:
        num_workers = 1
        random_episode = 0
        num_episode = 20
        batch_size = 600
        minibatch_size = 600
        num_epoch = 3
    else:
        num_workers: int = 12
        random_episode: int = 20
        num_episode: int = 1200
        batch_size: int =  4196
        minibatch_size: int = 256
        num_epoch: int = 3

    # trick use    
    use_target_kl: bool = False
    use_advantage_norm: bool = True
    use_clipped_value_loss: bool = True
    use_value_norm: bool = True
    use_clip_grad: bool = True

    # schedule
    schedule_adam: str = "fix"
    schedule_clip: str = "fix"

    save_num_episode: int = 10
    log_num_episode: int = 10
    eval_interval: int = 100
    eval_episode: int = 100 

    # params
    gamma: float = 0.99
    lamda: float = 0.95
    loss_coeff_value: float = 0.5
    loss_coeff_entropy: float = 0.01
    max_grad_norm: float = 0.5
    clip: float = 0.2
    lr: float = 1e-4
    vf_clip_param: float = 0.5
    target_kl = 0.03
    #field 用法??
    lr_schedule: List = field(default_factory=[[0, 0.0001], [2000000, 5e-05], [3000000, 1e-05]])
    EPS: float = 1e-10

    reload = False

    model_path: str = None
    obs_type: str = "vec"
    img_width: int = 224
    img_length: int = 224
    ego_vec_length: int = 8
    surr_vec_length: int = 7
    surr_agent_number: int = 10
    history_length: int = 5
    state_norm: bool = False

    target_speed = 30
    dt = 0.1

class CommonConfig:

    remote_path = str(Path(os.path.dirname(__file__)).resolve().parent.parent / 'myspace' / 'ryd'/ 'gaussian_ppo_v2')
    env_action_space = Box(low=np.array([-0.3925, -2.0]), high=np.array([0.3925, 2.0]), dtype=np.float32)