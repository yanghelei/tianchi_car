# ******************************************************************************
# * Copyright (C) Alibaba-inc - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# *****************************************************************************

from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class PolicyParam:
    seed: int = 1234

    num_workers: int = 1
    num_episode: int = 100000
    batch_size: int = 50
    minibatch_size: int = 10
    num_epoch: int = 10
    save_num_episode: int = 100
    log_num_episode: int = 10

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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    schedule_adam: str = "fix"
    schedule_clip: str = "fix"
    use_target_kl: bool = True
    advantage_norm: bool = True
    use_clipped_value_loss: bool = True
    lossvalue_norm: bool = False

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
