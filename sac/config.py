import torch
import numpy as np
from math import pi
from easydict import EasyDict
from copy import deepcopy

config = dict(
    exp_name='sac',

    seed=1,
    task='MatrixEnv-v1',
    reward_threshold=None,

    actor_lr=1e-4,
    critic_lr=1e-3,

    alpha_lr=3e-4,

    gamma=0.99,
    tau=0.001,

    alpha=0.05,
    auto_alpha=False,

    epoch=1e7,

    buffer_size=3e5,  # 3e5

    step_per_epoch=3e4,  # the number of transitions collected per epoch
    step_per_collect=3e3,  # trainer will collect "step_per_collect" transitions and do some policy network update repeatedly in each epoch.
    min_episode_per_collect=15 * 5,
    update_per_step=3,  # default: 0.2

    batch_size=256,

    training_num=15,  # 用于训练的环境数目

    logdir='/myspace/sac_v1.0.0',
    render=0.0,

    rew_norm=False,
    n_step=5,  # the number of steps to look ahead. Default to 1.

    device='cuda' if torch.cuda.is_available() else 'cpu',

    resume=True,
    save_interval=1,

    steer_prime_choices=np.array([-pi / 18, 0, pi / 18]),  # np.linspace(-pi/18, pi/18, 3)
    acc_prime_choice=np.array([-0.8, 0, 0.8]),
    action_per_dim=(3, 3),

    network=dict(
        sur_in=8,
        sur_hiddens=[],
        sur_out=64,

        ego_in=11,
        ego_hiddens=[],
        ego_out=32,

        frame_out=128,
        frame_hiddens=[],

        time_out=128,
        time_hiddens=[],
    ),

    dt=0.1,
    history_length=10,
    max_consider_nps=7,

    car=dict(
        width=2.11,
        length=3.89 * 2
    ),

    dangerous_distance=1,

)

cfg = EasyDict(config)

_debug_cfg = deepcopy(cfg)
_debug_cfg.buffer_size = 3e2
_debug_cfg.step_per_epoch = 3e1
_debug_cfg.step_per_collect = 16
_debug_cfg.min_episode_per_collect = 1
_debug_cfg.training_num = 4

debug_cfg = _debug_cfg

