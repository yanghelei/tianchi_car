import torch
import numpy as np
from math import pi
from easydict import EasyDict

from copy import deepcopy

config = dict(
    exp_name='rainbow',

    seed=1,
    task='MatrixEnv-v1',
    reward_threshold=None,

    exploration=dict(
        type='exp',
        start=0.95,
        end=0.05,
        decay=5e5
    ),
    # eps_train=0.1,
    eps_test=0.01,

    buffer_size=1e6,

    lr=1e-4,
    gamma=0.99,
    num_atoms=51,

    v_min=-10.0,
    v_max=10.0,

    noisy_std=0.15,
    n_step=5,  # the number of steps to look ahead. Default to 1.
    target_update_freq=3e4,  # 3e4

    epoch=1e7,
    step_per_epoch=3e4,  # the number of transitions collected per epoch
    step_per_collect=3e3,  # trainer will collect "step_per_collect" transitions and do some policy network update repeatedly in each epoch.

    min_episode_per_collect=15 * 3,

    update_per_step=1,

    batch_size=256,  # the batch size of sample data, which is going to feed in the policy network
    # hidden_sizes=[128, 128],

    training_num=15,  # 用于训练的环境数目
    test_num=0,  # the number of episodes for one policy evaluation

    logdir='/myspace/rainbow_v5.3.0',
    render=0.0,

    prioritized_replay=True,
    alpha=0.6,
    beta=0.4,
    beta_final=1.0,

    resume=True,
    resume_buffer=True,
    save_interval=1,

    device='cuda' if torch.cuda.is_available() else 'cpu',

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

        action_hiddens=[],
    ),

    max_consider_nps=7,
    dt=0.1,
    history_length=5,

    car=dict(
        width=2.11,
        length=3.89 * 2
    ),

    dangerous_distance=1,

)

cfg = EasyDict(config)

if __name__ == '__main__':
    a = np.linspace(-pi / 3.6, pi / 3.6, 11)  # 好像还是有点大
    print(a / pi * 180 * 0.1)

_debug_cfg = deepcopy(cfg)
_debug_cfg.buffer_size = 3e2
_debug_cfg.step_per_epoch = 3e1
_debug_cfg.step_per_collect = 16
_debug_cfg.min_episode_per_collect = 1
_debug_cfg.training_num = 4

debug_cfg = _debug_cfg