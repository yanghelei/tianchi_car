import torch
import numpy as np
from math import pi
from easydict import EasyDict

config = dict(
    exp_name='rainbow',

    seed=1,
    # task='CartPole-v1',
    task='MatrixEnv-v1',
    reward_threshold=None,

    exploration=dict(
        type='exp',
        start=0.95,
        end=0.05,
        decay=1e5
    ),
    # eps_train=0.1,
    eps_test=0.01,

    buffer_size=3e5,

    lr=1e-4,
    gamma=0.99,
    num_atoms=51,

    v_min=-10.0,
    v_max=10.0,

    noisy_std=0.1,
    n_step=5,  # the number of steps to look ahead. Default to 1.
    target_update_freq=1e4,  # v1.2: 2e4  v1.3: 2.8e3     v1.4: 1.4e3

    epoch=1e7,
    step_per_epoch=7000,  # the number of transitions collected per epoch
    step_per_collect=700,  # trainer will collect "step_per_collect" transitions and do some policy network update repeatedly in each epoch.
    update_per_step=0.2,

    batch_size=256,  # the batch size of sample data, which is going to feed in the policy network
    # hidden_sizes=[128, 128],

    training_num=14,  # 用于训练的环境数目
    test_num=20,  # the number of episodes for one policy evaluation

    logdir='/myspace/rainbow_v2.0',
    render=0.0,

    prioritized_replay=True,
    alpha=0.6,
    beta=0.4,
    beta_final=1.0,

    resume=True,
    save_interval=1,

    device='cuda' if torch.cuda.is_available() else 'cpu',

    action_low=np.array([-pi / 4.0, -0.88]),
    action_high=np.array([pi / 4.0,  0.88]),
    action_per_dim=(11, 3),

    network=dict(
        sur_dim=7,
        sur_hidden=128,
        ego_dim=9,
        ego_hidden=64,
        total_hidden=256,
        action_hidden=(256, 128),
    ),

    max_consider_nps=10,
    dt=0.1
    # history_length=5,

)

cfg = EasyDict(config)
