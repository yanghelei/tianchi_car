import torch
from math import pi
import numpy as np
from easydict import EasyDict

config = dict(
    exp_name='car_onppo_seed0',

    seed=1,
    # task='CartPole-v1',
    task='MatrixEnv-v1',
    reward_threshold=None,

    eps_train=0.1,
    eps_test=0.05,
    buffer_size=2e5,

    lr=1e-4,
    gamma=0.95,
    num_atoms=51,

    v_min=-10.0,
    v_max=10.0,

    noisy_std=0.1,
    n_step=11,  # the number of steps to look ahead. Default to 1.
    target_update_freq=200,

    epoch=10,
    step_per_epoch=1e4,  # the number of transitions collected per epoch
    step_per_collect=8,  # trainer will collect "step_per_collect" transitions and do some policy network update repeatedly in each epoch.
    update_per_step=0.125,

    batch_size=128,  # the batch size of sample data, which is going to feed in the policy network
    hidden_sizes=[128, 128],

    training_num=15,  # 用于训练的环境数目
    test_num=0,  # the number of episodes for one policy evaluation

    logdir='log',
    render=0.0,

    prioritized_replay=True,
    alpha=0.6,
    beta=0.4,
    beta_final=1.0,

    resume=True,
    save_interval=4,

    device='cuda' if torch.cuda.is_available() else 'cpu',

    action_low=np.array([-pi / 4.0, -6.0]),
    action_high=np.array([pi / 4.0, 2.0]),
    action_per_dim=(11, 11),

    surr_number=10,
    surr_vec_length=7,

)

cfg = EasyDict(config)
