import torch
import numpy as np
from math import pi
from easydict import EasyDict

config = dict(
    exp_name='ppo',

    seed=1,
    task='MatrixEnv-v1',
    reward_threshold=None,


    actor_lr=1e-4,
    critic_lr=1e-3,

    alpha_lr=3e-4,

    gamma=0.99,
    tau=0.001,

    alpha=0.05,
    auto_alpha=True,

    epoch=1e7,
    step_per_epoch=7500,  # the number of transitions collected per epoch
    step_per_collect=750,  # trainer will collect "step_per_collect" transitions and do some policy network update repeatedly in each epoch.
    update_per_step=0.2,

    batch_size=256,

    training_num=15,  # 用于训练的环境数目

    logdir='/myspace/sac_v0.1.0',
    render=0.0,

    rew_norm=False,
    n_step=5,  # the number of steps to look ahead. Default to 1.

    device='cuda' if torch.cuda.is_available() else 'cpu',

    per=dict(
        buffer_size=3e5,
        prioritized_replay=True,
        alpha=0.6,
        beta=0.4,
        beta_final=1.0,
    ),

    resume=True,
    save_interval=1,

    steer_prime_choices=np.array([-pi/18, 0, pi/18]),  # np.linspace(-pi/18, pi/18, 3)
    acc_prime_choice=np.array([-0.8, 0, 0.8]),
    action_per_dim=(3, 3),

    network=dict(
        sur_in=9,
        sur_hiddens=(64, 128),
        sur_out=128,

        ego_in=11,
        ego_hiddens=(64, 64),
        ego_out=64,

        total_hiddens=(256, 128),

        action_hidden=(256, 128),
    ),

    dt=0.1,
    history_length=5,
    max_consider_nps=10,
)

cfg = EasyDict(config)


if __name__ == '__main__':
    a = np.linspace(-pi/3.6, pi/3.6, 11)  # 好像还是有点大
    print(a/pi * 180 * 0.1)
