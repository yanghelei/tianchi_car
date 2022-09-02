# ******************************************************************************
# * Copyright (C) Alibaba-inc - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# *****************************************************************************

import os
import sys
import gym
import torch
from multiprocessing import Pool

sys.path.append(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + '/')
from geek.env.logger import Logger
from geek.env.matrix_env import DoneReason, Scenarios
from utils.processors import Processor

logger = Logger.get_logger(__name__)

torch.set_num_threads(1)

from utils.processors import get_observation_for_test
from ts_inherit.utils import to_torch_as, to_numpy
from ts_inherit.networks import MyActor
from ts_inherit.rainbow import MyRainbow
from tianshou.utils.net.discrete import NoisyLinear

from rainbow.config import cfg

cfg.action_space = gym.spaces.Discrete(cfg.action_per_dim[0] * cfg.action_per_dim[1])
cfg.action_shape = cfg.action_space.n

name = 'checkpoint.pth'


def load_policy(cfgs, name):
    def noisy_linear(x, y):
        return NoisyLinear(x, y, cfgs.noisy_std)

    _model = MyActor(
        cfgs.network,
        cfgs.action_shape,
        device=cfgs.device,
        softmax=True,
        num_atoms=cfgs.num_atoms,
        dueling_param=({"linear_layer": noisy_linear}, {"linear_layer": noisy_linear}),
    )

    optim = torch.optim.Adam(_model.parameters(), lr=cfgs.lr)

    _policy = MyRainbow(
        model=_model,
        optim=optim,
        discount_factor=cfgs.gamma,
        num_atoms=cfgs.num_atoms,
        v_min=cfgs.v_min,
        v_max=cfgs.v_max,
        estimation_step=cfgs.n_step,
        target_update_freq=cfgs.target_update_freq,
    ).to(cfgs.device)

    _policy.make_action_library(cfgs)

    log_path = os.path.join(cfgs.logdir, cfgs.task, "rainbow")

    logger.info(f"Loading agent under {log_path}")
    ckpt_path = os.path.join(log_path, name)
    if os.path.exists(ckpt_path):
        if name == 'best_policy.pth':
            best_policy = torch.load(ckpt_path, map_location='cpu')
            _policy.load_state_dict(best_policy)
            logger.info("Successfully restore policy.")
        else:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            _policy.load_state_dict(checkpoint['model'])
            logger.info("Successfully restore policy.")
    else:
        logger.info(f"Fail to restore policy in {ckpt_path}!")

    return _policy


def run(worker_index):
    try:
        env = gym.make("MatrixEnv-v1", scenarios=Scenarios.INFERENCE)
        obs = env.reset()
        policy = load_policy(cfg, name)
        while True:
            obs = get_observation_for_test(cfg=cfg, obs=obs)
            result = policy(obs)
            act = policy.map_action(to_numpy(result.act))[0]
            obs, reward, done, info = env.step(act)
            infer_done = DoneReason.INFERENCE_DONE == info.get("DoneReason", "")
            if done and not infer_done:
                obs = env.reset()
            elif infer_done:
                break
    except Exception as e:
        logger.info(f"{worker_index}, error: {str(e)}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    num_workers = 1

    pool = Pool(num_workers)
    pool_result = pool.map_async(run, list(range(num_workers)))
    pool_result.wait(timeout=3000)

    logger.info("inference done.")
