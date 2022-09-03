# ******************************************************************************
# * Copyright (C) Alibaba-inc - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# *****************************************************************************

from multiprocessing import Pool

import gym
import numpy
import os 
import sys 
from geek.env.logger import Logger
from geek.env.matrix_env import DoneReason, Scenarios

logger = Logger.get_logger(__name__)
from train.policy import PPOPolicy
from train.tools import EnvPostProcsser
from train.workers import EnvWorker
from pathlib import Path
import argparse 
import torch
from train.config import CommonConfig
# os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.
torch.set_num_threads(1)
model_dir = CommonConfig.remote_path

parser = argparse.ArgumentParser()
parser.add_argument('--load_model', action="store_true", default=False)
parser.add_argument('--num_workers', type=int, default=1)
args = parser.parse_args()
high_action = CommonConfig.env_action_space.high
low_action = CommonConfig.env_action_space.low
def run(worker_index):
    try:
        reach_goal = 0
        episode = 0
        logger.info(f'worker {worker_index} starting')
        env = gym.make("MatrixEnv-v1", scenarios=Scenarios.INFERENCE, render_id=worker_index)
        obs = env.reset()
        model = PPOPolicy(2)
        env_post_processer = EnvPostProcsser()
        if args.load_model:
            model.load_model(model_dir+'/network.pth', 'cpu')
            logger.info('model has been successfully loaded')
        vec_state, env_state = env_post_processer.reset(obs)
        while True:
            action, _, _, _ = model.select_action(env_state, vec_state, True)
            action = action.data.cpu().numpy()[0]
            steer = EnvWorker.lmap(action[0],[-1.0, 1.0],[low_action[0], high_action[0]],)
            acc = EnvWorker.lmap(action[1], [-1.0, 1.0], [low_action[1], high_action[1]])
            obs, _, done, info = env.step(numpy.array([steer, acc]))
            is_runtime_error = info.get(DoneReason.Runtime_ERROR, False)
            is_infer_error = info.get(DoneReason.INFERENCE_DONE, False)
            # 出现error则放弃本帧数据
            if is_infer_error or is_runtime_error:
                break
            if done :
                obs = env.reset()
                episode += 1
                env_post_processer.reset(obs)
                if info['reached_stoparea']:
                    logger.info(f"succeed !")
                    reach_goal += 1
                    logger.info(f'{worker_index}, reach_goal_rate:{reach_goal}/{episode}')
            env_state = env_post_processer.assemble_surr_vec_obs(obs)
            vec_state = env_post_processer.assemble_ego_vec_obs(obs)
    except Exception as e:
        logger.info(f"{worker_index}, error: {str(e)}")


if __name__ == "__main__":
    num_workers = args.num_workers
    pool = Pool(args.num_workers)
    logger.info('start inference')
    pool_result = pool.map_async(run, list(range(num_workers)))
    pool_result.wait(timeout=3000)
    logger.info("inference done.")
