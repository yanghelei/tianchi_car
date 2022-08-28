# ******************************************************************************
# * Copyright (C) Alibaba-inc - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# *****************************************************************************

from multiprocessing import Pool

import gym
import numpy
import os 
import sys 
sys.path.append(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + '/')
from geek.env.logger import Logger
from geek.env.matrix_env import DoneReason, Scenarios

logger = Logger.get_logger(__name__)
from train.policy import PPOPolicy
from train.tools import EnvPostProcsser
from train.workers import EnvWorker
from pathlib import Path
import argparse 
model_dir = str(Path(os.path.dirname(__file__)) / 'results' / 'model')

parser = argparse.ArgumentParser()
parser.add_argument('load_model', action="store_true", default=False)
parser.add_argument('num_workers', type=int, default=1)
args = parser.parse_args()

def run(worker_index):
    try:
        env = gym.make("MatrixEnv-v1", scenarios=Scenarios.INFERENCE)
        model = PPOPolicy(2)
        env_post_processer = EnvPostProcsser()
        if args.load_model:
            model.load_model(model_dir+'/network.pth', 'cpu')
            env_post_processer.surr_vec_normalize.load_model(model_dir+"/sur_norm.pth", 'cpu')
            env_post_processer.ego_vec_normalize.load_model(model_dir+"/ego_norm.pth", 'cpu')
        obs = env.reset()
        env_post_processer.reset(obs)
        while True:
            env_state = env_post_processer.assemble_surr_obs(obs, env)
            vec_state = env_post_processer.assemble_ego_vec_obs(obs)
            action, _, _, _ = model.select_action(env_state, vec_state, True)
            action = action.data.cpu().numpy()[0]
            steer = EnvWorker.lmap(action[0],[-1.0, 1.0],[-0.3925, 0.3925],)
            acc = EnvWorker.lmap(action[1], [-1.0, 1.0], [-6.0, 2.0])
            obs, _, done, info = env.step(numpy.array([steer, acc]))
            infer_done = DoneReason.INFERENCE_DONE == info.get("DoneReason", "")
            if done and not infer_done:
                logger.info(f"env rest")
                print(info)
            elif infer_done:
                break
    except Exception as e:
        logger.info(f"{worker_index}, error: {str(e)}")


if __name__ == "__main__":
    num_workers = args.num_workers
    pool = Pool(args.num_workers)
    pool_result = pool.map_async(run, list(range(num_workers)))
    pool_result.wait(timeout=3000)
    logger.info("inference done.")
