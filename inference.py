# ******************************************************************************
# * Copyright (C) Alibaba-inc - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# *****************************************************************************

from math import fabs
from multiprocessing import Pool
import gym
from geek.env.logger import Logger
from geek.env.matrix_env import DoneReason, Scenarios
from train.policy import PPOPolicy, CategoricalPPOPolicy
from train.tools import EnvPostProcsser
from train.workers import EnvWorker
import numpy as np 
import argparse 
import torch
from train.config import CommonConfig, PID2Config, PolicyParam
import re
import os 
from train.config import PIDConfig

logger = Logger.get_logger(__name__)
torch.set_num_threads(1)
model_dir = CommonConfig.remote_path
parser = argparse.ArgumentParser()
parser.add_argument('--load_model', action="store_true", default=False)
parser.add_argument('--start_episode', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--best', action='store_true', default=False)
args = parser.parse_args()
high_action = CommonConfig.env_action_space.high
low_action = CommonConfig.env_action_space.low
action_num = CommonConfig.action_num
action_repeat = PolicyParam.action_repeat

actions_map = EnvWorker._set_actions_map(action_num)

def read_history_models(model_path):
    all_index = []
    number = re.compile(r'\d+')
    files = os.listdir(model_path)
    for file in files:
        index = number.findall(file)
        if len(index) > 0:
            all_index.append(int(index[0]))
    all_index.sort() # from low to high sorting
    return all_index 

def action_transform(action, gaussian) -> np.array :
    if gaussian:
        high_action = CommonConfig.env_action_space.high
        low_action = CommonConfig.env_action_space.low
        steer = EnvWorker.lmap(action[0],[-1.0, 1.0],[low_action[0], high_action[0]],)
        acc = EnvWorker.lmap(action[1], [-1.0, 1.0], [low_action[1], high_action[1]])
        env_action = np.array([steer, acc])
    else:
        env_action = np.array(actions_map[action])
    return env_action

def run(worker_index):
    try:
        reach_goal = 0
        episode = 0
        logger.info(f'worker {worker_index} starting')
        env = gym.make("MatrixEnv-v1", scenarios=Scenarios.INFERENCE, render_id=worker_index)
        obs = env.reset()
        if PolicyParam.gaussian:
            model = PPOPolicy(2)
        else:
            model = CategoricalPPOPolicy(action_num)
        env_post_processer = EnvPostProcsser(stage=0, actions_map=actions_map, pid_params=PIDConfig, pid2_params=PID2Config)
        if args.load_model:
            if args.best:
                ckpt = torch.load(model_dir+f'/best_checkpoint.pth', 'cpu')
                start_episode = ckpt['episode']
            else:
                start_episode = args.start_episode
                if args.start_episode == 0:
                    index = read_history_models(model_dir)
                    start_episode = index[-1]
                ckpt =torch.load(model_dir+f'/checkpoint_{start_episode}.pth', 'cpu')
            model_state_dict = ckpt['model_state_dict']
            model.load_state_dict(model_state_dict)
            logger.info(f'model_{start_episode} has been successfully loaded')
        vec_state, env_state, available_actions = env_post_processer.reset(obs)
        acc_flag =False
        pid_flag2 = False
        while True:
            pid_flag = env_post_processer.judge_for_adjust_steer(obs)
            acc_flag = env_post_processer.judge_for_adjust_acc(obs, acc_flag)
            pid_flag2 = env_post_processer.judge_for_adjust_steer_2(obs, pid_flag2)
            action, _, _, _ = model.select_action(env_state, vec_state, True, available_actions)
            env_action = action_transform(action, PolicyParam.gaussian)
            if pid_flag:
                env_post_processer.pid_controller.turn_on()
                env_post_processer.pid_controller_2.turn_off()
                steer = env_post_processer.pid_control(flag=0, obs=obs)
                env_action[0] = steer
                # logger.info(f'worker: {worker_index}, steer: {steer}') 
            elif pid_flag2:
                env_post_processer.pid_controller_2.turn_on()
                env_post_processer.pid_controller.turn_off()
                steer = env_post_processer.pid_control(flag=1, obs=obs)
                env_action[0] = steer
            else:
                env_action = env_action
                env_post_processer.pid_controller.turn_off()
                env_post_processer.pid_controller_2.turn_off()
            if acc_flag:
                env_action[1] = -6
            for _ in range(action_repeat):
                obs, _, done, info = env.step(env_action)
                if done:
                    break
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
            available_actions = env_post_processer.get_available_actions(obs)  # TODO: available actions
    except Exception as e:
        logger.info(f"{worker_index}, error: {str(e)}")


if __name__ == "__main__":
    num_workers = args.num_workers
    pool = Pool(args.num_workers)
    logger.info('start inference')
    pool_result = pool.map_async(run, list(range(num_workers)))
    pool_result.wait(timeout=3000)
    logger.info("inference done.")
