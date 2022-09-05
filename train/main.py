# ******************************************************************************
# * Copyright (C) Alibaba-inc - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# *****************************************************************************

import os
import time
import shutil
from pathlib import Path
from cv2 import mean
import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
import sys
sys.path.append(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + '/')
from utils.norm import Normalization 
from train.config import PolicyParam, CommonConfig
from train.policy import PPOPolicy, CategoricalPPOPolicy
from train.workers import MemorySampler
from geek.env.logger import Logger
from ai_hub.notice import notice
from ai_hub import Logger as Writer
logger = Logger.get_logger(__name__)

class MulProPPO:
    def __init__(self, logger, task_name, load, start_episode):
        self.args = PolicyParam
        self.logger = logger
        self.global_sample_size = 0
        self.task_name = task_name
        self.load = load
        self.start_episode = start_episode
        self._make_dir()

        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        if self.args.device == "cuda":
            torch.cuda.manual_seed(self.args.seed)

        self.sampler = MemorySampler(self.args, self.logger)
        if self.args.gaussian:
            self.model = PPOPolicy(2).to(self.args.device)
        else:
            self.model = CategoricalPPOPolicy(CommonConfig.action_num).to(self.args.device)
        if load:
            self._load_model(self.args.model_path)
            self.logger.info('Successfully load pre-trained model ')
        self.optimizer = opt.Adam(self.model.parameters(), lr=self.args.lr)
        if self.args.use_value_norm:
            self.value_norm = Normalization(1, device = self.args.device)

        self.clip_now = self.args.clip
        self.start_time = time.time()

    def _load_model(self, model_path: str = None):
        if not model_path:
            return
        self.model.load_state_dict(torch.load(model_path, map_location=self.args.device))

    def _check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        # filter 'num_batches_tracked'
        missing_keys = [x for x in missing_keys if not x.endswith("num_batches_tracked")]
        if len(missing_keys) > 0:
            logger.info("[Warning] missing keys: {}".format(missing_keys))
            logger.info("missing keys:{}".format(len(missing_keys)))
        if len(unused_pretrained_keys) > 0:
            logger.info("[Warning] unused_pretrained_keys: {}".format(unused_pretrained_keys))
            logger.info("unused checkpoint keys:{}".format(len(unused_pretrained_keys)))
        logger.info("used keys:{}".format(len(used_pretrained_keys)))

        assert len(used_pretrained_keys) > 0, "check_key load NONE from pretrained checkpoint"
        return True

    def _make_dir(self):
        current_dir = os.path.abspath(".")
        self.exp_dir = current_dir + "/results/exp/"
        self.model_dir = current_dir + "/results/model/"
        try:
            os.makedirs(self.exp_dir)
            os.makedirs(self.model_dir)
        except:
            print("file is existed")
        
        self.writer = Writer(self.exp_dir, openid='oWbT458Ya1xKsC1d_E_RXWf0MNos')

    def cal_value_loss(self, env_state, old_values, returns):

        newvalues = self.model.get_value(env_state).flatten()
        value_pred_clipped = old_values + (
            newvalues - old_values
        ).clamp(-self.args.vf_clip_param, self.args.vf_clip_param)
        if self.args.use_value_norm:
            value_losses = 0.5*(newvalues - self.value_norm.normalize(returns)).pow(2)
            value_loss_clip = 0.5*(value_pred_clipped - self.value_norm.normalize(returns)).pow(2)
            loss_value = torch.max(value_losses, value_loss_clip).mean()
        else:
            value_losses = 0.5*(newvalues - returns).pow(2)
            value_loss_clip = 0.5*(value_pred_clipped - returns).pow(2)

        if self.args.use_clipped_value_loss:
            loss_value = torch.max(value_losses, value_loss_clip).mean()
        else:
            loss_value = torch.mean((newvalues - returns).pow(2))

        return loss_value

    def cal_pi_loss(self, oldlogproba, env_state, actions, advantages):

        minibatch_newlogproba, minibatch_entropy = self.model.eval(
            env_state, actions
        )                
        loss_entropy = -minibatch_entropy # this is -entropy
        assert oldlogproba.shape == minibatch_newlogproba.shape
        log_ratio = minibatch_newlogproba - oldlogproba
        ratio = torch.exp(log_ratio)
        approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item() # aprroximate the kl
        assert ratio.shape == advantages.shape
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_now, 1 + self.clip_now) * advantages
        loss_surr = -torch.mean(torch.min(surr1, surr2))

        return loss_surr, loss_entropy, approx_kl

    def update(self, batch, episode, batch_size):
        
        rewards = torch.from_numpy(np.array(batch.reward))
        values = torch.from_numpy(np.array(batch.value))
        masks = torch.from_numpy(np.array(batch.mask))
        actions = torch.from_numpy(np.array(batch.action))
        gaussian_actions = torch.from_numpy(np.array(batch.gaussian_action))
        sur_obs = torch.from_numpy(np.array(batch.sur_obs))
        vec_obs = torch.from_numpy(np.array(batch.vec_obs))
        oldlogproba = torch.from_numpy(np.array(batch.logproba))

        returns = torch.Tensor(batch_size)
        deltas = torch.Tensor(batch_size)
        advantages = torch.Tensor(batch_size)
        prev_advantage = torch.Tensor([0])
        prev_value = torch.Tensor([0])
        for i in reversed(range(batch_size)):
            if self.args.use_value_norm:
                deltas[i] = rewards[i] + self.args.gamma*self.value_norm.denormalize(prev_value) \
                            *masks[i] - self.value_norm.denormalize(values[i])
                advantages[i] = deltas[i] + self.args.gamma * self.args.lamda * prev_advantage * masks[i]
                returns[i] = advantages[i] + self.value_norm.denormalize(values[i]) # TD-lamda return
            else: 
                deltas[i] = rewards[i] + self.args.gamma*prev_value \
                            *masks[i] - values[i]
                advantages[i] = deltas[i] + self.args.gamma * self.args.lamda * prev_advantage * masks[i]
                returns[i] = advantages[i] + values[i]

            prev_advantage = advantages[i]
            prev_value = values[i]
        # update returns
        if self.args.use_value_norm:
            self.value_norm.update(returns)
        sur_obs = sur_obs.to(self.args.device)
        values = values.to(self.args.device)
        vec_obs = vec_obs.to(self.args.device)
        actions = actions.to(self.args.device)
        gaussian_actions = gaussian_actions.to(self.args.device)
        oldlogproba = oldlogproba.to(self.args.device)
        advantages = advantages.to(self.args.device)
        returns = returns.to(self.args.device)

        rand = np.random.permutation(batch_size)
        num_mini_batch = batch_size // self.args.minibatch_size
        sampler = [rand[i*self.args.minibatch_size:(i+1)*self.args.minibatch_size] for i in range(num_mini_batch)]
        
        for i_epoch in range(self.args.num_epoch):
            rand = np.random.permutation(batch_size)
            num_mini_batch = batch_size // self.args.minibatch_size
            sampler = [rand[i*self.args.minibatch_size:(i+1)*self.args.minibatch_size] for i in range(num_mini_batch)]
            for minibatch_ind in sampler:
                minibatch_sur_obs = sur_obs[minibatch_ind]
                minibatch_ego_obs = vec_obs[minibatch_ind]
                minibatch_env_state = self.model.get_env_feature(minibatch_sur_obs, 
                                                                    minibatch_ego_obs)
                minibatch_actions = actions[minibatch_ind]
                minibatch_values = values[minibatch_ind]
                minibatch_oldlogproba = oldlogproba[minibatch_ind]
                minibatch_gussain_actions = gaussian_actions[minibatch_ind]
                minibatch_advantages = advantages[minibatch_ind]
                # apply the advantage norm in the minibatch not the full_batch
                if self.args.use_advantage_norm:
                    minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) \
                                            / (minibatch_advantages.std() + self.args.EPS)
                minibatch_returns = returns[minibatch_ind]

                if self.args.gaussian:
                    loss_surr, loss_entropy, approx_kl = self.cal_pi_loss(
                                                            minibatch_oldlogproba, 
                                                            minibatch_env_state, 
                                                            minibatch_gussain_actions, 
                                                            minibatch_advantages)
                else:
                    loss_surr, loss_entropy, approx_kl = self.cal_pi_loss(
                                                            minibatch_oldlogproba, 
                                                            minibatch_env_state, 
                                                            minibatch_actions, 
                                                            minibatch_advantages)

                loss_value = self.cal_value_loss(minibatch_env_state, 
                                                 minibatch_values, 
                                                minibatch_returns)

                total_loss = (
                    loss_surr
                    + self.args.loss_coeff_value * loss_value
                    + self.args.loss_coeff_entropy * loss_entropy
                )

                if self.args.use_target_kl:
                    if approx_kl > 1.5*self.args.target_kl:
                        self.logger.info(f'Episode:{episode} Early stopping at epoch {i_epoch} due to reaching max kl.')
                self.optimizer.zero_grad()
                total_loss.backward()
                if self.args.use_clip_grad:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                # before training , rollout out some random episode to initiate
                if episode >= self.args.random_episode:
                    self.optimizer.step()

        # update normalization 
        self.model.update_norm(sur_obs.view(-1, sur_obs.shape[-1]), vec_obs.view(-1, vec_obs.shape[-1]))
        info = dict(kl=approx_kl, total_loss=total_loss.item(), loss_surr=loss_surr.item(),
                    loss_entropy=loss_entropy.item(), loss_value=loss_value.item())
        return info

    def schedule(self, i_episode):
        # clip linearly decreanse
        if self.args.schedule_clip == "linear":
            ep_ratio = 1 - ((i_episode) / self.args.num_episode)
            self.clip_now = self.args.clip * ep_ratio
        
        if self.args.schedule_clip == "fix":
            ep_ratio = 1 
            self.clip_now = self.args.clip * ep_ratio

        # lr linearly decrease
        if self.args.schedule_adam == "linear":
            ep_ratio = 1 - ((i_episode) / self.args.num_episode)
            lr_now = self.args.lr * ep_ratio
            for g in self.optimizer.param_groups:
                g["lr"] = lr_now
            iteration_reduce = self.args.lr * (1 - ep_ratio) # reduce 
        # 分段式调整 
        if self.args.schedule_adam == "layer":
            for item in self.args.lr_schedule:
                if self.global_sample_size >= item[0]:
                    lr_now = item[1]
            for g in self.optimizer.param_groups:
                g["lr"] = lr_now
            iteration_reduce = 0.0
        # 分段式+线性衰减
        if self.args.schedule_adam == "layer_linear":
            for idx, item in enumerate(self.args.lr_schedule):
                if self.global_sample_size >= item[0]:
                    lr_max = item[1]
                    data_num_min = item[0]
                    lr_min = self.args.lr_schedule[idx + 1][1]
                    data_num_max = self.args.lr_schedule[idx + 1][0]
            num_iteration = int((data_num_max - data_num_min) / self.args.batch_size)
            iteration_reduce = float((lr_max - lr_min) / num_iteration)
            self.args.lr = self.args.lr - iteration_reduce
            lr_now = self.args.lr
            for g in self.optimizer.param_groups:
                g["lr"] = lr_now

        if self.args.schedule_adam == "fix":
            lr_now = self.args.lr
            iteration_reduce = 0.0

        return lr_now, iteration_reduce

    def log(self, memory, rewards, info, episode):
        
        total_loss = info['total_loss']
        loss_surr = info['loss_surr']
        loss_value = info['loss_value']
        loss_entropy = info['loss_entropy']
        lr_now = info['lr_now']
        lr_iteration_reduce = info['lr_iteration_reduce']
        mean_reward = (np.sum(rewards) / memory.num_episode)
        mean_step = len(memory) // memory.num_episode
        reach_goal_rate = memory.arrive_goal_num / memory.num_episode

        self.logger.info("Finished iteration: " + str(episode))
        self.logger.info("reach goal rate: " + str(reach_goal_rate))
        self.logger.info("reward: " + str(round(mean_reward,2)))
        self.logger.info(
            "total loss: "
            + str(round(total_loss, 3))
            + " = "
            + str(round(loss_surr,3))
            + "+"
            + str(self.args.loss_coeff_value)
            + "*"
            + str(round(loss_value,3))
            + "+"
            + str(self.args.loss_coeff_entropy)
            + "*"
            + str(round(loss_entropy,3))
        )
        self.logger.info("Step: " + str(mean_step))
        self.logger.info("total data number: " + str(self.global_sample_size))
        self.logger.info(
            "lr now: " + str(lr_now) + "  lr reduce per iteration: " + str(lr_iteration_reduce)
        )
        self.writer.scalar_summary("reward", mean_reward, episode)
        self.writer.scalar_summary("reach_goal_rate", reach_goal_rate, episode)
        self.writer.scalar_summary('pi_loss', loss_surr, episode)
        self.writer.scalar_summary('value_loss', loss_value, episode)
        self.writer.scalar_summary('entropy_loss', loss_entropy, episode)
        self.writer.scalar_summary('approx_kl', info['kl'], episode)
        self.writer.scalar_summary('mean_step', mean_step, episode)

    def train(self):
        nc = notice("oWbT458Ya1xKsC1d_E_RXWf0MNos")
        nc.task_complete_notice(task_name="Training", task_progree=f"training Started {self.task_name}.")
        best_reward = -np.inf

        for i_episode in range(self.start_episode, self.start_episode+self.args.num_episode):

            memory = self.sampler.sample(self.model)
            batch = memory.sample()
            batch_size = len(memory)
            self.global_sample_size += batch_size
            # update policy
            info = self.update(batch, i_episode, batch_size)
            # schedule lr and clip
            lr_now, lr_iteration_reduce = self.schedule(i_episode)
            info['lr_now'] = lr_now
            info['lr_iteration_reduce'] = lr_iteration_reduce
            # log 
            if i_episode % self.args.log_num_episode == 0 or i_episode == (self.args.num_episode-1):
                self.logger.info(
                "----------------------" + str(i_episode) + "-------------------------"
                                ) 
                self.log(memory, batch.reward, info, i_episode)
                
            if i_episode % self.args.eval_interval == 0 \
                or i_episode == (self.args.num_episode-1) \
                and self.args.use_eval:
                memory = self.sampler.eval(self.model, self.args.eval_episode)
                batch = memory.sample()
                mean_reward = np.sum(batch.reward) / memory.num_episode
                mean_step = len(memory) // memory.num_episode
                if best_reward < mean_reward:
                    best_reward = mean_reward
                    torch.save(
                    self.model.state_dict(), remote_path + "/best_network.pth"
                            )
                self.logger.info(
                "--------------------" + 'Eval ' + str(i_episode) + "---------------------"
                                )
                reach_goal_rate = memory.arrive_goal_num / memory.num_episode
                self.logger.info("Mean Step: " + str(mean_step))
                self.logger.info("Success Rate: " + str(reach_goal_rate))
                self.logger.info('Mean Reward:' + str(mean_reward))
                self.logger.info('Best Reward:' + str(best_reward))
                self.writer.scalar_summary('Eval Successs Rate', reach_goal_rate, i_episode)
                self.writer.scalar_summary('Eval Mean Reward', mean_reward, i_episode)
                self.writer.scalar_summary('Eval Mean Step', mean_step, i_episode)
                nc = notice("oWbT458Ya1xKsC1d_E_RXWf0MNos")
                nc.task_complete_notice(task_name="Training", task_progree=f"{self.task_name} training {i_episode}")
                self.writer.show('Eval Successs Rate')
                self.writer.show('Eval Mean Reward')
                self.writer.show('Eval Mean Step')
                self.writer.show('reach_goal_rate')
                self.writer.show('reward')
                self.writer.show('approx_kl')
                self.writer.show('value_loss')
                self.writer.show('pi_loss')
                self.writer.show('entropy_loss')
                self.writer.show('mean_step')

            if i_episode % self.args.save_num_episode == 0 \
                and i_episode > self.args.random_episode \
                or i_episode == (self.args.num_episode-1):
                torch.save(
                    self.model.state_dict(), self.model_dir + f"network_{i_episode}.pth"
                )
                # 存储到云端
                torch.save(
                    self.model.state_dict(), remote_path + f"/network_{i_episode}.pth"
                )
                self.logger.info(f'model has been successfully saved : {remote_path}')

        self.sampler.close()

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='Base_PPO')
    parser.add_argument('--load', action='store_true', default=False)
    parser.add_argument('--start_episode', type=int, default=0)
    args = parser.parse_args()
    remote_path = CommonConfig.remote_path
    os.makedirs(remote_path, exist_ok=True)
    torch.set_num_threads(1)
    mpp = MulProPPO(logger, args.task_name, args.load, args.start_episode)
    mpp.train()
