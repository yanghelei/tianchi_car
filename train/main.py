# ******************************************************************************
# * Copyright (C) Alibaba-inc - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# *****************************************************************************

import os
import time
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
import sys
sys.path.append(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + '/')
from utils.norm import Normalization 
from tensorboardX import SummaryWriter
from train.config import PolicyParam, CommonConfig
from train.policy import PPOPolicy
from train.workers import MemorySampler
from geek.env.logger import Logger
from geek.env.matrix_env import DoneReason
# from ai_hub.notice import notice
# from ai_hub import Logger as Writer
logger = Logger.get_logger(__name__)

class MulProPPO:
    def __init__(self, logger) -> None:
        self.args = PolicyParam
        self.logger = logger
        self.global_sample_size = 0
        self._make_dir()

        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        if self.args.device == "cuda":
            torch.cuda.manual_seed(self.args.seed)

        self.sampler = MemorySampler(self.args, self.logger)
        self.model = PPOPolicy(2).to(self.args.device)
        self.optimizer = opt.Adam(self.model.parameters(), lr=self.args.lr)
        if self.args.use_value_norm:
            self.value_norm = Normalization(1, device = self.args.device)

        self.clip_now = self.args.clip
        self.start_episode = 0
        self._load_model(self.args.model_path)
        self.start_time = time.time()

    def _load_model(self, model_path: str = None):
        if not model_path:
            return
        pretrained_dict = torch.load(
            model_path, map_location=lambda storage, loc: storage.cuda(self.args.device)
        )
        if self._check_keys(self.model, pretrained_dict):
            self.model.load_state_dict(pretrained_dict, strict=False)

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
        
        # self.writer = Writer(self.exp_dir, openid='oWbT458Ya1xKsC1d_E_RXWf0MNos')

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

        if self.args.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.args.EPS)

        sur_obs = sur_obs.to(self.args.device)
        values = values.to(self.args.device)
        vec_obs = vec_obs.to(self.args.device)
        actions = actions.to(self.args.device)
        gaussian_actions = gaussian_actions.to(self.args.device)
        oldlogproba = oldlogproba.to(self.args.device)
        advantages = advantages.to(self.args.device)
        returns = returns.to(self.args.device)
        if self.args.use_value_norm:
            self.value_norm.update(returns)
        for i_epoch in range(int(self.args.num_epoch * batch_size / self.args.minibatch_size)):
            minibatch_ind = np.random.choice(
                batch_size, self.args.minibatch_size, replace=False
            )
            minibatch_sur_obs = sur_obs[minibatch_ind]
            minibatch_ego_obs = vec_obs[minibatch_ind]
            minibatch_env_state = self.model.get_env_feature(minibatch_sur_obs, 
                                                                minibatch_ego_obs)
            minibatch_actions = actions[minibatch_ind]
            minibatch_values = values[minibatch_ind]
            minibatch_oldlogproba = oldlogproba[minibatch_ind]
            minibatch_gussain_actions = gaussian_actions[minibatch_ind]
            minibatch_newlogproba = self.model.eval(
                minibatch_env_state, minibatch_gussain_actions
            )                
            loss_entropy = minibatch_newlogproba.mean() # this is -entropy
            minibatch_advantages = advantages[minibatch_ind]
            minibatch_returns = returns[minibatch_ind]
            minibatch_newvalues = self.model.get_value(minibatch_env_state).flatten()
            assert minibatch_oldlogproba.shape == minibatch_newlogproba.shape
            log_ratio = minibatch_newlogproba - minibatch_oldlogproba
            ratio = torch.exp(log_ratio)
            approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).item() # aprroximate the kl
            assert ratio.shape == minibatch_advantages.shape
            surr1 = ratio * minibatch_advantages
            surr2 = ratio.clamp(1 - self.clip_now, 1 + self.clip_now) * minibatch_advantages
            loss_surr = -torch.mean(torch.min(surr1, surr2))

            value_pred_clipped = minibatch_values + (
                minibatch_newvalues - minibatch_values
            ).clamp(-self.args.vf_clip_param, self.args.vf_clip_param)

            if self.args.use_value_norm:
                value_losses = 0.5*(minibatch_newvalues - self.value_norm.normalize(minibatch_returns)).pow(2)
                value_loss_clip = 0.5*(value_pred_clipped - self.value_norm.normalize(minibatch_returns)).pow(2)
                loss_value = torch.max(value_losses, value_loss_clip).mean()
            else:
                value_losses = 0.5*(minibatch_newvalues - minibatch_returns).pow(2)
                value_loss_clip = 0.5*(value_pred_clipped - minibatch_returns).pow(2)

            if self.args.use_clipped_value_loss:
                loss_value = torch.max(value_losses, value_loss_clip).mean()
            else:
                loss_value = torch.mean((minibatch_newvalues - minibatch_returns).pow(2))

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
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            # before training , rollout out some random episode to initiate
            if episode >= self.args.random_episode:
                self.optimizer.step()

        # update normalization 
        self.model.update_norm(sur_obs.view(-1, sur_obs.shape[-1]), vec_obs.view(-1, vec_obs.shape[-1]))
        return total_loss, loss_surr, loss_value, loss_entropy, rewards

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

    def train(self):
        # nc = notice("oWbT458Ya1xKsC1d_E_RXWf0MNos")
        # nc.task_complete_notice(task_name="Training", task_progree="training Started.")

        for i_episode in range(self.args.num_episode):

            memory = self.sampler.sample(self.model)
            batch = memory.sample()
            batch_size = len(memory)
            self.global_sample_size += batch_size
            # update policy
            total_loss, loss_surr, loss_value, loss_entropy, rewards = self.update(batch, i_episode, batch_size)
            # schedule lr and clip
            lr_now, lr_iteration_reduce = self.schedule(i_episode)
            if i_episode % self.args.log_num_episode == 0:
                self.logger.info(
                "----------------------" + str(i_episode) + "-------------------------"
                                ) 
                mean_reward = (torch.sum(rewards) / memory.num_episode).data
                mean_step = len(memory) // memory.num_episode
                reach_goal_rate = memory.arrive_goal_num / memory.num_episode

                reward = mean_reward.cpu().data.item()
                total_loss = total_loss.cpu().data.item()
                loss_surr = loss_surr.cpu().data.item()
                loss_value = loss_value.cpu().data.item()
                loss_entropy = loss_entropy.cpu().data.item()
                self.logger.info("Finished iteration: " + str(i_episode))
                self.logger.info("reach goal rate: " + str(reach_goal_rate))
                self.logger.info("reward: " + str(reward))
                self.logger.info(
                    "total loss: "
                    + str(total_loss)
                    + " = "
                    + str(loss_surr)
                    + "+"
                    + str(self.args.loss_coeff_value)
                    + "*"
                    + str(loss_value)
                    + "+"
                    + str(self.args.loss_coeff_entropy)
                    + "*"
                    + str(loss_entropy)
                )
                self.logger.info("Step: " + str(mean_step))
                self.logger.info("total data number: " + str(self.global_sample_size))
                self.logger.info(
                    "lr now: " + str(lr_now) + "  lr reduce per iteration: " + str(lr_iteration_reduce)
                )
                # self.writer.scalar_summary("reward", reward, i_episode)
                # self.writer.scalar_summary("total_loss", total_loss, i_episode)
                # self.writer.scalar_summary("reach_goal_rate", reach_goal_rate, i_episode)
                # self.writer.scalar_summary('pi_loss', loss_surr, i_episode)
                # self.writer.scalar_summary('value_loss', loss_value, i_episode)
                # self.writer.scalar_summary('entropy_loss', loss_entropy, i_episode)
                # self.writer.show('reach_goal_rate')
            if i_episode % self.args.save_num_episode == 0 and i_episode > self.args.random_episode:
                torch.save(
                    self.model.state_dict(), self.model_dir + "network.pth"
                )
                # 存储到云端
                torch.save(
                    self.model.state_dict(), remote_path + "/network.pth"
                )
                self.logger.info(f'model has been successfully saved : {remote_path}')
            
            # 超时则提前结束训练   
            if (time.time() - self.start_time) > 9*3600:
                break

        torch.save(
            self.model.state_dict(), self.model_dir + "network.pth"
        )
        # 存储到云端
        torch.save(
                    self.model.state_dict(), remote_path + "/network.pth"
                )
        self.sampler.close()

if __name__ == "__main__":
    remote_path = CommonConfig.remote_path
    os.makedirs(remote_path, exist_ok=True)
    # os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.
    torch.set_num_threads(1)
    mpp = MulProPPO(logger=logger)
    mpp.train()
