# ******************************************************************************
# * Copyright (C) Alibaba-inc - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# *****************************************************************************

from collections import deque
import faulthandler
import os
import time
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
from utils.util import polynomial_decay, polynomial_increase
import re
 
logger = Logger.get_logger(__name__)

class MulProPPO:
    def __init__(self, logger, args):
        self.params = PolicyParam
        self.args = args
        self.logger = logger
        self.global_sample_size = 0
        self.task_name = args.task_name
        self.load = args.load
        self.stage = args.stage
        self.best_reach_rate = -np.inf
        self.start_episode = args.start_episode
        # schedule
        self.lr_schedule = PolicyParam.learning_rate_schedule
        self.beta_schedule = PolicyParam.beta_schedule
        self.cr_schedule = PolicyParam.clip_range_schedule
        self.balance_schedule = PolicyParam.balance_schedule
        self.loss_ratio_schedule = PolicyParam.loss_ratio_schedule
        self.clip = self.cr_schedule['initial']
        self.beta = self.beta_schedule['initial']
        self.lr = self.lr_schedule['initial']
        self.balance = self.balance_schedule['initial']
        self.loss_ratio = self.loss_ratio_schedule['initial']
        self.loss_value_coeff = self.params.loss_coeff_value
        self._make_dir()
        torch.manual_seed(self.params.seed)
        np.random.seed(self.params.seed)
        if self.params.device == "cuda":
            torch.cuda.manual_seed(self.params.seed)
        self.sampler = MemorySampler(self.params, self.logger, args)
        if self.params.gaussian:
            self.model = PPOPolicy(2).to(self.params.device)
        else:
            self.model = CategoricalPPOPolicy(CommonConfig.action_num).to(self.params.device)
        self.optimizer = opt.Adam(self.model.parameters(), lr=self.lr)
        if self.params.use_value_norm:
            self.value_norm = Normalization(1, device = self.params.device)
        if args.load:
            if self.start_episode == 0:
                self.load_checkpoint()
            else:
                self.load_checkpoint(self.params.model_path+f'/checkpoint_{args.start_episode}.pth')
        self.num_episode = self.start_episode + self.params.num_episode
        if not self.load:
            self.warmup_episode = self.start_episode
        else:
            self.warmup_episode = self.start_episode + self.params.warmup_episode
        self.start_time = time.time()
        self.reward_deque = deque(maxlen=10)
        self.base_reward_deque = deque(maxlen=10)
        self.rule_reward_deque = deque(maxlen=10)
        self.collide_reward_deque = deque(maxlen=10)
        self.reach_rate_deque = deque(maxlen=10)

    def _make_dir(self):
        current_dir = os.path.abspath(".")
        self.exp_dir = current_dir + "/results/exp/"
        self.model_dir = current_dir + "/results/model/"
        try:
            os.makedirs(self.exp_dir)
            os.makedirs(self.model_dir)
        except:
            print("file is existed")
        
        self.writer = Writer(self.exp_dir, openid='oWbT458DhHKaykIOzAs_9nXUFL_M')

    def _read_history_models(self):
        all_index = []
        number = re.compile(r'\d+')
        files = os.listdir(self.params.model_path)
        for file in files:
            index = number.findall(file)
            if len(index) > 0:
                all_index.append(int(index[0]))
        all_index.sort() # from low to high sorting
        return all_index 

    def load_model(self):
        path = self.params.model_path+f'/network_{self.start_episode}.pth'
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint)
        self.logger.info(f'Successfully load pre-trained model : {path}')

    def load_checkpoint(self, path=None):

        if path is None:
            index = self._read_history_models()
            latest_index = index[-1]
            path = self.params.model_path+f'/checkpoint_{latest_index}.pth'
        checkpoint = torch.load(path)
        self.start_episode = checkpoint['episode']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.value_norm.load_state_dict(checkpoint['value_norm'])
        self.best_reach_rate = checkpoint['best_reach_rate']
        self.logger.info(f'Successfully load pre-trained model : {path}')
    
    def save_checkpoint(self, path, episode, best_reach_rate):

        checkpoint = {'episode': episode, 
                      'best_reach_rate': best_reach_rate,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'value_norm': self.value_norm.state_dict()
                      }
        torch.save(checkpoint, path)
        self.logger.info(f'model has been successfully saved : {path}')

    def cal_value_loss(self, env_state, old_values, returns):

        newvalues = self.model.get_value(env_state).flatten()
        value_pred_clipped = old_values + (
            newvalues - old_values
        ).clamp(-self.params.vf_clip_param, self.params.vf_clip_param)
        if self.params.use_value_norm:
            value_losses = 0.5*(newvalues - self.value_norm.normalize(returns)).pow(2)
            value_loss_clip = 0.5*(value_pred_clipped - self.value_norm.normalize(returns)).pow(2)
            loss_value = torch.max(value_losses, value_loss_clip).mean()
        else:
            value_losses = 0.5*(newvalues - returns).pow(2)
            value_loss_clip = 0.5*(value_pred_clipped - returns).pow(2)

        if self.params.use_clipped_value_loss:
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
        surr2 = ratio.clamp(1 - self.clip, 1 + self.clip) * advantages
        loss_surr = -torch.mean(torch.min(surr1, surr2))

        return loss_surr, loss_entropy, approx_kl

    def update(self, batch, episode, batch_size, loss_coef=0.5):
        
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
            if self.params.use_value_norm:
                deltas[i] = rewards[i] + self.params.gamma*self.value_norm.denormalize(prev_value) \
                            *masks[i] - self.value_norm.denormalize(values[i])
                advantages[i] = deltas[i] + self.params.gamma * self.params.lamda * prev_advantage * masks[i]
                returns[i] = advantages[i] + self.value_norm.denormalize(values[i]) # TD-lamda return
            else: 
                deltas[i] = rewards[i] + self.params.gamma*prev_value \
                            *masks[i] - values[i]
                advantages[i] = deltas[i] + self.params.gamma * self.params.lamda * prev_advantage * masks[i]
                returns[i] = advantages[i] + values[i]

            prev_advantage = advantages[i]
            prev_value = values[i]
        # update returns
        if self.params.use_value_norm:
            self.value_norm.update(returns)
        sur_obs = sur_obs.to(self.params.device)
        values = values.to(self.params.device)
        vec_obs = vec_obs.to(self.params.device)
        actions = actions.to(self.params.device)
        gaussian_actions = gaussian_actions.to(self.params.device)
        oldlogproba = oldlogproba.to(self.params.device)
        advantages = advantages.to(self.params.device)
        returns = returns.to(self.params.device)
        
        for i_epoch in range(self.params.num_epoch):
            rand = np.random.permutation(batch_size)
            minibatch_size = min(batch_size, self.args.minibatch_size)
            num_mini_batch = batch_size // minibatch_size
            sampler = [rand[i*minibatch_size:(i+1)*minibatch_size] for i in range(num_mini_batch)]
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
                if self.params.use_advantage_norm:
                    minibatch_advantages = (minibatch_advantages - minibatch_advantages.mean()) \
                                            / (minibatch_advantages.std() + self.params.EPS)
                minibatch_returns = returns[minibatch_ind]

                if self.params.gaussian:
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
                    + self.loss_value_coeff * loss_value
                    + self.beta * loss_entropy
                )

                if self.params.use_target_kl:
                    if approx_kl > 1.5*self.params.target_kl:
                        self.logger.info(f'Episode:{episode} Early stopping at epoch {i_epoch} due to reaching max kl.')
                        break
                # update lr 
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.lr
                self.optimizer.zero_grad()
                (loss_coef*total_loss).backward()
                if self.params.use_clip_grad:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm)
                # before training , rollout out some random episode to initiate
                if episode >= self.warmup_episode:
                    self.optimizer.step()

        # update normalization 
        self.model.update_norm(sur_obs.view(-1, sur_obs.shape[-1]), vec_obs.view(-1, vec_obs.shape[-1]))
        info = dict(kl=approx_kl, total_loss=total_loss.item(), loss_surr=loss_surr.item(),
                    loss_entropy=loss_entropy.item(), loss_value=loss_value.item())
        return info

    def log(self, memory, rewards, base_rewards, collide_rewards, rule_rewards, info, episode, success_info=None, failed_info=None):
        
        total_loss = info['total_loss']
        loss_surr = info['loss_surr']
        loss_value = info['loss_value']
        loss_entropy = info['loss_entropy']

        mean_reward = (np.sum(rewards) / memory.num_episode)
        mean_collide_reward = (np.sum(collide_rewards) / memory.num_episode)
        mean_base_reward = (np.sum(base_rewards) / memory.num_episode)
        mean_rule_reward = (np.sum(rule_rewards) / memory.num_episode)
        mean_step = len(memory) // memory.num_episode
        reach_goal_rate = memory.arrive_goal_num / memory.num_episode

        self.reward_deque.append(mean_reward)
        self.rule_reward_deque.append(mean_rule_reward)
        self.base_reward_deque.append(mean_base_reward)
        self.collide_reward_deque.append(mean_collide_reward)
        self.reach_rate_deque.append(reach_goal_rate)

        self.logger.info("Finished iteration: " + str(episode))
        self.logger.info("reach goal rate: " + str(reach_goal_rate))
        self.logger.info("reward: " + str(round(mean_reward,2)))
        self.logger.info(
            "total loss: "
            + str(round(total_loss, 3))
            + " = "
            + str(round(loss_surr,3))
            + "+"
            + str(self.loss_value_coeff)
            + "*"
            + str(round(loss_value,3))
            + "+"
            + str(round(self.beta, 5))
            + "*"
            + str(round(loss_entropy,3))
        )
        if success_info is not None:
            total_success_loss = success_info['total_loss']
            success_loss_surr = success_info['loss_surr']
            success_loss_value = success_info['loss_value']
            success_loss_entropy = success_info['loss_entropy']
            self.logger.info(
                "total success loss: "
                + str(round(total_success_loss, 3))
                + " = "
                + str(round(success_loss_surr,3))
                + "+"
                + str(self.loss_value_coeff)
                + "*"
                + str(round(success_loss_value,3))
                + "+"
                + str(round(self.beta, 5))
                + "*"
                + str(round(success_loss_entropy,3))
            )
        if failed_info is not None:
            total_fail_loss = failed_info['total_loss']
            fail_loss_surr = failed_info['loss_surr']
            fail_loss_value = failed_info['loss_value']
            fail_loss_entropy = failed_info['loss_entropy']
            
            self.logger.info(
                "total fail loss: "
                + str(round(total_fail_loss, 3))
                + " = "
                + str(round(fail_loss_surr,3))
                + "+"
                + str(self.loss_value_coeff)
                + "*"
                + str(round(fail_loss_value,3))
                + "+"
                + str(round(self.beta, 5))
                + "*"
                + str(round(fail_loss_entropy,3))
            )
        self.logger.info('Gaussian Policy:' + str(self.params.gaussian))
        self.logger.info("Step: " + str(mean_step))
        self.logger.info("total data number: " + str(self.global_sample_size))
        self.logger.info(
            "lr: " + str(round(self.lr, 5)) + ' clip: '+ str(round(self.clip, 5)) + ' entropy: '+str(round(self.beta, 5)))
        self.logger.info('balance:' + str(round(self.balance, 5)))
        if self.params.use_loss_balance:
            self.logger.info('loss_ratio' + str(round(self.loss_ratio, 5)))
        self.writer.scalar_summary("reward", np.mean(self.reward_deque), episode)
        self.writer.scalar_summary('base_reward', np.mean(self.base_reward_deque), episode)
        self.writer.scalar_summary('collide_reward', np.mean(self.collide_reward_deque), episode)
        self.writer.scalar_summary('rule_reward', np.mean(self.rule_reward_deque), episode)
        self.writer.scalar_summary("reach_goal_rate", np.mean(self.reach_rate_deque), episode)
        self.writer.scalar_summary('pi_loss', loss_surr, episode)
        self.writer.scalar_summary('value_loss', loss_value, episode)
        self.writer.scalar_summary('entropy_loss', loss_entropy, episode)
        self.writer.scalar_summary('approx_kl', info['kl'], episode)
        self.writer.scalar_summary('mean_step', mean_step, episode)

    def train(self):
        nc = notice("oWbT458DhHKaykIOzAs_9nXUFL_M")
        nc.task_complete_notice(task_name="Training", task_progree=f"training Started {self.task_name}.")

        for i_episode in range(self.start_episode, self.num_episode):

            success_memory, failed_memory = self.sampler.sample(self.model, self.balance)
            total_memory = success_memory + failed_memory
            success_batch_size = len(success_memory)
            failed_batch_size = len(failed_memory)
            if success_batch_size > 0: 
                success_batch = success_memory.sample()
            if failed_batch_size > 0:
                failed_batch = failed_memory.sample()
            total_batch = total_memory.sample()
            # schedule hyper_params
            batch_size = len(total_memory)
            self.global_sample_size += batch_size
            self.lr = polynomial_decay(self.lr_schedule["initial"], self.lr_schedule["final"], self.lr_schedule["max_decay_steps"], self.lr_schedule["power"], i_episode)
            self.beta = polynomial_decay(self.beta_schedule["initial"], self.beta_schedule["final"], self.beta_schedule["max_decay_steps"], self.beta_schedule["power"], i_episode)
            self.clip = polynomial_decay(self.cr_schedule["initial"], self.cr_schedule["final"], self.cr_schedule["max_decay_steps"], self.cr_schedule["power"], i_episode)
            self.balance = polynomial_increase(self.balance_schedule["initial"], self.balance_schedule["final"], self.balance_schedule["max_decay_steps"], self.balance_schedule["power"], i_episode)
            self.loss_ratio = polynomial_decay(self.loss_ratio_schedule["initial"], self.loss_ratio_schedule["final"], self.loss_ratio_schedule["max_decay_steps"], self.loss_ratio_schedule["power"], i_episode)
            # update policy
            success_info = None
            failed_info = None 
            if success_batch_size > 0 and failed_batch_size > 0 and self.params.use_loss_balance:
                success_info = self.update(success_batch, i_episode, success_batch_size, loss_coef=self.loss_ratio)
                failed_info = self.update(failed_batch, i_episode, failed_batch_size, loss_coef=1-self.loss_ratio)
                info = {}
                for key, value in success_info.items():
                    info[key] = success_info[key]*self.loss_ratio + failed_info[key]*(1-self.loss_ratio)
            else:
                info = self.update(total_batch, i_episode, batch_size, loss_coef=1)

            # log 
            if i_episode % self.params.log_num_episode == 0 or i_episode == (self.num_episode-1):
                self.logger.info(
                "----------------------" + str(i_episode) + "-------------------------"
                                ) 
                self.log(total_memory, total_batch.reward, total_batch.base_reward, total_batch.collide_reward, total_batch.rule_reward, info, i_episode, success_info=success_info, failed_info=failed_info)
                
            if (i_episode % self.params.eval_interval == 0 or i_episode == (self.num_episode-1)) and self.params.use_eval:
                memory = self.sampler.eval(self.model, self.params.eval_episode)
                batch = memory.sample()
                mean_reward = np.sum(batch.reward) / memory.num_episode
                mean_step = len(memory) // memory.num_episode
                reach_goal_rate = memory.arrive_goal_num / memory.num_episode
                if self.best_reach_rate < reach_goal_rate:
                    self.best_reach_rate = reach_goal_rate
                    path = remote_path + "/best_checkpoint.pth"
                    self.save_checkpoint(path, i_episode, self.best_reach_rate)
                self.logger.info(
                "--------------------" + 'Eval ' + str(i_episode) + "---------------------"
                                )
                self.logger.info("Mean Step: " + str(mean_step))
                self.logger.info("Success Rate: " + str(reach_goal_rate))
                self.logger.info('Mean Reward:' + str(mean_reward))
                self.logger.info('Best Reach_Rate:' + str(self.best_reach_rate))
                self.writer.scalar_summary('Eval Successs Rate', reach_goal_rate, i_episode)
                self.writer.scalar_summary('Eval Mean Reward', mean_reward, i_episode)
                self.writer.scalar_summary('Eval Mean Step', mean_step, i_episode)
                nc = notice("oWbT458DhHKaykIOzAs_9nXUFL_M")
                nc.task_complete_notice(task_name="Training", task_progree=f"{self.task_name} training {i_episode}")
                self.writer.show('Eval Successs Rate')
                self.writer.show('Eval Mean Reward')
                self.writer.show('Eval Mean Step')
                self.writer.show('reach_goal_rate')
                self.writer.show('reward')
                self.writer.show('base_reward')
                self.writer.show('collide_reward')
                self.writer.show('rule_reward')
                self.writer.show('approx_kl')
                self.writer.show('value_loss')
                self.writer.show('pi_loss')
                self.writer.show('entropy_loss')
                self.writer.show('mean_step')

            if i_episode % self.params.save_num_episode == 0 and i_episode > (self.warmup_episode) or i_episode == (self.num_episode-1):
                save_path = remote_path + f"/checkpoint_{i_episode}.pth"
                self.save_checkpoint(save_path, i_episode, self.best_reach_rate)

        self.sampler.close()

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='Base_PPO')
    parser.add_argument('--load', action='store_true', default=False)
    parser.add_argument('--start_episode', type=int, default=0)
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=5120)
    parser.add_argument('--minibatch_size', type=int, default=512)
    args = parser.parse_args()
    remote_path = CommonConfig.remote_path
    os.makedirs(remote_path, exist_ok=True)
    torch.set_num_threads(1)
    mpp = MulProPPO(logger, args)
    mpp.train()
