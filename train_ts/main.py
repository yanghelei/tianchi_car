# -*- coding : utf-8 -*-
# @Time     : 2022/8/25 下午1:14
# @Author   : gepeng
# @ FileName: main.py.py
# @ Software: Pycharm


import os
import numpy as np
import torch
import gym
import gym.spaces as spaces
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import sys
from algo_ts.data import Batch
import pprint

project_path = os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(project_path)

from algo_ts.data import Collector, VectorReplayBuffer
from algo_ts.env import DummyVectorEnv
from algo_ts.policy import SACPolicy
from algo_ts.trainer import offpolicy_trainer
from algo_ts.utils import TensorboardLogger
from algo_ts.utils.net.common import Net
from algo_ts.exploration import GaussianNoise

from algo_ts.utils.net.continuous import ActorProb
from algo_ts.utils.net.continuous import Critic

from geek.env.matrix_env import Scenarios
from train_ts.tools import EnvPostProcsser, initialize_weights


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="MatrixEnv-v1")
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--reward-threshold', type=float, default=np.inf)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--actor-lr', type=float, default=3e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    parser.add_argument('--noise_std', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--auto_alpha', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--start-timesteps', type=int, default=10000)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--step-per-epoch', type=int, default=12000)
    parser.add_argument('--step-per-collect', type=int, default=200)
    parser.add_argument('--update-per-step', type=float, default=0.2)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256, 128])
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument('--rew-norm', type=bool, default=False)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # envpostprocesser args
    parser.add_argument('--target-speed', type=int, default=30)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--history-length', type=int, default=5)
    parser.add_argument('--img-width', type=int, default=224)
    parser.add_argument('--img-length', type=int, default=224)
    parser.add_argument('--ego-vec-length', type=int, default=8)
    parser.add_argument('--surr-vec-length', type=int, default=7)
    parser.add_argument('--surr-agent-number', type=int, default=10)
    parser.add_argument('--obs-type', type=str, default='vec')
    return parser.parse_args()


class CnnFeatureNet(nn.Module):
    def __init__(self):
        super(CnnFeatureNet, self).__init__()
        self.act_cnn1 = nn.Conv2d(15, 16, 3)
        self.act_cv1_pool = nn.MaxPool2d(3, stride=2)
        self.act_cnn2 = nn.Conv2d(16, 16, 3)
        self.act_cv2_pool = nn.MaxPool2d(3, stride=2)
        self.act_cnn3 = nn.Conv2d(16, 16, 3)
        self.act_cv3_pool = nn.MaxPool2d(3, stride=2)
        self.act_cnn4 = nn.Conv2d(16, 16, 3)
        self.act_cv4_pool = nn.MaxPool2d(3, stride=2)
        self.act_cnn5 = nn.Conv2d(16, 16, 3)
        self.out_size = 1296

    def forward(self, img_state):
        mt_tmp = F.relu(self.act_cnn1(img_state))
        mt_tmp = self.act_cv1_pool(mt_tmp)
        mt_tmp = F.relu(self.act_cnn2(mt_tmp))
        mt_tmp = self.act_cv2_pool(mt_tmp)
        mt_tmp = F.relu(self.act_cnn3(mt_tmp))
        mt_tmp = self.act_cv3_pool(mt_tmp)
        mt_tmp = F.relu(self.act_cnn4(mt_tmp))
        mt_tmp = self.act_cv4_pool(mt_tmp)
        mt_tmp = F.relu(self.act_cnn5(mt_tmp))
        mt_tmp = torch.flatten(mt_tmp, 1)
        return mt_tmp


class VecFeatureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.act_cnn1 = nn.Conv2d(1, 1, (5, 1))
        self.out_size = 70

    def forward(self, vec_state):
        vec_state = F.relu(self.act_cnn1(vec_state))
        vec_state = torch.flatten(vec_state, 1)
        return vec_state


class SACPolicyTrainer:
    def __init__(self):
        super(SACPolicyTrainer, self).__init__()
        self.args = get_args()
        self.env_post_processer = EnvPostProcsser(self.args)

        assert self.args.training_num == self.args.test_num, '尚未实现测试环境数量变化定制'

        self.feature_net = (
            VecFeatureNet() if self.args.obs_type == "vec" else CnnFeatureNet()
        )

        initialize_weights(self.feature_net, 'orthogonal')

    def preprocess_fn(self, **kwargs):
        # if obs && env_id exist -> reset
        # if obs_next/rew/done/info/env_id exist -> normal step
        if 'obs' in kwargs:
            raw_obs = kwargs['obs']
        else:
            raw_obs = kwargs['obs_next']

        assert self.args.obs_type == 'cnn', '尚未实现cnn特征提取'
        assert self.args.obs_type == 'vec', '应该传入env参数'
        env_state, multi_npc_info_dict = self.env_post_processer.assemble_surr_obs(raw_obs)
        vec_state = self.env_post_processer.assemble_ego_vec_obs(raw_obs)

        env_feature = self.feature_net(env_state)
        complex_feature_obs = torch.cat((env_feature.data, vec_state.reshape(env_feature.shape[0], -1)), 1)
        if 'obs' in kwargs:
            return Batch(obs=complex_feature_obs)
        else:
            reward = self.env_post_processer.assemble_reward(raw_obs, kwargs['info'], multi_npc_info_dict)
            return Batch(obs_next=complex_feature_obs, rew=reward)

    def train(self):
        env = gym.make(self.args.task, scenarios=Scenarios.TRAINING)

        # obs = env.reset()
        # print(obs['player']['property'])

        self.args.action_space = spaces.Box(low=np.array([-np.pi / 4, -6], dtype=np.float64),
                                            high=np.array([np.pi / 4, 2], dtype=np.float64),
                                            shape=(2,), dtype=np.float64)

        self.args.state_shape = 110
        self.args.action_shape = self.args.action_space.shape
        self.args.max_action = self.args.action_space.high

        train_envs = DummyVectorEnv(
            [lambda: gym.make(self.args.task, scenarios=Scenarios.TRAINING) for _ in range(self.args.training_num)]
        )

        test_envs = DummyVectorEnv(
            [lambda: gym.make(self.args.task, scenarios=Scenarios.TRAINING) for _ in range(self.args.test_num)]
        )

        # seed
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        # todo: check seed!
        train_envs.seed(self.args.seed)
        test_envs.seed(self.args.seed)

        # model
        net = Net(self.args.state_shape, hidden_sizes=self.args.hidden_sizes, device=self.args.device)
        actor = ActorProb(
            net,
            self.args.action_shape,
            max_action=self.args.max_action,
            device=self.args.device,
            unbounded=True
        ).to(self.args.device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=self.args.actor_lr)
        net_c1 = Net(
            self.args.state_shape,
            self.args.action_shape,
            hidden_sizes=self.args.hidden_sizes,
            concat=True,
            device=self.args.device
        )
        critic1 = Critic(net_c1, device=self.args.device).to(self.args.device)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=self.args.critic_lr)
        net_c2 = Net(
            self.args.state_shape,
            self.args.action_shape,
            hidden_sizes=self.args.hidden_sizes,
            concat=True,
            device=self.args.device
        )
        critic2 = Critic(net_c2, device=self.args.device).to(self.args.device)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=self.args.critic_lr)

        if self.args.auto_alpha:
            # todo: check target_entropy！
            target_entropy = -np.prod(self.args.action_space.shape)
            log_alpha = torch.zeros(1, requires_grad=True, device=self.args.device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=self.args.alpha_lr)
            self.args.alpha = (target_entropy, log_alpha, alpha_optim)

        policy = SACPolicy(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            tau=self.args.tau,
            gamma=self.args.gamma,
            alpha=self.args.alpha,
            reward_normalization=self.args.rew_norm,
            exploration_noise=GaussianNoise(0, self.args.noise_std),
            action_space=env.action_space
        )

        # 权重初始化
        initialize_weights(policy, "orthogonal")

        # load a previous policy
        if self.args.resume_path:
            policy.load_state_dict(torch.load(self.args.resume_path, map_location=self.args.device))
            print("Loaded agent from: ", self.args.resume_path)

        # collector
        train_collector = Collector(
            policy,
            train_envs,
            VectorReplayBuffer(self.args.buffer_size, len(train_envs)),
            exploration_noise=True,
            preprocess_fn=self.preprocess_fn,
        )
        test_collector = Collector(
            policy,
            test_envs,
            preprocess_fn=self.preprocess_fn,
        )
        # todo: env can't random sample action!!
        # train_collector.collect(n_step=self.args.buffer_size)

        # log
        log_path = os.path.join(self.args.logdir, self.args.task, 'sac')
        writer = SummaryWriter(log_path)
        logger = TensorboardLogger(writer)

        def save_best_fn(policy):
            torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

        def stop_fn(mean_rewards):
            return mean_rewards >= self.args.reward_threshold

        # trainer
        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            self.args.epoch,
            self.args.step_per_epoch,
            self.args.step_per_collect,
            self.args.test_num,
            self.args.batch_size,
            update_per_step=self.args.update_per_step,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger
        )

        assert stop_fn(result['best_reward'])

        if __name__ == '__main__':
            pprint.pprint(result)
            # Let's watch its performance!
            policy.eval()
            test_envs.seed(self.args.seed)
            test_collector.reset()
            result = test_collector.collect(n_episode=self.args.test_num, render=self.args.render)
            rews, lens = result["rews"], result["lens"]
            print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == '__main__':
    s = SACPolicyTrainer()
    s.train()

