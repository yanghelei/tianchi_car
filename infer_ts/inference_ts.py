# -*- coding : utf-8 -*-
# @Time     : 2022/8/26 下午6:45
# @Author   : gepeng
# @ FileName: inference_ts.py
# @ Software: Pycharm

import os.path

import gym
import gym.spaces as spaces
import numpy as np

from geek.env.logger import Logger
from geek.env.matrix_env import DoneReason, Scenarios
from train_ts.tools import EnvPostProcsser
from train_ts.main import get_args, VecFeatureNet
from algo_ts.utils.net.common import Net
from algo_ts.policy import SACPolicy

from algo_ts.utils.net.continuous import ActorProb
from algo_ts.utils.net.continuous import Critic
from algo_ts.exploration import GaussianNoise
from algo_ts.data.batch import Batch
from algo_ts.data import to_numpy
import torch

logger = Logger.get_logger(__name__)


class Inference(object):
    def __init__(self):
        self.args = get_args()

        self.data = Batch(
            obs={}, act={}, rew={}, done={}, obs_next={}, info={}, policy={}
        )

    @staticmethod
    def preprocess_obs(raw_obs):
        env_post_processer = EnvPostProcsser(args=get_args())
        feature_net = VecFeatureNet()

        env_state, _ = env_post_processer.assemble_surr_obs(raw_obs)
        vec_state = env_post_processer.assemble_ego_vec_obs(raw_obs)
        env_feature = feature_net(env_state)
        complex_feature_obs = torch.cat((env_feature.data, vec_state.reshape(env_feature.shape[0], -1)), 1)

        return complex_feature_obs

    def load_model(self, model_path):
        self.args.action_space = spaces.Box(low=np.array([-np.pi / 4, -6], dtype=np.float64),
                                            high=np.array([np.pi / 4, 2], dtype=np.float64),
                                            shape=(2,), dtype=np.float64)

        self.args.state_shape = 110
        self.args.action_shape = self.args.action_space.shape
        self.args.max_action = self.args.action_space.high

        # 设置模型
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
            action_space=self.args.action_space
        )

        # 加载模型
        policy.load_state_dict(torch.load(model_path))

        return policy

    def _get_model_path(self):
        project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_path, 'train_ts', 'log', self.args.task, 'sac', 'policy.pth')
        return model_path

    def run_inference(self):
        try:
            env = gym.make("MatrixEnv-v1", scenarios=Scenarios.INFERENCE)
            obs = env.reset()
            feature_obs = self.preprocess_obs([obs])

            policy = self.load_model(model_path=self._get_model_path())
            while True:
                self.data.obs = feature_obs
                action = to_numpy(policy(self.data, None).act).squeeze(0)
                # print(action)
                observation, reward, done, info = env.step(action)

                infer_done = DoneReason.INFERENCE_DONE == info.get("DoneReason", "")
                if done and not infer_done:
                    obs = env.reset()
                    feature_obs = self.preprocess_obs([obs])
                elif infer_done:
                    break
        except Exception as e:
            logger.info(f"error: {str(e)}")


if __name__ == "__main__":
    infer = Inference()
    infer.run_inference()

    logger.info("inference done.")
