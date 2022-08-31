# ******************************************************************************
# * Copyright (C) Alibaba-inc - All Rights Reserved
# * Unauthorized copying of this file, via any medium is strictly prohibited
# *****************************************************************************

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from train.config import PolicyParam
from train.tools import initialize_weights
import numpy as np 
from torch.distributions.normal import Normal
from utils.norm import Normalization

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

class TimeVecFeatureNet(nn.Module):
    def __init__(self, input_shape, num, hidden_size):
        """
        input_shape is the vector length; num is the number of sequence;  
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_1 = nn.Linear(input_shape, hidden_size)
        self.layer_2 = nn.Linear(num, 1)
        
    def forward(self, input):
        """
        input shape : [batch_size, vector_num, vector_length]
        """
        batch_size, _, _ = input.shape
        layer_1_out = F.relu(self.layer_1(input)).permute(0,2,1).contiguous()
        layer_2_out = F.relu(self.layer_2(layer_1_out))
        layer_2_out = layer_2_out.view(batch_size, self.hidden_size)
        return layer_2_out

class PPOPolicy(nn.Module):
    def __init__(self, num_outputs):
        nn.Module.__init__(self)
        self.policy_param = PolicyParam
        self.sur_feature_net = TimeVecFeatureNet(70, 5, 128)
        self.ego_feature_net = TimeVecFeatureNet(8, 5, 128)
        self.actor_net = nn.Sequential(
            OrderedDict(
                [
                    ("actor_1", nn.Linear(256, 256)),
                    ("actor_relu_1", nn.ReLU()),
                    ("actor_2", nn.Linear(256, 256)),
                    ("actor_relu_2", nn.ReLU()),
                    ("actor_mu", nn.Linear(256, num_outputs)),
                ]
            )
        )

        self.critic_net = nn.Sequential(
            OrderedDict(
                [
                    ("critic_1", nn.Linear(256, 256)),
                    ("critic_relu_1", nn.ReLU()),
                    ("critic_2", nn.Linear(256, 256)),
                    ("critic_relu2", nn.ReLU()),
                    ("critic_output", nn.Linear(256, 1)),
                ]
            )
        )

        # std is independent of the states (empirically better)
        self.log_std = torch.nn.Parameter(torch.zeros(num_outputs), requires_grad=True)
        self.sur_obs_norm = Normalization(70)
        self.ego_obs_norm = Normalization(8)
        initialize_weights(self, "orthogonal", scale = np.sqrt(2))

    def get_env_feature(self, sur_obs, ego_obs):
        
        ego_obs = self.ego_obs_norm.normalize(ego_obs)
        sur_obs = self.sur_obs_norm.normalize(sur_obs)
        ego_feature = self.ego_feature_net(ego_obs)
        sur_feature = self.sur_feature_net(sur_obs)
        env_feature = torch.concat([ego_feature, sur_feature], dim=1)

        return env_feature

    def get_value(self, env_feature):

        values = self.critic_net(env_feature)

        return values

    def update_norm(self, sur_obs, ego_obs):
        
        self.sur_obs_norm.update(sur_obs)
        self.ego_obs_norm.update(ego_obs)
    
    def select_action(self, sur_obs, ego_obs, deterministic=False):
        # 归一化
        sur_obs = self.sur_obs_norm.normalize(sur_obs)
        ego_obs = self.ego_obs_norm.normalize(ego_obs)
        env_feature = self.get_env_feature(sur_obs, ego_obs)
        action_mean = self.actor_net(env_feature)
        value = self.critic_net(env_feature)
        action_std = torch.exp(self.log_std)
        action_distribution = Normal(action_mean, action_std)
        # 采样
        if deterministic:
            gaussian_action = action_distribution.sample()
        else:
            gaussian_action = action_mean
        logproba = self._cal_tanhlogproba(action_distribution, gaussian_action)
        # 裁剪到[-1, 1]
        tanh_action = torch.tanh(gaussian_action)
        return tanh_action, gaussian_action, logproba, value

    @staticmethod
    def _cal_tanhlogproba(pi_distribution, pi_action):
        # 计算经过tanh之后的分布的logporba (参照spinningup的SAC实现)
        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        return logp_pi

    def eval(self, env_feature, actions, gussian=True):
        
        # 先转化成gaussian_action (invese tanh)
        if not gussian:
            from utils.util import TanhBijector
            actions = TanhBijector.inverse(actions) 

        action_mean = self.actor_net(env_feature)
        action_std = torch.exp(self.log_std)
        action_distribution = Normal(action_mean, action_std)
        logp_pi = self._cal_tanhlogproba(action_distribution, actions)

        return logp_pi

    def load_model(self, model_path, device):
        self.load_state_dict(torch.load(model_path, map_location=device))
