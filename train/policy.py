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
from utils.util import orthogonal_init_
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
        self.layer_1 = orthogonal_init_(nn.Linear(input_shape, hidden_size))
        self.layer_2 = orthogonal_init_(nn.Linear(num, 1))
        
    def forward(self, input):
        """
        input shape : [batch_size, vector_num, vector_length]
        """
        batch_size, _, _ = input.shape
        layer_1_out = F.relu(self.layer_1(input)).permute(0,2,1).contiguous()
        layer_2_out = F.relu(self.layer_2(layer_1_out))
        layer_2_out = layer_2_out.view(batch_size, self.hidden_size)
        return layer_2_out

class Feature_Net(nn.Module):
    def __init__(self):

        super().__init__()
        self.sur_feature_net = TimeVecFeatureNet(70, 5, 128)
        self.ego_feature_net = TimeVecFeatureNet(8, 5, 128)
        self.feature_net = nn.Sequential(
                           orthogonal_init_(nn.Linear(256, 256)), 
                           nn.ReLU())
    
    def forward(self, sur_obs, ego_obs):

        ego_feature = self.ego_feature_net(ego_obs)
        sur_feature = self.sur_feature_net(sur_obs)
        env_feature = torch.concat([ego_feature, sur_feature], dim=1)
        env_feature = self.feature_net(env_feature)

        return env_feature

class PPOPolicy(nn.Module):
    def __init__(self, num_outputs):
        nn.Module.__init__(self)
        self.policy_param = PolicyParam
        self.share_net = self.policy_param.share
        self.independent_std = self.policy_param.independent_std
        # actor and critic share the feature network 
        if self.share_net:
            self.feature_net = Feature_Net()
        else:
            self.actor_feature_net = Feature_Net()
            self.critic_feature_net = Feature_Net()
        # orthogonal_init trick
        #initialization of weights with scaling np.sqrt(2), and the biases are set to 0
        # the policy output layer weights are initialized with the scale of 0.01
        if self.independent_std:
            self.actor_net = nn.Sequential(
                OrderedDict(
                    [
                        ("actor_1", orthogonal_init_(nn.Linear(256, 256))),
                        ("actor_relu_1", nn.ReLU()),
                        ("actor_mu", orthogonal_init_(nn.Linear(256, num_outputs), gain=0.01)),
                    ]
                )
            )
            # std is independent of the states
            self.log_std = torch.nn.Parameter(torch.zeros(num_outputs), requires_grad=True)
        else:
            self.actor_net = nn.Sequential(
                OrderedDict(
                    [
                        ("actor_1", orthogonal_init_(nn.Linear(256, 256))),
                        ("actor_relu_1", nn.ReLU()),
                        ("actor_mu", orthogonal_init_(nn.Linear(256, num_outputs*2), gain=0.01)),
                    ]
                )
            )

        self.critic_net = nn.Sequential(
            OrderedDict(
                [
                    ("critic_1", orthogonal_init_(nn.Linear(256, 256))),
                    ("critic_relu_1", nn.ReLU()),
                    ("critic_output", orthogonal_init_(nn.Linear(256, 1))),
                ]
            )
        )

        self.sur_obs_norm = Normalization(70)
        self.ego_obs_norm = Normalization(8)

    def get_env_feature(self, sur_obs, ego_obs):
        
        ego_obs = torch.clamp(self.ego_obs_norm.normalize(ego_obs), min=-5, max=5)
        sur_obs = torch.clamp(self.sur_obs_norm.normalize(sur_obs), min=-5, max=5)
        if self.share_net:
            env_feature = self.feature_net(sur_obs, ego_obs)
        else:
            actor_env_feature = self.actor_feature_net(sur_obs, ego_obs)
            critic_env_feature = self.critic_feature_net(sur_obs, ego_obs)
            env_feature = dict(actor_env_feature=actor_env_feature, 
                               critic_env_feature=critic_env_feature)
        return env_feature

    def get_value(self, env_feature):
        
        if type(env_feature) == dict:
            values = self.critic_net(env_feature['critic_env_feature'])
        else:
            values = self.critic_net(env_feature)

        return values

    def update_norm(self, sur_obs, ego_obs):
        
        self.sur_obs_norm.update(sur_obs)
        self.ego_obs_norm.update(ego_obs)
    
    def select_action(self, sur_obs, ego_obs, deterministic=False):
        
        env_feature = self.get_env_feature(sur_obs, ego_obs)
        if type(env_feature) == dict:
            action_out = self.actor_net(env_feature['actor_env_feature'])
            value = self.critic_net(env_feature['critic_env_feature'])
        else:
            action_out = self.actor_net(env_feature)
            value = self.critic_net(env_feature)
        if self.independent_std:
            action_mean = action_out 
            action_std = torch.exp(self.log_std)
        else:
            action_mean, log_std = torch.chunk(action_out, 2, dim=-1)
            action_std = torch.exp(log_std)

        action_distribution = Normal(action_mean, action_std)
        # 采样
        if not deterministic:
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
        if type(env_feature) == dict:
            action_out = self.actor_net(env_feature['actor_env_feature'])
        else:
            action_out = self.actor_net(env_feature)
        if self.independent_std:
            action_mean = action_out 
            action_std = torch.exp(self.log_std)
        else:
            action_mean, log_std = torch.chunk(action_out, 2, dim=-1)
            action_std = torch.exp(log_std)
            
        action_distribution = Normal(action_mean, action_std)
        logp_pi = self._cal_tanhlogproba(action_distribution, actions)

        return logp_pi

    def load_model(self, model_path, device):
        self.load_state_dict(torch.load(model_path, map_location=device))
