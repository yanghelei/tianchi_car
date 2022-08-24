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
        super.__init__()
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
        self.actor_logit = nn.Sequential(
            OrderedDict(
                [
                    ("actor_1", nn.Linear(256, 256)),
                    ("actor_relu_1", nn.ReLU()),
                    ("actor_2", nn.Linear(256, 256)),
                    ("actor_relu_2", nn.ReLU()),
                ]
            )
        )

        self.critic_value = nn.Sequential(
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

        self.logit = nn.Sequential(nn.Linear(256, num_outputs * 2), nn.Tanh())

        initialize_weights(self, "orthogonal", scale = np.sqrt(2))

    
    def get_env_feature(self, sur_obs, ego_obs):

        ego_feature = self.ego_feature_net(ego_obs)
        sur_feature = self.sur_feature_net(sur_obs)
        env_feature = torch.concat([ego_feature, sur_feature], dim=1)

        return env_feature

    def _forward_actor(self, env_feature):

        logits = self.actor_logit(env_feature)
        action_logit = self.logit(logits)
        action_mean, action_logstd = torch.chunk(action_logit, 2, dim=-1)
        values = self.critic_value(env_feature)
        return action_mean, action_logstd, values

    def _forward_critic(self, env_feature):

        values = self.critic_value(env_feature)
        return values

    def forward(self, sur_obs, ego_obs):
        action_mean, action_logstd = self._forward_actor(sur_obs, ego_obs)
        critic_value = self._forward_critic(sur_obs, ego_obs)
        return action_mean, action_logstd, critic_value

    def select_action(self, action_mean, action_logstd):
        action_std = torch.exp(action_logstd)
        action = torch.normal(action_mean, action_std)
        logproba = self._normal_logproba(action, action_mean, action_logstd, action_std)
        return action, logproba

    @staticmethod
    def _normal_logproba(x, mean, logstd, std=None):
        if std is None:
            std = torch.exp(logstd)

        std_sq = std.pow(2)
        logproba = -0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)
        return logproba.sum(1)

    def get_logproba(self, env_feature, actions):
        action_mean, action_logstd = self._forward_actor(env_feature)
        logproba = self._normal_logproba(actions, action_mean, action_logstd)
        entropy = torch.distributions.Normal(action_mean, torch.exp(action_logstd)).entropy()
        return logproba, entropy
