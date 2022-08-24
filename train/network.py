import torch.nn as nn 
from backborn.mlp_layer import mlp
import numpy as np 
import torch.nn.functional as F
import torch 
from torch.distributions import Categorical, Normal

class TimeVecFeature(nn.Module):
    def __init__(self, input_shape, num, hidden_size):
        """
        input_shape is the vector length; num is the number of sequence;  
        """
        super.__init__()
        self.hidden_size = hidden_size
        self.layer_1 = nn.Linear(input_shape, hidden_size)
        self.layer_2 = nn.Linear(num, 1)
        nn.init.orthogonal_(self.layer_1, np.sqrt(2))
        nn.init.orthogonal_(self.layer_2, np.sqrt(2))
        
    def forward(self, input):
        """
        input shape : [batch_size, vector_num, vector_length]
        """
        batch_size, _, _ = input.shape
        layer_1_out = F.relu(self.layer_1(input)).permute(0,2,1).contiguous()
        layer_2_out = F.relu(self.layer_2(layer_1_out))
        layer_2_out = layer_2_out.view(batch_size, self.hidden_size)
        return layer_2_out

class MLPCategoricalAC(nn.Module):

    def __init__(self, act_dim):
        super().__init__()
        self.act_dim = act_dim
        # share the feature net 
        self.sur_feature = TimeVecFeature(70, 5, 128)
        self.ego_feature = TimeVecFeature(8, 5, 128)
        self.pi_net = mlp([256, 256, act_dim], activation=nn.Relu) 
        self.v_net = mlp([256, 256, 1], activation=nn.Relu)

    def forward(self, input):

        ego_vec, sur_vec = input 
        ego_state = self.ego_feature(ego_vec)
        sur_state = self.sur_feature(sur_vec)
        full_state = torch.concat([ego_state, sur_state])
        pi = self.pi_net(full_state)
        value = self.v_net(full_state)

        return pi, value 
    





        