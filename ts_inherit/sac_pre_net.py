import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from utils.networks.utils import check, orthogonal_init_
from utils.networks.norm import Normalization

from tianshou.utils.net.common import MLP


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
        # layer_1_out = F.relu(self.layer_1(input)).permute(0, 2, 1).contiguous()
        layer_1_out = input.permute(0, 2, 1).contiguous()
        layer_2_out = F.relu(self.layer_2(layer_1_out))
        layer_2_out = layer_2_out.view(batch_size, self.hidden_size)
        return layer_2_out


class PreNetworks(nn.Module):

    def __init__(self, cfgs):
        super().__init__()

        self.device = cfgs.device

        self.sur_norm = Normalization(input_shape=cfgs.network.sur_in, device=self.device)
        self.ego_norm = Normalization(input_shape=cfgs.network.ego_in, device=self.device)

        self.sur_project = MLP(input_dim=cfgs.network.sur_in,
                               output_dim=cfgs.network.sur_out,
                               hidden_sizes=cfgs.network.sur_hiddens,
                               device=self.device,
                               flatten_input=False)
        self.ego_project = MLP(input_dim=cfgs.network.ego_in,
                               output_dim=cfgs.network.ego_out,
                               hidden_sizes=cfgs.network.ego_hiddens,
                               device=self.device,
                               flatten_input=False)
        self.project = MLP(input_dim=cfgs.network.sur_out + cfgs.network.ego_out,
                           output_dim=cfgs.network.frame_out,
                           hidden_sizes=cfgs.network.frame_hiddens,
                           device=self.device,
                           flatten_input=False)

        self.time_transform = TimeVecFeatureNet(input_shape=cfgs.network.frame_out,
                                                num=cfgs.history_length,
                                                hidden_size=cfgs.network.time_out)

        self.output_dim = cfgs.network.time_out

        self.tpdv = dict(dtype=torch.float32, device=self.device)

    def forward(self, obs, state=None):

        sur_obs_data = check(obs['sur_obs']["data"]).to(**self.tpdv)
        sur_obs_data = self.sur_norm.normalize(sur_obs_data)
        sur_feat = self.sur_project(sur_obs_data)
        sur_feat = torch.mean(sur_feat, dim=2, keepdim=False)  # dim: <env, history, agent, feat>

        ego_obs_data = check(obs['ego_obs']["data"]).to(**self.tpdv)
        ego_obs_data = self.ego_norm.normalize(ego_obs_data)
        ego_feat = self.ego_project(ego_obs_data)
        ego_feat = torch.mean(ego_feat, dim=2, keepdim=False)

        feats = torch.cat([sur_feat, ego_feat], dim=2)
        feats = self.project(feats)

        # 聚合历史n帧的信息
        feats = self.time_transform(feats)

        return feats, state
