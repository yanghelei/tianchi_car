import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from tianshou.utils.net.common import MLP
from utils.networks.norm import Normalization
from utils.networks.utils import check, orthogonal_init_


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
        layer_1_out = F.relu(self.layer_1(input)).permute(0, 2, 1).contiguous()
        layer_2_out = F.relu(self.layer_2(layer_1_out))
        layer_2_out = layer_2_out.view(batch_size, self.hidden_size)
        return layer_2_out


class MyActor(nn.Module):
    def __init__(
            self,
            cfgs,
            action_shape=0,
            norm_layer=None,
            activation=nn.ReLU,
            device="cpu",
            softmax=False,
            concat=False,
            num_atoms=1,
            dueling_param=None,
            linear_layer=nn.Linear,
    ):
        super().__init__()
        self.device = device
        self.softmax = softmax
        self.num_atoms = num_atoms

        self.sur_norm = Normalization(input_shape=cfgs.sur_in, device=self.device)
        self.ego_norm = Normalization(input_shape=cfgs.ego_in, device=self.device)

        self.sur_project = MLP(input_dim=cfgs.sur_in,
                               output_dim=cfgs.sur_out,
                               hidden_sizes=cfgs.sur_hiddens,
                               device=self.device,
                               flatten_input=False)
        self.ego_project = MLP(input_dim=cfgs.ego_in,
                               output_dim=cfgs.ego_out,
                               hidden_sizes=cfgs.ego_hiddens,
                               device=self.device,
                               flatten_input=False)
        self.project = MLP(input_dim=cfgs.sur_out + cfgs.ego_out,
                           output_dim=cfgs.total_hiddens[-1],
                           hidden_sizes=cfgs.total_hiddens,
                           device=self.device,
                           flatten_input=False)

        self.output_dim = cfgs.total_hiddens[-1]

        self.time_transform = TimeVecFeatureNet(input_shape=self.output_dim,
                                                num=5,
                                                hidden_size=self.output_dim)

        self.tpdv = dict(dtype=torch.float32, device=device)

        input_dim = self.output_dim  # int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape)) * num_atoms

        if concat:
            input_dim = action_dim

        self.use_dueling = dueling_param is not None
        output_dim = action_dim if not self.use_dueling and not concat else 0

        self.model = MLP(input_dim=input_dim,
                         output_dim=output_dim,
                         hidden_sizes=cfgs.action_hiddens,
                         norm_layer=norm_layer,
                         activation=activation,
                         device=device,
                         linear_layer=linear_layer)

        self.output_dim = self.model.output_dim

        if self.use_dueling:  # dueling DQN
            q_kwargs, v_kwargs = dueling_param  # type: ignore
            q_output_dim, v_output_dim = 0, 0

            if not concat:
                q_output_dim, v_output_dim = action_dim, num_atoms

            q_kwargs = {
                **q_kwargs,
                "input_dim": self.output_dim,
                "output_dim": q_output_dim,
                "device": self.device
            }
            v_kwargs = {
                **v_kwargs,
                "input_dim": self.output_dim,
                "output_dim": v_output_dim,
                "device": self.device
            }
            self.Q, self.V = MLP(**q_kwargs), MLP(**v_kwargs)
            self.output_dim = self.Q.output_dim

    def pre_process(self, obs):

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

        # 聚合前5帧的信息
        feats = self.time_transform(feats)

        return feats

    def forward(self, obs, state=None, info=None):
        if info is None:
            info = dict()

        feats = self.pre_process(obs)

        logits = self.model(feats)
        bsz = logits.shape[0]

        if self.use_dueling:  # Dueling DQN
            q, v = self.Q(logits), self.V(logits)
            if self.num_atoms > 1:
                q = q.view(bsz, -1, self.num_atoms)
                v = v.view(bsz, -1, self.num_atoms)
            logits = q - q.mean(dim=1, keepdim=True) + v
        elif self.num_atoms > 1:
            logits = logits.view(bsz, -1, self.num_atoms)
        if self.softmax:
            logits = torch.softmax(logits, dim=-1)

        return logits, state