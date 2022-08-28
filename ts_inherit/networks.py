import torch
import numpy as np
from torch import nn
from tianshou.utils.net.common import MLP
from utils.networks.mlp import MLPLayer
from utils.networks.norm import Normalization
from utils.networks.utils import check


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

        self.sur_norm = Normalization(input_shape=cfgs.sur_dim, device=self.device)
        self.ego_norm = Normalization(input_shape=cfgs.ego_dim, device=self.device)

        self.sur_project = MLPLayer(input_dim=cfgs.sur_dim,
                                    hidden_size=cfgs.sur_hidden,
                                    layer_N=1,
                                    use_orthogonal=True,
                                    use_ReLU=True)
        self.ego_project = MLPLayer(input_dim=cfgs.ego_dim,
                                    hidden_size=cfgs.ego_hidden,
                                    layer_N=1,
                                    use_orthogonal=True,
                                    use_ReLU=True)
        self.project = MLPLayer(input_dim=cfgs.sur_hidden+cfgs.ego_hidden,
                                hidden_size=cfgs.total_hidden,
                                layer_N=1,
                                use_orthogonal=True,
                                use_ReLU=True)

        self.tpdv = dict(dtype=torch.float32, device=device)

        input_dim = cfgs.total_hidden  # int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape)) * num_atoms

        if concat:
            input_dim = action_dim

        self.use_dueling = dueling_param is not None
        output_dim = action_dim if not self.use_dueling and not concat else 0

        self.model = MLP(input_dim, output_dim, cfgs.action_hidden, norm_layer, activation, device, linear_layer)

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

        sur_feats = list()
        sur_obs = obs['sur_obs']
        for _idx in range(len(sur_obs['n'])):
            n = sur_obs['n'][_idx]

            data = sur_obs['data'][_idx][:n]
            data = check(data).to(**self.tpdv)
            data = self.sur_norm.normalize(data)

            sur_feat = self.sur_project(data)
            sur_feat = torch.mean(sur_feat, dim=0, keepdim=False)
            sur_feats.append(sur_feat)
        sur_feats = torch.stack(sur_feats)

        ego_feats = list()
        ego_obs = obs['ego_obs']
        for _idx in range(len(ego_obs['n'])):
            n = ego_obs['n'][_idx]

            data = ego_obs['data'][_idx][:n]
            data = check(data).to(**self.tpdv)
            data = self.ego_norm.normalize(data)

            ego_feat = self.ego_project(data)
            ego_feat = torch.mean(ego_feat, dim=0, keepdim=False)
            ego_feats.append(ego_feat)
        ego_feats = torch.stack(ego_feats)

        total_feats = torch.cat([sur_feats, ego_feats], dim=1)
        total_feats = self.project(total_feats)
        return total_feats

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
