import torch
import numpy as np
from torch import nn
from algo.utils.net.common import MLP
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Union, Type


class MyActor(nn.Module):
    def __init__(
            self,
            config,
            sur_norm,
            ego_norm,
            norm_layer=None,
            activation=nn.ReLU,
            linear_layer=nn.Linear,
            device: Union[str, int, torch.device] = "cpu",
            softmax_output=True
    ) -> None:
        super().__init__()
        self.device = device
        self.config = config

        self.sur_norm = sur_norm
        self.ego_norm = ego_norm

        self.sur_feature_net = MLP(
            input_dim=48,
            output_dim=128,
            hidden_sizes=(),
            device=self.device,
            flatten_input=False,
        )

        self.ego_feature_net = MLP(
            input_dim=11,
            output_dim=64,
            hidden_sizes=(),
            device=self.device,
            flatten_input=False,
        )

        self.last_input_dim = 192 * config.history_length
        self.output_dim = config.action_shape

        self.last = MLP(input_dim=self.last_input_dim,
                        output_dim=self.output_dim,
                        hidden_sizes=config.hidden_sizes,
                        norm_layer=norm_layer,
                        activation=activation,
                        device=device,
                        linear_layer=linear_layer)

        self.softmax_output = softmax_output

        self.tpdv = dict(dtype=torch.float32, device=config.device)

    def pre_process(self, obs):
        sur_obs_data = torch.from_numpy(obs['sur_obs']).to(**self.tpdv)  # dim: < env, history, feats >
        sur_obs_data = self.sur_norm.normalize(sur_obs_data)

        sur_obs_data = self.sur_feature_net(sur_obs_data)  # dim: <env, history, new_feats_dim>

        ego_obs_data = torch.from_numpy(obs['ego_obs']).to(**self.tpdv)
        ego_obs_data = self.ego_norm.normalize(ego_obs_data)

        ego_obs_data = self.ego_feature_net(ego_obs_data)

        feats = torch.cat([sur_obs_data, ego_obs_data], dim=2)

        mask = torch.from_numpy(obs['mask']).to(device=self.config.device)

        return feats, mask  # dim: <env, feats>

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Any = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Any]:
        """Mapping: obs -> logits."""
        feats, mask = self.pre_process(obs)

        logits = self.last(feats)
        if self.softmax_output:
            logits = F.softmax(logits, dim=-1)
        return logits, mask, state


class MyCritic(nn.Module):
    def __init__(
            self,
            config,
            sur_norm,
            ego_norm,
            norm_layer=None,
            activation=nn.ReLU,
            device: Union[str, int, torch.device] = "cpu",
            linear_layer: Type[nn.Linear] = nn.Linear,
    ) -> None:
        super().__init__()
        self.device = device
        self.config = config

        self.sur_norm = sur_norm
        self.ego_norm = ego_norm

        self.sur_feature_net = MLP(
            input_dim=48,
            output_dim=128,
            hidden_sizes=(),
            device=self.device,
            flatten_input=False
        )

        self.ego_feature_net = MLP(
            input_dim=11,
            output_dim=64,
            hidden_sizes=(),
            device=self.device,
            flatten_input=False
        )

        self.last_input_dim = 192 * config.history_length
        self.output_dim = config.action_shape

        self.last = MLP(input_dim=self.last_input_dim,
                        output_dim=self.output_dim,
                        hidden_sizes=config.hidden_sizes,
                        norm_layer=norm_layer,
                        activation=activation,
                        device=device,
                        linear_layer=linear_layer)

        self.output_dim = self.last.output_dim

        self.tpdv = dict(dtype=torch.float32, device=config.device)

    def pre_process(self, obs):
        sur_obs_data = torch.from_numpy(obs['sur_obs']).to(**self.tpdv)  # dim: < env, history, feats >
        sur_obs_data = self.sur_norm.normalize(sur_obs_data)

        sur_obs_data = self.sur_feature_net(sur_obs_data)  # dim: <env, history, new_feats_dim>

        ego_obs_data = torch.from_numpy(obs['ego_obs']).to(**self.tpdv)
        ego_obs_data = self.ego_norm.normalize(ego_obs_data)

        ego_obs_data = self.ego_feature_net(ego_obs_data)

        feats = torch.cat([sur_obs_data, ego_obs_data], dim=2)

        mask = torch.from_numpy(obs['mask']).to(device=self.config.device)

        return feats, mask  # dim: <env, feats>

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            info=None,
    ) -> torch.Tensor:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        if info is None:
            info = dict()

        feats, mask = self.pre_process(obs)
        logits = self.last(feats)
        logits = torch.where(mask, logits, mask)
        return logits