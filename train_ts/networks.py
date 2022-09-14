import torch
import numpy as np
from torch import nn
from algo_ts.utils.net.common import MLP
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Union, Type


class MyActor(nn.Module):
    """Simple actor network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> action_shape.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param float max_action: the scale for the final action logits. Default to
        1.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~algo_torch.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

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

        self.sur_norm = sur_norm
        self.ego_norm = ego_norm

        self.sur_feature_net = MLP(
            input_dim=54,
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

    def pre_process(self, obs):
        sur_obs_data = torch.from_numpy(obs['sur_obs'])  # dim: < env, history, feats >
        sur_obs_data = self.sur_norm.normalize(sur_obs_data)

        sur_obs_data = self.sur_feature_net(sur_obs_data)  # dim: <env, history, new_feats_dim>

        ego_obs_data = torch.from_numpy(obs['ego_obs'])
        ego_obs_data = self.ego_norm.normalize(ego_obs_data)

        ego_obs_data = self.ego_feature_net(ego_obs_data)

        feats = torch.cat([sur_obs_data, ego_obs_data], dim=2)

        return feats  # dim: <env, feats>

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            state: Any = None,
            info: Dict[str, Any] = {},
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Any]:
        """Mapping: obs -> logits."""
        feats = self.pre_process(obs)

        logits = self.last(feats)
        if self.softmax_output:
            logits = F.softmax(logits, dim=-1)
        return logits, state


class MyCritic(nn.Module):
    """Simple critic network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.
    :param bool flatten_input: whether to flatten input data for the last layer.
        Default to True.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~algo_torch.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

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

        self.sur_norm = sur_norm
        self.ego_norm = ego_norm

        self.sur_feature_net = MLP(
            input_dim=54,
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

    def pre_process(self, obs):
        sur_obs_data = torch.from_numpy(obs['sur_obs'])  # dim: < env, history, feats >
        sur_obs_data = self.sur_norm.normalize(sur_obs_data)

        sur_obs_data = self.sur_feature_net(sur_obs_data)  # dim: <env, history, new_feats_dim>

        ego_obs_data = torch.from_numpy(obs['ego_obs'])
        ego_obs_data = self.ego_norm.normalize(ego_obs_data)

        ego_obs_data = self.ego_feature_net(ego_obs_data)

        feats = torch.cat([sur_obs_data, ego_obs_data], dim=2)

        return feats  # dim: <env, feats>

    def forward(
            self,
            obs: Union[np.ndarray, torch.Tensor],
            info=None,
    ) -> torch.Tensor:
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        if info is None:
            info = dict()

        feats = self.pre_process(obs)
        logits = self.last(feats)
        return logits