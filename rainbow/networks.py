import torch
import numpy as np
from torch import nn
from tianshou.utils.net.common import MLP


class MyActor(nn.Module):
    def __init__(
            self,
            state_shape,
            action_shape=0,
            hidden_sizes=(),
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

        input_dim = int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape)) * num_atoms
        if concat:
            input_dim = action_dim

        self.use_dueling = dueling_param is not None
        output_dim = action_dim if not self.use_dueling and not concat else 0

        self.model = MLP(input_dim, output_dim, hidden_sizes, norm_layer, activation, device, linear_layer)

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

    def forward(self, obs, state=None, info=None):
        if info is None:
            info = dict()
        logits = self.model(obs)
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
