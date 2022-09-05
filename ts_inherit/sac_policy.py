from typing import Any, Dict, Optional, Union

import torch
import itertools
import numpy as np
from math import pi
from torch.distributions import Categorical

from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import DiscreteSACPolicy

from sac.config import cfg


class SacPolicy(DiscreteSACPolicy):

    def set_logger(self, logger):
        if not hasattr(self, 'logger'):
            setattr(self, 'logger', logger)

    def make_action_library(self, cfgs):
        steer_prime_choices = cfgs.steer_prime_choices
        acc_prime_choice = cfgs.acc_prime_choice
        self.action_library = list(itertools.product(steer_prime_choices, acc_prime_choice))

    def map_action(self, data):
        assert hasattr(self, 'action_library')
        assert len(data.act.shape) <= 2, f"Unknown action format with shape {data.act.shape}."

        obs = data.obs['ego_obs']['data']
        act = data.act

        if len(data.act.shape) == 1:
            action = list()
            for _idx in range(len(act)):
                _act = act[_idx]
                _steer_prime, _acc_prime = self.action_library[_act]
                _steer = np.clip(obs[_idx][-1][0][5] + _steer_prime * cfg.dt, -pi/36.0, pi/36.0)
                _acc = np.clip(obs[_idx][-1][0][7] + _acc_prime * cfg.dt, -2.0, 2.0)
                action.append([_steer, _acc])
            return np.array(action, dtype=np.float32)
        # if len(data.act.shape) == 1:
        #     return np.array([self.action_library[int(a)] for a in data.act])
        # return np.array([[self.action_library[int(a)] for a in a_] for a_ in data.act])

    def sync_weight(self) -> None:
        self.soft_update(self.critic1_old, self.critic1, self.tau)
        self.soft_update(self.critic2_old, self.critic2, self.tau)

    def forward(  # type: ignore
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            input: str = "obs",
            **kwargs: Any,
    ) -> Batch:
        obs = batch[input]

        logits, hidden = self.actor(obs, state=state, info=batch.info)
        dist = Categorical(logits=logits)

        if self._deterministic_eval and not self.training:
            act = logits.argmax(axis=-1)
        else:
            act = dist.sample()

        return Batch(logits=logits, act=act, state=hidden, dist=dist)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs: s_{t+n}
        obs_next_result = self(batch, input="obs_next")
        dist = obs_next_result.dist
        target_q = dist.probs * torch.min(
            self.critic1_old(batch.obs_next),
            self.critic2_old(batch.obs_next),
        )
        target_q = target_q.sum(dim=-1) + self._alpha * dist.entropy()
        return target_q

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        weight = batch.pop("weight", 1.0)
        target_q = batch.returns.flatten()
        act = to_torch(batch.act[:, np.newaxis], device=target_q.device, dtype=torch.long)

        # critic 1
        current_q1 = self.critic1(batch.obs).gather(1, act).flatten()
        td1 = current_q1 - target_q
        critic1_loss = (td1.pow(2) * weight).mean()

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        # critic 2
        current_q2 = self.critic2(batch.obs).gather(1, act).flatten()
        td2 = current_q2 - target_q
        critic2_loss = (td2.pow(2) * weight).mean()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        # actor
        dist = self(batch).dist
        entropy = dist.entropy()
        with torch.no_grad():
            current_q1a = self.critic1(batch.obs)
            current_q2a = self.critic2(batch.obs)
            q = torch.min(current_q1a, current_q2a)
        actor_loss = -(self._alpha * entropy + (dist.probs * q).sum(dim=-1)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_prob = -entropy.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_prob).mean()
            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()

        self.sync_weight()

        result = {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
        }
        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()  # type: ignore

        return result
