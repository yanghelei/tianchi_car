import time
import torch
import warnings
import numpy as np
from typing import Any, Dict, List, Optional, Union

from tianshou.data import (
    Batch,
    Collector,
    CachedReplayBuffer,
    ReplayBuffer,
    ReplayBufferManager,
    VectorReplayBuffer,
    to_numpy,
)


class MyCollector(Collector):

    def set_logger(self, logger, type='test'):
        logger_name = type + '_logger'
        if not hasattr(self, logger_name):
            setattr(self, logger_name, logger)

    def _assign_buffer(self, buffer: Optional[ReplayBuffer]) -> None:
        """Check if the buffer matches the constraint."""
        if buffer is None:
            buffer = VectorReplayBuffer(self.env_num, self.env_num)
        elif isinstance(buffer, ReplayBufferManager):
            assert buffer.buffer_num >= self.env_num
            if isinstance(buffer, CachedReplayBuffer):
                assert buffer.cached_buffer_num >= self.env_num
        else:  # ReplayBuffer or PrioritizedReplayBuffer
            assert buffer.maxsize > 0
            if self.env_num > 1:
                if type(buffer) == ReplayBuffer:
                    buffer_type = "ReplayBuffer"
                    vector_type = "VectorReplayBuffer"
                else:
                    buffer_type = "PrioritizedReplayBuffer"
                    vector_type = "PrioritizedVectorReplayBuffer"
                raise TypeError(
                    f"Cannot use {buffer_type}(size={buffer.maxsize}, ...) to collect "
                    f"{self.env_num} envs,\n\tplease use {vector_type}(total_size="
                    f"{buffer.maxsize}, buffer_num={self.env_num}, ...) instead."
                )
        self.buffer = buffer

    def reset(
        self,
        reset_buffer: bool = True,
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Reset the environment, statistics, current data and possibly replay memory.

        :param bool reset_buffer: if true, reset the replay buffer that is attached
            to the collector.
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Defaults to None (extra keyword arguments)
        """
        # use empty Batch for "state" so that self.data supports slicing
        # convert empty Batch to None when passing data to policy
        self.data = Batch(
            obs={}, act={}, rew={}, done={}, obs_next={}, info={}, policy={}
        )
        self.reset_env(gym_reset_kwargs)
        if reset_buffer:
            self.reset_buffer()
        self.reset_stat()

    def reset_stat(self) -> None:
        """Reset the statistic variables."""
        self.collect_step, self.collect_episode, self.collect_time = 0, 0, 0.0

    def collect(self, n_step=None, n_episode=None, random=False, render=None, no_grad=True, gym_reset_kwargs=None,) -> Dict[str, Any]:
        """Collect a specified number of step or episode.

        To ensure unbiased sampling result with n_episode option, this function will
        first collect ``n_episode - env_num`` episodes, then for the last ``env_num``
        episodes, they will be collected evenly from each env.

        :param int n_step: how many steps you want to collect.
        :param int n_episode: how many episodes you want to collect.
        :param bool random: whether to use random policy for collecting data. Default to False.
        :param float render: the sleep time between rendering consecutive frames. Default to None (no rendering).
        :param bool no_grad: whether to retain gradient in policy.forward(). Default to True (no gradient retaining).
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's reset function. Defaults to None (extra keyword arguments)

        .. note::

            One and only one collection number specification is permitted, either
            ``n_step`` or ``n_episode``.

        :return: A dict including the following keys

            * ``n/ep`` collected number of episodes.
            * ``n/st`` collected number of steps.
            * ``rews`` array of episode reward over collected episodes.
            * ``lens`` array of episode length over collected episodes.
            * ``idxs`` array of episode start index in buffer over collected episodes.
            * ``rew`` mean of episodic rewards.
            * ``len`` mean of episodic lengths.
            * ``rew_std`` standard error of episodic rewards.
            * ``len_std`` standard error of episodic lengths.
        """
        assert not self.env.is_async, "Please use AsyncCollector if using async venv."
        if n_step is not None:
            assert n_episode is None, f"Only one of n_step or n_episode is allowed in Collector. collect, got n_step={n_step}, n_episode={n_episode}."
            assert n_step > 0
            if not n_step % self.env_num == 0:
                warnings.warn(f"n_step={n_step} is not a multiple of #env ({self.env_num}), which may cause extra transitions collected into the buffer.")
            ready_env_ids = np.arange(self.env_num)
        elif n_episode is not None:
            assert n_episode > 0
            ready_env_ids = np.arange(min(self.env_num, n_episode))
            self.data = self.data[:min(self.env_num, n_episode)]
        else:
            raise TypeError("Please specify at least one (either n_step or n_episode) in AsyncCollector.collect().")

        start_time = time.time()

        step_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []
        episode_start_indices = []

        reach_goal = 0
        time_out = 0
        collision=0

        while True:
            assert len(self.data) == len(ready_env_ids)
            # restore the state: if the last state is None, it won't store
            last_state = self.data.policy.pop("hidden_state", None)

            # get the next action
            if random:
                try:
                    act_sample = [self._action_space[i].sample() for i in ready_env_ids]
                except TypeError:  # envpool's action space is not for per-env
                    act_sample = [self._action_space.sample() for _ in ready_env_ids]
                act_sample = self.policy.map_action_inverse(act_sample)  # type: ignore
                self.data.update(act=act_sample)
            else:
                if no_grad:
                    with torch.no_grad():  # faster than retain_grad version
                        # self.data.obs will be used by agent to get result
                        result = self.policy(self.data, last_state)
                else:
                    result = self.policy(self.data, last_state)
                # update state / act / policy into self.data
                policy = result.get("policy", Batch())
                assert isinstance(policy, Batch)
                state = result.get("state", None)
                if state is not None:
                    policy.hidden_state = state  # save state into buffer
                act = to_numpy(result.act)
                if self.exploration_noise:
                    act = self.policy.exploration_noise(act, self.data)
                self.data.update(policy=policy, act=act)

            # get bounded and remapped actions first (not saved into buffer)
            action_remap = self.policy.map_action(self.data)
            # step in env
            result = self.env.step(action_remap, ready_env_ids)  # type: ignore
            obs_next, rew, done, info = result

            """ Batch 不支持 key 为 int 的字典，此处对 info 进行处理 """
            for _idx in range(len(info)):
                agent_infos = dict()
                # for key, value in info[_idx]['agent_infos'].items():
                #     agent_infos[str(key)] = value
                info[_idx]['agent_infos'] = agent_infos

            self.data.update(obs_next=obs_next, rew=rew, done=done, info=info)
            if self.preprocess_fn:
                self.data.update(
                    self.preprocess_fn(
                        obs_next=self.data.obs_next,
                        rew=self.data.rew,
                        done=self.data.done,
                        info=self.data.info,
                        policy=self.data.policy,
                        env_id=ready_env_ids,
                    )
                )

            if render:
                self.env.render()
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)

            # add data into the buffer
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(self.data, buffer_ids=ready_env_ids)

            # collect statistics
            step_count += len(ready_env_ids)

            if np.any(done):
                env_ind_local = np.where(done)[0]
                if hasattr(self, 'test_logger'):
                    for env_ind in env_ind_local:
                        if info[env_ind]['reached_stoparea']:
                            reach_goal += 1
                        elif info[env_ind]['timeout']:
                            time_out += 1
                        elif info[env_ind]['collided']:
                            collision += 1
                        else:
                            self.test_logger.info(f'Done without reasonable condition!')
                env_ind_global = ready_env_ids[env_ind_local]
                episode_count += len(env_ind_local)
                episode_lens.append(ep_len[env_ind_local])
                episode_rews.append(ep_rew[env_ind_local])
                episode_start_indices.append(ep_idx[env_ind_local])
                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.
                self._reset_env_with_ids(env_ind_local, env_ind_global, gym_reset_kwargs)
                for i in env_ind_local:
                    self._reset_state(i)

                # remove surplus env id from ready_env_ids
                # to avoid bias in selecting environments
                if n_episode:
                    surplus_env_num = len(ready_env_ids) - (n_episode - episode_count)
                    if surplus_env_num > 0:
                        mask = np.ones_like(ready_env_ids, dtype=bool)
                        mask[env_ind_local[:surplus_env_num]] = False
                        ready_env_ids = ready_env_ids[mask]
                        self.data = self.data[mask]

            self.data.obs = self.data.obs_next

            if (n_step and step_count >= n_step) or (n_episode and episode_count >= n_episode):
                break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        if n_episode:
            self.data = Batch(obs={}, act={}, rew={}, done={}, obs_next={}, info={}, policy={})
            self.reset_env()

        if episode_count > 0:
            rews, lens, idxs = list(map(np.concatenate, [episode_rews, episode_lens, episode_start_indices]))
            rew_mean, rew_std = rews.mean(), rews.std()
            len_mean, len_std = lens.mean(), lens.std()
        else:
            rews, lens, idxs = np.array([]), np.array([], int), np.array([], int)
            rew_mean = rew_std = len_mean = len_std = 0

        if hasattr(self, 'train_logger'):
            self.train_logger.info(f'Collector collected:{step_count}\tCost:{max(time.time() - start_time, 1e-9)}s')
        if hasattr(self, 'test_logger'):
            self.test_logger.info(f'Reach goal rate:{reach_goal}/{episode_count} \tCollision rate:{collision}/{episode_count} \tTimeout rate:{time_out}/{episode_count}')

        return {
            "n/ep": episode_count,
            "n/st": step_count,
            "rews": rews,
            "lens": lens,
            "idxs": idxs,
            "rew": rew_mean,
            "len": len_mean,
            "rew_std": rew_std,
            "len_std": len_std,
        }

    def _reset_env_with_ids(
        self,
        local_ids: Union[List[int], np.ndarray],
        global_ids: Union[List[int], np.ndarray],
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        gym_reset_kwargs = gym_reset_kwargs if gym_reset_kwargs else {}
        rval = self.env.reset(global_ids, **gym_reset_kwargs)
        returns_info = isinstance(rval, (tuple, list)) and len(rval) == 2 and (
            isinstance(rval[1], dict) or isinstance(rval[1][0], dict)  # type: ignore
        )
        if returns_info:
            obs_reset, info = rval
            if self.preprocess_fn:
                processed_data = self.preprocess_fn(
                    obs=obs_reset, info=info, env_id=global_ids
                )
                obs_reset = processed_data.get("obs", obs_reset)
                info = processed_data.get("info", info)
            self.data.info[local_ids] = info
        else:
            obs_reset = rval
            if self.preprocess_fn:
                obs_reset = self.preprocess_fn(obs=obs_reset, env_id=global_ids
                                               ).get("obs", obs_reset)
        self.data.obs_next[local_ids] = obs_reset

    def _reset_state(self, id: Union[int, List[int]]) -> None:
        """Reset the hidden state: self.data.state[id]."""
        if hasattr(self.data.policy, "hidden_state"):
            state = self.data.policy.hidden_state  # it is a reference
            if isinstance(state, torch.Tensor):
                state[id].zero_()
            elif isinstance(state, np.ndarray):
                state[id] = None if state.dtype == object else 0
            elif isinstance(state, Batch):
                state.empty_(id)

    def reset_env(self, gym_reset_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """Reset all of the environments."""
        gym_reset_kwargs = gym_reset_kwargs if gym_reset_kwargs else {}
        rval = self.env.reset(**gym_reset_kwargs)
        returns_info = isinstance(rval, (tuple, list)) and len(rval) == 2 and (
            isinstance(rval[1], dict) or isinstance(rval[1][0], dict)  # type: ignore
        )
        if returns_info:
            obs, info = rval
            if self.preprocess_fn:
                processed_data = self.preprocess_fn(
                    obs=obs, info=info, env_id=np.arange(self.env_num)
                )
                obs = processed_data.get("obs", obs)
                info = processed_data.get("info", info)
                self.data.info = info
        else:
            obs = rval
            if self.preprocess_fn:
                obs = self.preprocess_fn(obs=obs, env_id=np.arange(self.env_num)).get("obs", obs)
        self.data.obs = obs

    def reset_buffer(self, keep_statistics: bool = False) -> None:
        """Reset the data buffer."""
        self.buffer.reset(keep_statistics=keep_statistics)
