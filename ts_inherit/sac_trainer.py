from typing import Any, Dict, Tuple, Union

import tqdm
import time
from math import inf

from tianshou.trainer.utils import gather_info
from tianshou.utils import DummyTqdm, tqdm_config
from tianshou.trainer import OffpolicyTrainer
from ts_inherit.utils import test_episode


class SacTrainer(OffpolicyTrainer):

    def policy_update_fn(self, data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """ Perform off-policy updates."""
        assert self.train_collector is not None
        start_time = time.time()
        update_num = round(self.update_per_step * result["n/st"])
        loss = dict(
            actor=0,
            critic1=0,
            critic2=0
        )
        for _ in range(update_num):  # update_per_step(0.125) * step_per_collect(700)
            self.gradient_step += 1
            losses = self.policy.update(self.batch_size, self.train_collector.buffer)
            loss['actor'] += losses["loss/actor"]
            loss['critic1'] += losses["loss/critic1"]
            loss['critic2'] += losses["loss/critic2"]
            self.log_update_data(data, losses)

        a_loss = round(loss['actor']/update_num, 2)
        c1_loss = round(loss['critic1']/update_num, 2)
        c2_loss = round(loss['critic2']/update_num, 2)

        cost_time = round(time.time() - start_time, 2)

        self.logger.logger.info(f'After {update_num} learning, loss_a: {a_loss}, loss_c1: {c1_loss}, loss_c2: {c2_loss}, Cost:{cost_time}s')

    def reset(self) -> None:
        """ Initialize or reset the instance to yield a new iterator from zero. """
        self.is_run = False
        self.env_step = 0  # 收集了多少步的数据

        if self.resume_from_log:
            self.start_epoch, self.env_step, self.gradient_step = self.logger.restore_data()
            self.best_epoch, self.best_reward, self.best_reward_std = self.logger.restore_best()
            if self.best_reward > -inf:
                self.logger.logger.info(f'Resume: Epoch #{self.best_epoch}: best_reward: {self.best_reward:.6f} ± {self.best_reward_std:.6f}.')

        self.last_rew, self.last_len = 0.0, 0
        self.start_time = time.time()

        if self.train_collector is not None:
            self.train_collector.reset_stat()

            if self.train_collector.policy != self.policy:
                self.test_in_train = False
            elif self.test_collector is None:
                self.test_in_train = False

        self.epoch = self.start_epoch
        self.stop_fn_flag = False
        self.iter_num = 0

    def __iter__(self):  # type: ignore
        self.reset()
        return self

    def __next__(self) -> Union[None, Tuple[int, Dict[str, Any], Dict[str, Any]]]:
        """ Perform one epoch (both train and eval). """
        self.epoch += 1
        self.iter_num += 1

        if self.iter_num > 1:

            # iterator exhaustion check
            if self.epoch > self.max_epoch:
                raise StopIteration

            # exit flag 1, when stop_fn succeeds in train_step or test_step
            if self.stop_fn_flag:
                raise StopIteration

        # set policy in train mode
        self.policy.train()

        epoch_stat: Dict[str, Any] = dict()

        if self.show_progress:
            progress = tqdm.tqdm
        else:
            progress = DummyTqdm

        # perform n step_per_epoch
        self.logger.logger.info(f'---------------------Epoch:{self.epoch}-Training---------------------')
        with progress(total=self.step_per_epoch, desc=f"Epoch #{self.epoch}", **tqdm_config) as t:
            while t.n < t.total and not self.stop_fn_flag:
                data: Dict[str, Any] = dict()
                result: Dict[str, Any] = dict()
                if self.train_collector is not None:
                    data, result, self.stop_fn_flag = self.train_step()
                    t.update(result["n/st"])
                    if self.stop_fn_flag:
                        t.set_postfix(**data)
                        break
                else:
                    assert self.buffer, "No train_collector or buffer specified"
                    result["n/ep"] = len(self.buffer)
                    result["n/st"] = int(self.gradient_step)
                    t.update()

                self.policy_update_fn(data, result)
                t.set_postfix(**data)

            if t.n <= t.total and not self.stop_fn_flag:
                t.update()

        # for offline RL
        if self.train_collector is None:
            self.env_step = self.gradient_step * self.batch_size

        if not self.stop_fn_flag:
            self.logger.save_data(self.epoch, self.env_step, self.gradient_step, self.save_checkpoint_fn)
            # test
            if self.test_collector is not None:
                self.logger.logger.info(f'---------------------Epoch:{self.epoch}-Testing---------------------')
                test_stat, self.stop_fn_flag = self.test_step()
                if not self.is_run:
                    epoch_stat.update(test_stat)

        if not self.is_run:
            epoch_stat.update({k: v.get() for k, v in self.stat.items()})
            epoch_stat["gradient_step"] = self.gradient_step
            epoch_stat.update(
                {
                    "env_step": self.env_step,
                    "rew": self.last_rew,
                    "len": int(self.last_len),
                    "n/ep": int(result["n/ep"]),
                    "n/st": int(result["n/st"]),
                }
            )
            info = gather_info(
                self.start_time,
                self.train_collector,
                self.test_collector,
                self.best_reward,
                self.best_reward_std
            )
            return self.epoch, epoch_stat, info
        else:
            return None

    def test_step(self) -> Tuple[Dict[str, Any], bool]:
        """Perform one testing step."""
        assert self.episode_per_test is not None
        assert self.test_collector is not None
        stop_fn_flag = False
        test_result = test_episode(
            policy=self.policy,
            collector=self.test_collector,
            test_fn=self.test_fn,
            epoch=self.epoch,
            n_episode=self.episode_per_test,
            logger=self.logger,
            global_step=self.env_step,
            reward_metric=self.reward_metric
        )
        rew, rew_std = test_result["rew"], test_result["rew_std"]
        if self.best_epoch < 0 or self.best_reward < rew:
            self.best_epoch = self.epoch
            self.best_reward = float(rew)
            self.best_reward_std = rew_std
            self.logger.save_best(self.best_epoch, self.best_reward, self.best_reward_std)
            if self.save_best_fn:
                self.save_best_fn(self.policy)
        if self.verbose:
            info = f"Epoch #{self.epoch}: test_reward: {rew:.6f} ± {rew_std:.6f}, best_reward: {self.best_reward:.6f} ± {self.best_reward_std:.6f} in #{self.best_epoch}"
            self.logger.logger.info(info)
            print(f"Epoch #{self.epoch}: test_reward: {rew:.6f} ± {rew_std:.6f}, best_reward: {self.best_reward:.6f} ± {self.best_reward_std:.6f} in #{self.best_epoch}")
        if not self.is_run:
            test_stat = {"test_reward": rew, "test_reward_std": rew_std, "best_reward": self.best_reward, "best_reward_std": self.best_reward_std, "best_epoch": self.best_epoch}
        else:
            test_stat = {}
        if self.stop_fn and self.stop_fn(self.best_reward):
            stop_fn_flag = True

        return test_stat, stop_fn_flag

    def train_step(self) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        """Perform one training step."""
        assert self.episode_per_test is not None
        assert self.train_collector is not None
        stop_fn_flag = False
        if self.train_fn:
            self.train_fn(self.epoch, self.env_step)

        result = self.train_collector.collect(n_step=self.step_per_collect, n_episode=self.episode_per_collect)

        if result["n/ep"] > 0 and self.reward_metric:
            rew = self.reward_metric(result["rews"])
            result.update(rews=rew, rew=rew.mean(), rew_std=rew.std())
        self.env_step += int(result["n/st"])
        self.logger.log_train_data(result, self.env_step)
        self.last_rew = result["rew"] if result["n/ep"] > 0 else self.last_rew
        self.last_len = result["len"] if result["n/ep"] > 0 else self.last_len
        data = {
            "env_step": str(self.env_step),
            "rew": f"{self.last_rew:.2f}",
            "len": str(int(self.last_len)),
            "n/ep": str(int(result["n/ep"])),
            "n/st": str(int(result["n/st"])),
        }
        if result["n/ep"] > 0:
            if self.test_in_train and self.stop_fn and self.stop_fn(result["rew"]):
                assert self.test_collector is not None
                test_result = test_episode(self.policy, self.test_collector, self.test_fn, self.epoch, self.episode_per_test, self.logger, self.env_step)
                if self.stop_fn(test_result["rew"]):
                    stop_fn_flag = True
                    self.best_reward = test_result["rew"]
                    self.best_reward_std = test_result["rew_std"]
                else:
                    self.policy.train()

        return data, result, stop_fn_flag


def sac_policy_trainer(*args, **kwargs) -> Dict[str, Union[float, str]]:  # type: ignore
    """Wrapper for OffPolicyTrainer run method.

    It is identical to ``OffpolicyTrainer(...).run()``.

    :return: See :func:`~tianshou.trainer.gather_info`.
    """
    return SacTrainer(*args, **kwargs).run()
