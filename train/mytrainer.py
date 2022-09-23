# -*- coding : utf-8 -*-
# @Time     : 2022/9/20 下午1:24
# @Author   : gepeng
# @ FileName: mytrainer.py
# @ Software: Pycharm

from algo.trainer.base import BaseTrainer
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import tqdm

from algo.data import Collector
from algo.policy import BasePolicy
from algo.trainer.utils import gather_info, test_episode
from algo.utils import (
    BaseLogger,
    DummyTqdm,
    LazyLogger,
    tqdm_config,
)


class MyTrainer(BaseTrainer):

    def __init__(
            self,
            policy: BasePolicy,
            train_collector: Collector,
            test_collector: Optional[Collector],
            max_epoch: int,
            step_per_epoch: int,
            step_per_collect: int,
            episode_per_test: int,
            batch_size: int,
            update_per_step: Union[int, float] = 1,
            train_fn: Optional[Callable[[int, int], None]] = None,
            test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
            stop_fn: Optional[Callable[[float], bool]] = None,
            save_best_fn: Optional[Callable[[BasePolicy], None]] = None,
            save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None,
            resume_from_log: bool = False,
            reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            logger: BaseLogger = LazyLogger(),
            verbose: bool = True,
            show_progress: bool = True,
            test_in_train: bool = True,
            **kwargs: Any,
    ):
        super().__init__(
            learning_type="offpolicy",
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=max_epoch,
            step_per_epoch=step_per_epoch,
            step_per_collect=step_per_collect,
            episode_per_test=episode_per_test,
            batch_size=batch_size,
            update_per_step=update_per_step,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            resume_from_log=resume_from_log,
            reward_metric=reward_metric,
            logger=logger,
            verbose=verbose,
            show_progress=show_progress,
            test_in_train=test_in_train,
            **kwargs,
        )

    def reset(self) -> None:
        """Initialize or reset the instance to yield a new iterator from zero."""
        self.is_run = False
        self.env_step = 0

        self.last_rew, self.last_len = 0.0, 0
        self.start_time = time.time()
        if self.train_collector is not None:
            self.train_collector.reset_stat()

        self.epoch = self.start_epoch
        self.iter_num = 0

    def __iter__(self):  # type: ignore
        self.reset()
        return self

    def __next__(self) -> Union[None, Tuple[int, Dict[str, Any], Dict[str, Any]]]:
        """Perform one epoch (both train and eval)."""
        self.epoch += 1
        self.iter_num += 1

        if self.iter_num > 1:

            # iterator exhaustion check
            if self.epoch > self.max_epoch:
                raise StopIteration

        # set policy in train mode
        self.policy.train()

        epoch_stat: Dict[str, Any] = dict()

        if self.show_progress:
            progress = tqdm.tqdm
        else:
            progress = DummyTqdm

        # perform n step_per_epoch
        with progress(
                total=self.step_per_epoch, desc=f"Epoch #{self.epoch}", **tqdm_config
        ) as t:
            while t.n < t.total:
                data, result = self.train_step()
                t.update(result["n/st"])

                self.policy_update_fn(data, result)
                t.set_postfix(**data)

            # 处理末尾数据更新
            if t.n <= t.total:
                t.update()

        # test
        # 由于不能利用新建的环境测试，使用训练收集器进行测试以保存模型
        if self.train_collector is not None:
            test_stat = self.test_step()
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
                self.start_time, self.train_collector, self.test_collector,
                self.best_reward, self.best_reward_std
            )
            return self.epoch, epoch_stat, info
        else:
            return None

    def test_step(self) -> Tuple[Dict[str, Any], bool]:
        """Perform one testing step."""
        # 使用训练收集器进行测试
        assert self.train_collector is not None
        assert self.episode_per_test == self.train_collector.env_num
        test_result = test_episode(
            self.policy, self.train_collector, self.test_fn, self.epoch,
            self.episode_per_test, self.logger, self.env_step, self.reward_metric
        )
        rew, rew_std = test_result["rew"], test_result["rew_std"]
        if self.best_epoch < 0 or self.best_reward < rew:
            self.best_epoch = self.epoch
            self.best_reward = float(rew)
            self.best_reward_std = rew_std
            if self.save_best_fn:
                self.save_best_fn(self.policy)
        if self.verbose:
            print(
                f"Epoch #{self.epoch}: test_reward: {rew:.6f} ± {rew_std:.6f},"
                f" best_reward: {self.best_reward:.6f} ± "
                f"{self.best_reward_std:.6f} in #{self.best_epoch}"
            )

        if not self.is_run:
            test_stat = {
                "test_reward": rew,
                "test_reward_std": rew_std,
                "best_reward": self.best_reward,
                "best_reward_std": self.best_reward_std,
                "best_epoch": self.best_epoch
            }
        else:
            test_stat = {}

        return test_stat

    def train_step(self) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        """Perform one training step."""
        assert self.episode_per_test is not None
        assert self.train_collector is not None
        if self.train_fn:
            self.train_fn(self.epoch, self.env_step)
        result = self.train_collector.collect(
            n_step=self.step_per_collect, n_episode=self.episode_per_collect
        )
        if result["n/ep"] > 0 and self.reward_metric:
            rew = self.reward_metric(result["rews"])
            result.update(rews=rew, rew=rew.mean(), rew_std=rew.std())
        self.env_step += int(result["n/st"])

        self.last_rew = result["rew"] if result["n/ep"] > 0 else self.last_rew
        self.last_len = result["len"] if result["n/ep"] > 0 else self.last_len
        data = {
            "env_step": str(self.env_step),
            "rew": f"{self.last_rew:.2f}",
            "len": str(int(self.last_len)),
            "n/ep": str(int(result["n/ep"])),
            "n/st": str(int(result["n/st"])),
        }

        return data, result

    def log_update_data(self, data: Dict[str, Any], losses: Dict[str, Any]) -> None:
        """Log losses to current logger."""
        for k in losses.keys():
            self.stat[k].add(losses[k])
            losses[k] = self.stat[k].get()
            data[k] = f"{losses[k]:.3f}"
        self.logger.log_update_data(losses, self.gradient_step)

    def policy_update_fn(self, data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Perform off-policy updates."""
        assert self.train_collector is not None
        for _ in range(round(self.update_per_step * result["n/st"])):
            self.gradient_step += 1
            losses = self.policy.update(self.batch_size, self.train_collector.buffer)
            self.log_update_data(data, losses)


def my_trainer(*args, **kwargs) -> Dict[str, Union[float, str]]:  # type: ignore
    """Wrapper for OffPolicyTrainer run method.

    It is identical to ``OffpolicyTrainer(...).run()``.

    :return: See :func:`~algo_torch.trainer.gather_info`.
    """
    return MyTrainer(*args, **kwargs).run()


my_trainer_iter = MyTrainer