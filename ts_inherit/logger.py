from typing import Callable, Optional, Tuple

from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from geek.env.logger import Logger
from math import inf


class MyLogger(TensorboardLogger):

    def __init__(
            self,
            writer: SummaryWriter,
            train_interval: int = 1000,
            test_interval: int = 1,
            update_interval: int = 1000,
            save_interval: int = 1,
            write_flush: bool = True,
    ):
        self.logger = Logger.get_logger(__name__)

        super().__init__(writer, train_interval, test_interval, update_interval, save_interval, write_flush)

    def save_best(self, best_epoch, best_reward, best_reward_std):
        self.write("best/epoch", best_epoch, {"best/epoch": best_epoch})
        self.write("best/reward", best_reward, {"best/reward": best_reward})
        self.write("best/reward_std", best_reward_std, {"best/reward_std": best_reward_std})

    def restore_best(self):
        ea = event_accumulator.EventAccumulator(self.writer.log_dir)
        ea.Reload()

        try:  # epoch / gradient_step
            best_epoch = ea.scalars.Items("best/epoch")[-1].step
            best_reward = ea.scalars.Items("best/reward")[-1].step
            best_reward_std = ea.scalars.Items("best/reward_std")[-1].step
        except KeyError:
            best_epoch, best_reward, best_reward_std = -1, -inf, 0

        return best_epoch, best_reward, best_reward_std

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None,
    ) -> None:
        if save_checkpoint_fn and epoch - self.last_save_step >= self.save_interval:
            self.last_save_step = epoch
            save_checkpoint_fn(epoch, env_step, gradient_step)
            self.logger.info(f"----------------Saved Check Point - Epoch: {epoch} - Env step: {env_step}----------------")
            self.write("save/epoch", epoch, {"save/epoch": epoch})
            self.write("save/env_step", env_step, {"save/env_step": env_step})
            self.write(
                "save/gradient_step", gradient_step,
                {"save/gradient_step": gradient_step}
            )

    def restore_data(self) -> Tuple[int, int, int]:
        ea = event_accumulator.EventAccumulator(self.writer.log_dir)
        ea.Reload()

        try:  # epoch / gradient_step
            epoch = ea.scalars.Items("save/epoch")[-1].step
            self.last_save_step = self.last_log_test_step = epoch
            gradient_step = ea.scalars.Items("save/gradient_step")[-1].step
            self.last_log_update_step = gradient_step
        except KeyError:
            epoch, gradient_step = 0, 0
        try:  # offline trainer doesn't have env_step
            env_step = ea.scalars.Items("save/env_step")[-1].step
            self.last_log_train_step = env_step
        except KeyError:
            env_step = 0

        return epoch, env_step, gradient_step

    def log_train_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during training.

        :param collect_result: a dict containing information of data collected in
            training stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        if collect_result["n/ep"] > 0:
            if step - self.last_log_train_step >= self.train_interval:
                log_data = {
                    "train/episode": collect_result["n/ep"],
                    "train/reward": collect_result["rew"],
                    "train/length": collect_result["len"],
                }
                self.write("train/env_step", step, log_data)
                self.last_log_train_step = step

    def log_test_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during evaluating.

        :param collect_result: a dict containing information of data collected in
            evaluating stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        assert collect_result["n/ep"] > 0
        if step - self.last_log_test_step >= self.test_interval:
            log_data = {
                "test/env_step": step,
                "test/reward": collect_result["rew"],
                "test/length": collect_result["len"],
                "test/reward_std": collect_result["rew_std"],
                "test/length_std": collect_result["len_std"],
            }
            self.write("test/env_step", step, log_data)
            self.last_log_test_step = step
