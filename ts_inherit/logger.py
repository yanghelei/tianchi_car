from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger

from geek.env.logger import Logger


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
                # self.logger.info(
                #     'step:' + str(step) + '\t' +
                #     'reward:' + str(collect_result["rew"]) + '\t' +
                #     'length:' + str(collect_result["len"]) + '\t'
                # )
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
