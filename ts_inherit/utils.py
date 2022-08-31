from typing import Any, Dict


def test_episode(policy, collector, test_fn, epoch, n_episode, logger=None, global_step=None, reward_metric=None) -> Dict[str, Any]:
    """A simple wrapper of testing policy in collector."""
    collector.reset_env()
    collector.reset_buffer()
    policy.eval()
    if test_fn:
        test_fn(epoch, global_step)
    result = collector.collect(n_episode=n_episode)
    if reward_metric:
        rew = reward_metric(result["rews"])
        result.update(rews=rew, rew=rew.mean(), rew_std=rew.std())
    if logger and global_step is not None:
        logger.log_test_data(result, global_step)
    return result
