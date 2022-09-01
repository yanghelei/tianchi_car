import torch
import numpy as np
from copy import deepcopy
from numbers import Number
from tianshou.data.batch import Batch, _parse_value
from typing import Any, Dict, Optional, Union, no_type_check


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


@no_type_check
def to_numpy(x: Any) -> Union[Batch, np.ndarray]:
    """Return an object without torch.Tensor."""
    if isinstance(x, torch.Tensor):  # most often case
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):  # second often case
        return x
    elif isinstance(x, (np.number, np.bool_, Number)):
        return np.asanyarray(x)
    elif x is None:
        return np.array(None, dtype=object)
    elif isinstance(x, (dict, Batch)):
        x = Batch(x) if isinstance(x, dict) else deepcopy(x)
        x.to_numpy()
        return x
    elif isinstance(x, (list, tuple)):
        return to_numpy(_parse_value(x))
    else:  # fallback
        return np.asanyarray(x)


@no_type_check
def to_torch(
        x: Any,
        dtype: Optional[torch.dtype] = None,
        device: Union[str, int, torch.device] = "cpu",
) -> Union[Batch, torch.Tensor]:
    """Return an object without np.ndarray."""
    if isinstance(x, np.ndarray) and issubclass(
            x.dtype.type, (np.bool_, np.number)
    ):  # most often case
        x = torch.from_numpy(x).to(device)
        if dtype is not None:
            x = x.type(dtype)
        return x
    elif isinstance(x, torch.Tensor):  # second often case
        if dtype is not None:
            x = x.type(dtype)
        return x.to(device)
    elif isinstance(x, (np.number, np.bool_, Number)):
        return to_torch(np.asanyarray(x), dtype, device)
    elif isinstance(x, (dict, Batch)):
        x = Batch(x, copy=True) if isinstance(x, dict) else deepcopy(x)
        x.to_torch(dtype, device)
        return x
    elif isinstance(x, (list, tuple)):
        return to_torch(_parse_value(x), dtype, device)
    else:  # fallback
        raise TypeError(f"object {x} cannot be converted to torch.")


@no_type_check
def to_torch_as(x: Any, y: torch.Tensor) -> Union[Batch, torch.Tensor]:
    """Return an object without np.ndarray.

    Same as ``to_torch(x, dtype=y.dtype, device=y.device)``.
    """
    assert isinstance(y, torch.Tensor)
    return to_torch(x, dtype=y.dtype, device=y.device)
