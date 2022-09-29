"""Env package."""

from algo.env.gym_wrappers import ContinuousToDiscrete, MultiDiscreteToDiscrete
from algo.env.venv_wrappers import VectorEnvNormObs, VectorEnvWrapper
from algo.env.venvs import (
    BaseVectorEnv,
    DummyVectorEnv,
    RayVectorEnv,
    ShmemVectorEnv,
    SubprocVectorEnv,
)

try:
    from algo.env.pettingzoo_env import PettingZooEnv
except ImportError:
    pass

__all__ = [
    "BaseVectorEnv",
    "DummyVectorEnv",
    "SubprocVectorEnv",
    "ShmemVectorEnv",
    "RayVectorEnv",
    "VectorEnvWrapper",
    "VectorEnvNormObs",
    "PettingZooEnv",
    "ContinuousToDiscrete",
    "MultiDiscreteToDiscrete",
]
