"""Policy package."""
# isort:skip_file

from algo.policy.base import BasePolicy
from algo.policy.random import RandomPolicy
from algo.policy.modelfree.dqn import DQNPolicy
from algo.policy.modelfree.bdq import BranchingDQNPolicy
from algo.policy.modelfree.c51 import C51Policy
from algo.policy.modelfree.rainbow import RainbowPolicy
from algo.policy.modelfree.qrdqn import QRDQNPolicy
from algo.policy.modelfree.iqn import IQNPolicy
from algo.policy.modelfree.fqf import FQFPolicy
from algo.policy.modelfree.pg import PGPolicy
from algo.policy.modelfree.a2c import A2CPolicy
from algo.policy.modelfree.npg import NPGPolicy
from algo.policy.modelfree.ddpg import DDPGPolicy
from algo.policy.modelfree.ppo import PPOPolicy
from algo.policy.modelfree.trpo import TRPOPolicy
from algo.policy.modelfree.td3 import TD3Policy
from algo.policy.modelfree.sac import SACPolicy
from algo.policy.modelfree.redq import REDQPolicy
from algo.policy.modelfree.discrete_sac import DiscreteSACPolicy
from algo.policy.imitation.base import ImitationPolicy
from algo.policy.imitation.bcq import BCQPolicy
from algo.policy.imitation.cql import CQLPolicy
from algo.policy.imitation.td3_bc import TD3BCPolicy
from algo.policy.imitation.discrete_bcq import DiscreteBCQPolicy
from algo.policy.imitation.discrete_cql import DiscreteCQLPolicy
from algo.policy.imitation.discrete_crr import DiscreteCRRPolicy
from algo.policy.imitation.gail import GAILPolicy
from algo.policy.modelbased.psrl import PSRLPolicy
from algo.policy.modelbased.icm import ICMPolicy
from algo.policy.multiagent.mapolicy import MultiAgentPolicyManager

__all__ = [
    "BasePolicy",
    "RandomPolicy",
    "DQNPolicy",
    "BranchingDQNPolicy",
    "C51Policy",
    "RainbowPolicy",
    "QRDQNPolicy",
    "IQNPolicy",
    "FQFPolicy",
    "PGPolicy",
    "A2CPolicy",
    "NPGPolicy",
    "DDPGPolicy",
    "PPOPolicy",
    "TRPOPolicy",
    "TD3Policy",
    "SACPolicy",
    "REDQPolicy",
    "DiscreteSACPolicy",
    "ImitationPolicy",
    "BCQPolicy",
    "CQLPolicy",
    "TD3BCPolicy",
    "DiscreteBCQPolicy",
    "DiscreteCQLPolicy",
    "DiscreteCRRPolicy",
    "GAILPolicy",
    "PSRLPolicy",
    "ICMPolicy",
    "MultiAgentPolicyManager",
]
