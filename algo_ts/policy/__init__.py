"""Policy package."""
# isort:skip_file

from algo_ts.policy.base import BasePolicy
from algo_ts.policy.random import RandomPolicy
from algo_ts.policy.modelfree.dqn import DQNPolicy
from algo_ts.policy.modelfree.bdq import BranchingDQNPolicy
from algo_ts.policy.modelfree.c51 import C51Policy
from algo_ts.policy.modelfree.rainbow import RainbowPolicy
from algo_ts.policy.modelfree.qrdqn import QRDQNPolicy
from algo_ts.policy.modelfree.iqn import IQNPolicy
from algo_ts.policy.modelfree.fqf import FQFPolicy
from algo_ts.policy.modelfree.pg import PGPolicy
from algo_ts.policy.modelfree.a2c import A2CPolicy
from algo_ts.policy.modelfree.npg import NPGPolicy
from algo_ts.policy.modelfree.ddpg import DDPGPolicy
from algo_ts.policy.modelfree.ppo import PPOPolicy
from algo_ts.policy.modelfree.trpo import TRPOPolicy
from algo_ts.policy.modelfree.td3 import TD3Policy
from algo_ts.policy.modelfree.sac import SACPolicy
from algo_ts.policy.modelfree.redq import REDQPolicy
from algo_ts.policy.modelfree.discrete_sac import DiscreteSACPolicy
from algo_ts.policy.imitation.base import ImitationPolicy
from algo_ts.policy.imitation.bcq import BCQPolicy
from algo_ts.policy.imitation.cql import CQLPolicy
from algo_ts.policy.imitation.td3_bc import TD3BCPolicy
from algo_ts.policy.imitation.discrete_bcq import DiscreteBCQPolicy
from algo_ts.policy.imitation.discrete_cql import DiscreteCQLPolicy
from algo_ts.policy.imitation.discrete_crr import DiscreteCRRPolicy
from algo_ts.policy.imitation.gail import GAILPolicy
from algo_ts.policy.modelbased.psrl import PSRLPolicy
from algo_ts.policy.modelbased.icm import ICMPolicy
from algo_ts.policy.multiagent.mapolicy import MultiAgentPolicyManager

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
