import os
import sys

sys.path.append(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + '/')
import gym
import numpy as np
import torch
import pickle
from geek.env.matrix_env import Scenarios
from torch.utils.tensorboard import SummaryWriter

from ts_inherit.logger import MyLogger

from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from tianshou.data import PrioritizedVectorReplayBuffer, VectorReplayBuffer

# from tianshou.utils.net.common import Net
# from tianshou.utils.net.discrete import Actor, Critic

from ts_inherit.sac_actor import Actor
from ts_inherit.sac_critic import Critic
from ts_inherit.sac_pre_net import PreNetworks
from ts_inherit.sac_policy import SacPolicy
from ts_inherit.sac_collector import SacCollector
from ts_inherit.sac_trainer import sac_policy_trainer
from utils.processors import set_seed, Processor


def make_train_env(cfgs, render_id=None):
    if render_id is not None:
        env = gym.make(cfgs.task, scenarios=Scenarios.TRAINING)
        # env = gym.make(cfgs.task, scenarios=Scenarios.TRAINING, render_id=str(render_id))
    else:
        env = gym.make(cfgs.task, scenarios=Scenarios.TRAINING)
    if not hasattr(env, 'action_space'):
        setattr(env, 'action_space', cfgs.action_space)
    return env


def train(cfgs):
    train_envs = SubprocVectorEnv([lambda i=_i: make_train_env(cfgs, i) for _i in range(cfgs.training_num)])

    set_seed(seed=cfgs.seed)

    # model
    actor_pre = PreNetworks(cfgs=cfgs)
    actor = Actor(actor_pre, cfgs.action_shape, softmax_output=False, device=cfgs.device).to(cfgs.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=cfgs.actor_lr)

    c1_pre = PreNetworks(cfgs=cfgs)
    critic1 = Critic(c1_pre, last_size=cfgs.action_shape, device=cfgs.device).to(cfgs.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=cfgs.critic_lr)

    c2_pre = PreNetworks(cfgs=cfgs)
    critic2 = Critic(c2_pre, last_size=cfgs.action_shape, device=cfgs.device).to(cfgs.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=cfgs.critic_lr)

    if cfgs.auto_alpha:
        target_entropy = 0.98 * np.log(np.prod(cfgs.action_shape))
        log_alpha = torch.zeros(1, requires_grad=True, device=cfgs.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=cfgs.alpha_lr)
        alpha = tuple([target_entropy, log_alpha, alpha_optim])
    else:
        alpha = cfgs.alpha

    policy = SacPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=cfgs.tau,
        gamma=cfgs.gamma,
        alpha=alpha,
        estimation_step=cfgs.n_step,
        reward_normalization=cfgs.rew_norm
    )

    policy.make_action_library(cfgs)

    # log
    log_path = os.path.join(cfgs.logdir, cfgs.task, "discrete_sac")
    writer = SummaryWriter(log_path)
    logger = MyLogger(writer, save_interval=cfgs.save_interval)

    tianchi_logger = logger.logger
    tianchi_logger.info('device: ' + str(cfgs.device))

    train_processor = Processor(
        cfgs,
        tianchi_logger,
        models=[actor_pre, c1_pre, c2_pre],
        n_env=cfgs.training_num,
        update_norm=True
    )

    # collector
    train_collector = SacCollector(
        policy,
        train_envs,
        PrioritizedVectorReplayBuffer(
            total_size=cfgs.per.buffer_size,
            buffer_num=len(train_envs),
            alpha=cfgs.per.alpha,
            beta=cfgs.per.beta,
            weight_norm=True,
        ),
        preprocess_fn=train_processor.preprocess_fn,
        exploration_noise=True
    )
    train_collector.set_logger(tianchi_logger, type='train')

    # policy logger
    policy.set_logger(tianchi_logger)

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        torch.save(policy.state_dict(), ckpt_path)
        buffer_path = os.path.join(log_path, "train_buffer.pkl")
        pickle.dump(train_collector.buffer, open(buffer_path, "wb"))
        return ckpt_path

    if cfgs.resume:
        # load from existing checkpoint
        logger.logger.info(f"Loading agent under {log_path}")
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        if os.path.exists(ckpt_path):
            policy.load_state_dict(torch.load(ckpt_path, map_location=cfgs.device))
            logger.logger.info("Successfully restore policy and optim.")
        else:
            logger.logger.info("Fail to restore policy and optim.")

        buffer_path = os.path.join(log_path, "train_buffer.pkl")
        if os.path.exists(buffer_path):
            train_collector.buffer = pickle.load(open(buffer_path, "rb"))
            logger.logger.info("Successfully restore buffer.")
        else:
            logger.logger.info("Fail to restore buffer.")

    if len(train_collector.buffer) == 0:
        warm_up = int(cfgs.per.buffer_size / 5)
        train_collector.collect(n_step=warm_up, random=True)
        tianchi_logger.info(f"------------Warmup Collect {warm_up} transitions------------")
    else:
        tianchi_logger.info(f"------------Buffer has {len(train_collector.buffer)} transitions------------")

    # trainer
    result = sac_policy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=None,  # 测试相关
        max_epoch=cfgs.epoch,
        step_per_epoch=cfgs.step_per_epoch,
        step_per_collect=cfgs.step_per_collect,
        episode_per_test=0,  # 每次test多少个episode
        batch_size=cfgs.batch_size,
        update_per_step=cfgs.update_per_step,
        train_fn=None,
        test_fn=None,
        stop_fn=None,
        save_best_fn=None,
        logger=logger,
        show_progress=False,
        resume_from_log=cfgs.resume,
        save_checkpoint_fn=save_checkpoint_fn,
    )


if __name__ == '__main__':
    torch.set_num_threads(1)

    from sac.config import cfg

    cfg.action_space = gym.spaces.Discrete(cfg.action_per_dim[0] * cfg.action_per_dim[1])
    cfg.action_shape = cfg.action_space.n

    train(cfgs=cfg)
