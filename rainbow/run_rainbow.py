import os
import sys

sys.path.append(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + '/')
import gym
import torch
import pprint
import pickle
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import PrioritizedVectorReplayBuffer, VectorReplayBuffer, Collector
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger

from tianshou.utils.net.discrete import NoisyLinear, Actor

from geek.env.matrix_env import Scenarios
from rainbow.utils import set_seed, preprocess_fn
# from rainbow.collector import MyCollector
from rainbow.policy import MyRainbow
from rainbow.networks import MyActor


def make_train_env(cfgs):
    env = gym.make(cfgs.task, scenarios=Scenarios.TRAINING)
    if not hasattr(env, 'action_space'):
        setattr(env, 'action_space', cfgs.action_space)
    return env


def make_test_env(cfgs):
    env = gym.make(cfgs.task, scenarios=Scenarios.INFERENCE)
    if not hasattr(env, 'action_space'):
        setattr(env, 'action_space', cfgs.action_space)
    return env


def train(cfgs):
    train_envs = SubprocVectorEnv([lambda: make_train_env(cfgs) for _ in range(cfgs.training_num)])

    set_seed(seed=cfgs.seed)

    def noisy_linear(x, y):
        return NoisyLinear(x, y, cfgs.noisy_std)

    net = MyActor(
        cfgs.state_shape,
        cfgs.action_shape,
        hidden_sizes=cfgs.hidden_sizes,
        device=cfgs.device,
        softmax=True,
        num_atoms=cfgs.num_atoms,
        dueling_param=({"linear_layer": noisy_linear}, {"linear_layer": noisy_linear}),
    )

    optim = torch.optim.Adam(net.parameters(), lr=cfgs.lr)

    policy = MyRainbow(
        model=net,
        optim=optim,
        discount_factor=cfgs.gamma,
        num_atoms=cfgs.num_atoms,
        v_min=cfgs.v_min,
        v_max=cfgs.v_max,
        estimation_step=cfgs.n_step,
        target_update_freq=cfgs.target_update_freq,
    ).to(cfgs.device)

    policy.make_action_library(cfgs)

    # buffer
    if cfgs.prioritized_replay:
        buf = PrioritizedVectorReplayBuffer(total_size=cfgs.buffer_size, buffer_num=len(train_envs), alpha=cfgs.alpha, beta=cfgs.beta, weight_norm=True, )
    else:
        buf = VectorReplayBuffer(cfgs.buffer_size, buffer_num=len(train_envs))

    # collector
    train_collector = Collector(policy, train_envs, buf, preprocess_fn=preprocess_fn, exploration_noise=True)

    train_collector.collect(n_step=cfgs.batch_size * cfgs.training_num)

    # log
    log_path = os.path.join(cfgs.logdir, cfgs.task, "rainbow")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer, save_interval=cfgs.save_interval)

    def save_best_fn(policy):
        # 保存最优模型
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    # def stop_fn(mean_rewards):
    #     # 终止条件
    #     return mean_rewards >= cfgs.reward_threshold

    def train_fn(epoch, env_step):
        # 在每次训练前执行的操作
        # eps annealing, just a demo
        if env_step <= 10000:
            policy.set_eps(cfgs.eps_train)
        elif env_step <= 50000:
            eps = cfgs.eps_train - (env_step - 10000) / 40000 * (0.9 * cfgs.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * cfgs.eps_train)
        # beta annealing, just a demo
        if cfgs.prioritized_replay:
            if env_step <= 10000:
                beta = cfgs.beta
            elif env_step <= 50000:
                beta = cfgs.beta - (env_step - 10000) / 40000 * (cfgs.beta - cfgs.beta_final)
            else:
                beta = cfgs.beta_final
            buf.set_beta(beta)

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        # Example: saving by epoch num
        # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
        torch.save({"model": policy.state_dict(), "optim": optim.state_dict(), }, ckpt_path)
        buffer_path = os.path.join(log_path, "train_buffer.pkl")
        pickle.dump(train_collector.buffer, open(buffer_path, "wb"))
        return ckpt_path

    if cfgs.resume:
        # load from existing checkpoint
        print(f"Loading agent under {log_path}")
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=cfgs.device)
            policy.load_state_dict(checkpoint['model'])
            policy.optim.load_state_dict(checkpoint['optim'])
            print("Successfully restore policy and optim.")
        else:
            print("Fail to restore policy and optim.")
        buffer_path = os.path.join(log_path, "train_buffer.pkl")
        if os.path.exists(buffer_path):
            train_collector.buffer = pickle.load(open(buffer_path, "rb"))
            print("Successfully restore buffer.")
        else:
            print("Fail to restore buffer.")

    # trainer
    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=None,
        max_epoch=cfg.epoch,
        step_per_epoch=cfg.step_per_epoch,
        step_per_collect=cfg.step_per_collect,
        episode_per_test=0,
        batch_size=cfg.batch_size,
        update_per_step=cfg.update_per_step,
        train_fn=train_fn,
        test_fn=None,
        stop_fn=None,
        save_best_fn=save_best_fn,
        logger=logger,
        resume_from_log=cfg.resume,
        save_checkpoint_fn=save_checkpoint_fn,
    )

    # assert stop_fn(result["best_reward"])

    pprint.pprint(result)

    return policy


def evaluate(cfgs, policy):
    # Let's watch its performance!
    env = gym.make(cfgs.task)
    policy.eval()
    policy.set_eps(cfgs.eps_test)
    collector = Collector(policy, env)
    result = collector.collect(n_episode=10, render=cfgs.render)
    rews, lens = result["rews"], result["lens"]
    print(f"The trained model reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == '__main__':
    from rainbow.config import cfg

    cfg.action_space = gym.spaces.Discrete(cfg.action_per_dim[0] * cfg.action_per_dim[1])
    cfg.observation_space = gym.spaces.Box(-1e4, 1e4, shape=(70,))

    cfg.state_shape = cfg.observation_space.shape
    cfg.action_shape = cfg.action_space.n

    policy = train(cfgs=cfg)
