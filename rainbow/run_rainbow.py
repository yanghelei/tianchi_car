import os
import sys
sys.path.append(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + '/')
import gym
import torch
import pprint
import pickle
from geek.env.matrix_env import Scenarios
from torch.utils.tensorboard import SummaryWriter

from ts_inherit.logger import MyLogger
from ts_inherit.networks import MyActor
from ts_inherit.rainbow import MyRainbow
from ts_inherit.collector import MyCollector
from ts_inherit.trainer import my_policy_trainer

from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from tianshou.utils.net.discrete import NoisyLinear
from tianshou.data import PrioritizedVectorReplayBuffer, VectorReplayBuffer

from utils.processors import set_seed, Processor
# from utils.exploration import get_epsilon_greedy_fn


def make_train_env(cfgs, render_id=None):
    if render_id:
        env = gym.make(cfgs.task, scenarios=Scenarios.TRAINING, render_id=str(render_id))
    else:
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
    train_envs = SubprocVectorEnv([lambda: make_train_env(cfgs, i) for i in range(cfgs.training_num)])
    test_envs = DummyVectorEnv([lambda: make_train_env(cfgs) for _ in range(15-cfgs.training_num)])

    set_seed(seed=cfgs.seed)

    def noisy_linear(x, y):
        return NoisyLinear(x, y, cfgs.noisy_std)

    net = MyActor(
        cfgs.network,
        cfgs.action_shape,
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

    # log
    log_path = os.path.join(cfgs.logdir, cfgs.task, "rainbow")
    writer = SummaryWriter(log_path)
    logger = MyLogger(writer, save_interval=cfgs.save_interval)

    logger.logger.info('device: ' + str(cfgs.device))

    train_processor = Processor(cfgs, net, logger.logger, n_env=cfgs.training_num)
    test_processor = Processor(cfgs, net, logger.logger, n_env=15-cfgs.training_num)

    # collector
    train_collector = MyCollector(policy, train_envs, buf, preprocess_fn=train_processor.preprocess_fn, exploration_noise=True)
    train_collector.set_logger(logger.logger, type='train')
    test_collector = MyCollector(policy, test_envs, preprocess_fn=test_processor.preprocess_fn, exploration_noise=False)
    test_collector.set_logger(logger.logger)

    # policy logger
    policy.set_logger(logger.logger)

    train_collector.collect(n_step=cfgs.batch_size * cfgs.training_num, random=True)

    def save_best_fn(policy):
        # 保存最优模型
        torch.save(policy.state_dict(), os.path.join(log_path, "best_policy.pth"))
        logger.logger.info('save best model into ' + str(os.path.join(log_path, "best_policy.pth")) + ' successfully!')

    # def stop_fn(mean_rewards):
    #     # 终止条件
    #     return mean_rewards >= cfgs.reward_threshold

    def train_fn(epoch, env_step):
        # 在每次训练前执行的操作
        if env_step >= cfgs.exploration.decay:
            policy.set_eps(cfgs.exploration.end)
        else:
            policy.set_eps((cfgs.exploration.start - cfgs.exploration.end) * (1 - env_step / cfgs.exploration.decay) + cfgs.exploration.end)

        # beta annealing, just a demo
        if cfgs.prioritized_replay:
            if env_step <= 10000:
                beta = cfgs.beta
            elif env_step <= 50000:
                beta = cfgs.beta - (env_step - 10000) / 40000 * (cfgs.beta - cfgs.beta_final)
            else:
                beta = cfgs.beta_final
            buf.set_beta(beta)

    def test_fn(epoch, env_step):
        policy.set_eps(cfgs.eps_test)

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        torch.save({"model": policy.state_dict(), "optim": optim.state_dict(), }, ckpt_path)
        buffer_path = os.path.join(log_path, "train_buffer.pkl")
        pickle.dump(train_collector.buffer, open(buffer_path, "wb"))
        return ckpt_path

    if cfgs.resume:
        # load from existing checkpoint
        logger.logger.info(f"Loading agent under {log_path}")
        ckpt_path = os.path.join(log_path, "checkpoint.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=cfgs.device)
            policy.load_state_dict(checkpoint['model'])
            policy.optim.load_state_dict(checkpoint['optim'])
            logger.logger.info("Successfully restore policy and optim.")
        else:
            logger.logger.info("Fail to restore policy and optim.")
        buffer_path = os.path.join(log_path, "train_buffer.pkl")
        if os.path.exists(buffer_path):
            train_collector.buffer = pickle.load(open(buffer_path, "rb"))
            logger.logger.info("Successfully restore buffer.")
        else:
            logger.logger.info("Fail to restore buffer.")

    # trainer
    result = my_policy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=cfgs.epoch,
        step_per_epoch=cfgs.step_per_epoch,
        step_per_collect=cfgs.step_per_collect,
        episode_per_test=cfgs.test_num,  # 每次test多少个episode
        batch_size=cfgs.batch_size,
        update_per_step=cfgs.update_per_step,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=None,
        save_best_fn=save_best_fn,
        logger=logger,
        show_progress=False,
        resume_from_log=cfgs.resume,
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
    collector = MyCollector(policy, env)
    result = collector.collect(n_episode=20, render=cfgs.render)
    rews, lens = result["rews"], result["lens"]
    print(f"The trained model reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == '__main__':
    torch.set_num_threads(1)

    from rainbow.config import cfg

    cfg.action_space = gym.spaces.Discrete(cfg.action_per_dim[0] * cfg.action_per_dim[1])
    cfg.action_shape = cfg.action_space.n

    policy = train(cfgs=cfg)
