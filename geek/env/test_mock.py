import gym
import numpy

from geek.env.matrix_env import Scenarios

env = gym.make("MatrixEnv", scenarios=Scenarios.TRAINING)
obs = env.reset()
while True:
    observation, reward, done, info = env.step(numpy.array([0.1, 0]))
    if done:
        print(f"{env.instance_key} is done for done msg:{info}.")
        obs = env.reset()
