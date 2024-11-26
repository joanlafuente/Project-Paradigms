# Import necessary libraries
from collections import namedtuple, deque
import numpy as np
from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import torch.optim as optim
from IPython.display import clear_output
import matplotlib.pyplot as plt
from copy import deepcopy
import collections
import random
import wandb
import copy
import gymnasium as gym
from gymnasium.wrappers import MaxAndSkipObservation, ResizeObservation, GrayscaleObservation, FrameStackObservation, ReshapeObservation
import ale_py

# ------------------------------------------------------------------------------------------------

# Environment used
ENV_NAME = "ALE/Breakout-v5"

class ScaledFloatFrame(gym.ObservationWrapper):

    def observation(self, obs):

        return np.array(obs).astype(np.float32) / 255.0

def make_env(env_name):

    print()
    env = gym.make(env_name, obs_type="grayscale")
    print()
    print("Standard Env.        : {}".format(env.observation_space.shape))

    env = MaxAndSkipObservation(env, skip=4)
    print("MaxAndSkipObservation: {}".format(env.observation_space.shape))

    env = ResizeObservation(env, (84, 84))
    print("ResizeObservation    : {}".format(env.observation_space.shape))
    
    env = FrameStackObservation(env, stack_size=4)
    print("FrameStackObservation: {}".format(env.observation_space.shape))
    
    env = ScaledFloatFrame(env)
    print("ScaledFloatFrame     : {}".format(env.observation_space.shape))
    
    return env

# Create the environment
env = make_env(ENV_NAME)

print("\nAction space is {} ".format(env.action_space))
print("Observation space is {} ".format(env.observation_space))

# ------------------------------------------------------------------------------------------------


