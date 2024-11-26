
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

# Define the Dueling DQN architecture
class DuelingDQN(nn.Module):

    def __init__(self, input_shape, output_shape):

        super(DuelingDQN, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=5, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1600, 512),
            nn.ReLU()
        )

        self.value_prediction = nn.Linear(512, 1)
        self.advantage_prediction = nn.Linear(512, output_shape)

    def forward(self, x):
        
        embedding = self.net(x)
        value = self.value_prediction(embedding)
        advantage = self.advantage_prediction(embedding)
        q_values = value + advantage - advantage.mean(dim=-1).unsqueeze(-1)
        return q_values