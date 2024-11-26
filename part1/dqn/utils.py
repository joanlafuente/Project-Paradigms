
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

def compute_loss(model, target_model, states, actions, rewards, dones, next_states, gamma=0.99, criterion=nn.MSELoss()):
    
    Q_values = model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

    next_state_values = target_model(next_states).max(1)[0]
    next_state_values[dones] = 0.0
    next_state_values = next_state_values.detach()

    expected_Q_values = next_state_values * gamma + rewards

    return criterion(Q_values, expected_Q_values)

# ------------------------------------------------------------------------------------------------

class PrioritizedExperienceReplay:

    """
    D'aquest metode no se si es correcte del tot, en comptes de calcular el 
    td_error quan faig el sampling ho he implementat en el moment en que safegeix al buffer.

    D'aquesta forma no augmenta molt tant al numero de claculs extra (Sino shauria de clacular 
    per a tot el buffer cada vegada que fem sampling)
    """
    
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.priorities = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience, model, target_model, device="cpu"):
        td_error = compute_td_error(model, target_model, experience.state, experience.action, experience.reward, experience.done, experience.new_state, device=device)
        self.priorities.append(td_error)
        self.buffer.append(experience)

    def sample(self, BATCH_SIZE, alpha=0.6, beta=0.4, epsilon=0.01):
        priorities = np.array(self.priorities)
        priorities = priorities + epsilon
        probabilities = priorities ** alpha
        probabilities = probabilities / probabilities.sum()

        indices = np.random.choice(len(self.buffer), BATCH_SIZE, p=probabilities)
        weights = (1/len(self.buffer) * 1/probabilities[indices]) ** beta
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        states = torch.from_numpy(np.array(states, dtype=np.float32))
        actions = torch.from_numpy(np.array(actions, dtype=np.int64))
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32))
        dones = torch.from_numpy(np.array(dones, dtype=bool))
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32))
        weights = torch.from_numpy(np.array(weights, dtype=np.float32))

        return states, actions, rewards, dones, next_states, weights