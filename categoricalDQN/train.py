import gymnasium as gym
import ale_py
from ale_py import ALEInterface
from gymnasium.wrappers import MaxAndSkipObservation, ResizeObservation, GrayscaleObservation, FrameStackObservation, ReshapeObservation
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import sys

from model import C51DQN
from agent import ReplayBuffer

# ------------------------------------------------------------------------------------

# ale = ALEInterface()

# version
print("Using Gymnasium version {}".format(gym.__version__))

gym.register_envs(ale_py)

# ------------------------------------------------------------------------------------

class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


def make_env(env_name, render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    print("Standard Env.        : {}".format(env.observation_space.shape))
    env = MaxAndSkipObservation(env, skip=4)
    print("MaxAndSkipObservation: {}".format(env.observation_space.shape))
    #env = FireResetEnv(env)
    env = ResizeObservation(env, (84, 84))
    print("ResizeObservation    : {}".format(env.observation_space.shape))
    env = GrayscaleObservation(env, keep_dim=True)
    print("GrayscaleObservation : {}".format(env.observation_space.shape))
    env = ImageToPyTorch(env)
    print("ImageToPyTorch       : {}".format(env.observation_space.shape))
    env = ReshapeObservation(env, (84, 84))
    print("ReshapeObservation   : {}".format(env.observation_space.shape))
    env = FrameStackObservation(env, stack_size=4)
    print("FrameStackObservation: {}".format(env.observation_space.shape))
    env = ScaledFloatFrame(env)
    print("ScaledFloatFrame     : {}".format(env.observation_space.shape))
    
    return env

def project_distribution(next_dist, rewards, dones, n_atoms, v_min, v_max, gamma):
    batch_size = rewards.shape[0]
    delta_z = (v_max - v_min) / (n_atoms - 1)
    z = torch.linspace(v_min, v_max, n_atoms).unsqueeze(0)
    rewards = rewards.unsqueeze(1)
    dones = dones.unsqueeze(1)
    
    # Compute the projected distribution
    next_z = rewards + (1 - dones) * gamma * z
    next_z = next_z.clamp(min=v_min, max=v_max)
    b = (next_z - v_min) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    m = torch.zeros_like(next_dist)
    for i in range(batch_size):
        for j in range(n_atoms):
            m[i, l[i, j]] += next_dist[i, j] * (u[i, j] - b[i, j])
            m[i, u[i, j]] += next_dist[i, j] * (b[i, j] - l[i, j])
    return m

# ------------------------------------------------------------------------------------

ENV_NAME = "ALE/Breakout-v5"
test_env = gym.make(ENV_NAME)
# env = gym.make(ENV_NAME)

env = make_env(ENV_NAME)

def print_env_info(name, env):
    obs, _ = env.reset()
    print("*** {} Environment ***".format(name))
    print("Environment obs. : {}".format(env.observation_space.shape))
    print("Observation shape: {}, type: {} and range [{},{}]".format(obs.shape, obs.dtype, np.min(obs), np.max(obs)))
    print("Observation sample:\n{}".format(obs))

# ------------------------------------------------------------------------------------

# Hyperparameters
n_atoms = 51
v_min = -10
v_max = 10
gamma = 0.99
batch_size = 32
learning_rate = 0.00025
buffer_capacity = 10000
epsilon = 0.1

# Initialize components
state_shape = env.observation_space.shape # (4, 84, 84)
n_actions = env.action_space.n # 4
replay_buffer = ReplayBuffer(buffer_capacity)
c51_dqn = C51DQN(state_shape, n_actions, n_atoms, v_min, v_max)
target_dqn = C51DQN(state_shape, n_actions, n_atoms, v_min, v_max)
target_dqn.load_state_dict(c51_dqn.state_dict())
optimizer = optim.Adam(c51_dqn.parameters(), lr=learning_rate)

# Training loop
for episode in range(1, 501):
    state, _ = env.reset() # env.reset() returns a tuple (state, info)
    # state -> (4, 84, 84) -> numpy array

    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # (4, 84, 84) -> (1, 4, 84, 84) Add batch dimension & convert to tensor
    done = False
    total_reward = 0

    while not done:
        # Select action using epsilon-greedy policy
        if random.random() < epsilon: # Random action
            action = env.action_space.sample()
            # print("Random action: ", action)
        else: # Greedy action based on the current Q-values
            dist = c51_dqn(state)
            q_values = (dist * c51_dqn.z_values).sum(dim=2)
            action = torch.argmax(q_values, dim=1).item()
            # print("Action: ", action)

        # Take action in the environment
        next_state, reward, terminated, truncated, info = env.step(action) # env.step() returns a tuple (next_state, reward, done, info)
        # state -> (4, 84, 84) -> numpy array
        total_reward += reward
        next_state = torch.tensor(next_state, dtype=torch.float32) # (4, 84, 84) convert to tensor
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state

        # Sample a batch from the replay buffer and train the network
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            # next_states = next_states.squeeze(1)  # Removes the second dimension if it has size 1
            dones = torch.tensor(dones, dtype=torch.float32)

            # Compute target distribution
            with torch.no_grad():
                next_dist = target_dqn(next_states)
                next_q_values = (next_dist * target_dqn.z_values).sum(dim=2)
                next_actions = torch.argmax(next_q_values, dim=1)
                next_dist = next_dist[range(batch_size), next_actions]
                target_dist = project_distribution(next_dist, rewards, dones, n_atoms, v_min, v_max, gamma)

            # Compute loss
            dist = c51_dqn(states)
            dist = dist[range(batch_size), actions]
            loss = -torch.sum(target_dist * torch.log(dist + 1e-6), dim=1).mean()

            # Update network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Update target network
    if episode % 10 == 0:
        target_dqn.load_state_dict(c51_dqn.state_dict())

    print(f"Episode {episode}, Total Reward: {total_reward}")
