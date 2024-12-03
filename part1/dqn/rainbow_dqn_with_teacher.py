import gymnasium as gym
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import random
from copy import deepcopy
from gymnasium.wrappers import MaxAndSkipObservation, ResizeObservation, FrameStackObservation
import cv2
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import collections
import wandb
import tensorboard
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.wrappers import MaxAndSkipObservation, ResizeObservation, GrayscaleObservation, FrameStackObservation, ReshapeObservation
import ale_py

# ---------------------------------------------------------------------------------------------------------------

class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

def make_env(env_name):
    env = gym.make(env_name, obs_type="grayscale")
    print("Standard Env.        : {}".format(env.observation_space.shape))
    
    env = MaxAndSkipObservation(env, skip=4)
    print("MaxAndSkipObservation: {}".format(env.observation_space.shape))
    
    env = ResizeObservation(env, (84, 84))
    print("ResizeObservation    : {}".format(env.observation_space.shape))
    
    env = FrameStackObservation(env, stack_size=4)
    print("FrameStackObservation: {}".format(env.observation_space.shape))
    
    env = ScaledFloatFrame(env)
    print("ScaledFloatFrame     : {}\n".format(env.observation_space.shape))
    
    return env

# Define namedtuple for transitions
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'done', 'next_state'])

# ---------------------------------------------------------------------------------------------------------------

# --- Rainbow DQN components ---
class NoisyLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(NoisyLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.zeros(out_features))
        self.sigma_bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        epsilon_w = torch.randn_like(self.mu_weight)
        epsilon_b = torch.randn_like(self.mu_bias)
        weight = self.mu_weight + self.sigma_weight * epsilon_w
        bias = self.mu_bias + self.sigma_bias * epsilon_b
        return F.linear(x, weight, bias)

class RainbowDQN(nn.Module):
    def __init__(self, input_shape, num_actions, num_atoms=51, v_min=-10.0, v_max=10.0):
        super(RainbowDQN, self).__init__()
        
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc = nn.Sequential(
            NoisyLayer(7 * 7 * 64, 512), nn.ReLU()
        )

        # Value and Advantage streams using Noisy layers
        self.value = NoisyLayer(512, num_atoms)
        self.advantage = NoisyLayer(512, num_actions * num_atoms)
    
    '''
    def forward(self, x):
        features = self.features(x)
        fc_output = self.fc(features)
        
        value = self.value(fc_output).view(-1, 1, self.num_atoms)
        advantage = self.advantage(fc_output).view(-1, self.num_atoms, -1)
        
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    '''

    def forward(self, x):
        features = self.features(x)
        fc_output = self.fc(features)

        # Compute the value for each atom
        value = self.value(fc_output).view(-1, 1, self.num_atoms)  # (batch_size, 1, num_atoms)
        
        # Compute the advantage for each atom-action pair
        advantage = self.advantage(fc_output).view(-1, self.num_actions, self.num_atoms)  # (batch_size, num_actions, num_atoms)
        
        # Combine value and advantage to get Q-values
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values

    
    def get_distribution(self, state):
        q_values = self.forward(state)
        dist = F.softmax(q_values, dim=2)  # Softmax over atoms
        return dist

    def get_q_value(self, state, action):
        dist = self.get_distribution(state)
        batch_size = dist.size(0)
        return torch.sum(dist * self.get_atoms().view(1, 1, -1), dim=2)  # Sum over atoms

    def get_atoms(self):
        return torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)

# ---------------------------------------------------------------------------------------------------------------

# --- Prioritized Experience Replay ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.pos = 0

    def append(self, transition, td_error):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_priority if td_error is None else abs(td_error)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        transitions = [self.buffer[idx] for idx in indices]
        batch = Transition(*zip(*transitions))
        return batch, torch.tensor(weights, dtype=torch.float32), indices

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)  # Return the number of stored transitions

# ---------------------------------------------------------------------------------------------------------------

# --- Training function with all Rainbow features ---
def train(env, model, buffer, optimizer, device, num_episodes=500, batch_size=32, gamma=0.99, target_update_freq=1000, plateau_window=30, plateau_threshold=0.01, n_steps=3):
    
    model.train() # Set model to training mode

    target_net = deepcopy(model) # Initialize target network with the same weights
    target_net.eval() # Set target network to evaluation mode

    epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.1, 2000 # -- Epsilon-greedy parameters --
    epsilon = epsilon_start # Initialize epsilon
    frame_idx = 0 

    # TensorBoard writer
    writer = SummaryWriter(log_dir="runs/rainbow_dqn")

    # Initialize ReduceLROnPlateau scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7) # Reduce learning rate on plateau

    mean_rewards = []

    for episode in range(num_episodes):

        state, _ = env.reset()
        state = torch.tensor(state, device=device).squeeze(-1).unsqueeze(0).float() # Shape: (1, 4, 84, 84)

        episode_reward = 0
        done = False
        n_step_buffer = deque(maxlen=n_steps) # n-step buffer

        while not done:

            frame_idx += 1
            epsilon = max(epsilon_end, epsilon_start - frame_idx / epsilon_decay) # Linearly decay epsilon

            # -- Epsilon-greedy exploration --
            if random.random() > epsilon:

                # Get Q-values and compute the action with the highest value
                q_values = model(state)  # Shape: (batch_size, num_actions, num_atoms)
                action = q_values.mean(dim=2).argmax(dim=1).item()  # Take the argmax across actions, averaged across atoms

            else:

                action = random.randint(0, env.action_space.n - 1)

            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.tensor(next_state, device=device).squeeze(-1).unsqueeze(0).float() # Shape: (1, 4, 84, 84)
            
            # Store experience in n-step buffer
            n_step_buffer.append((state, action, reward, done, next_state))
            state = next_state
            episode_reward += reward

            if len(n_step_buffer) == n_steps:

                # Compute multi-step return
                total_reward = sum([r[2] * (gamma ** idx) for idx, r in enumerate(n_step_buffer)])
                done_mask = n_step_buffer[-1][3]
                buffer.append((state, action, total_reward, done_mask, next_state), None)

            if len(buffer) >= batch_size:
                
                batch, weights, indices = buffer.sample(batch_size)

                # Prepare batched tensors
                states = torch.cat(batch.state).squeeze(-1).float()
                actions = torch.tensor(batch.action, device=device).unsqueeze(1)
                rewards = torch.tensor(batch.reward, device=device)
                dones = torch.tensor(batch.done, device=device).float()
                next_states = torch.cat(batch.next_state).squeeze(-1).float()

                # Compute Q-values using Double DQN
                q_values = model(states)
                next_q_values = target_net.get_distribution(next_states)  # get_distribution gives probability over atoms
                next_actions = q_values.argmax(1, keepdim=True)  # Double DQN action selection
                next_q_values = next_q_values.gather(1, next_actions)  # Get the next state's q-values for the selected action

                # Compute the target for multi-step (n-step) updates
                rewards_expanded = rewards.unsqueeze(1).unsqueeze(2).expand(-1, model.num_actions, model.num_atoms)
                dones_expanded = dones.unsqueeze(1).unsqueeze(2).expand(-1, model.num_actions, model.num_atoms)
                targets = rewards_expanded + gamma * (1 - dones_expanded) * next_q_values

                dist = model.get_distribution(states)
                action_dist = dist.gather(1, actions.unsqueeze(-1).expand(-1, -1, model.num_atoms))
                loss = -(targets * torch.log(action_dist + 1e-6)).sum(dim=2).mean()

                # -- Backpropagate the loss --
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Log loss to TensorBoard
                writer.add_scalar("Loss", loss.item(), frame_idx)

            # Update target network every `target_update_freq` frames
            if frame_idx % target_update_freq == 0:
                target_net.load_state_dict(model.state_dict())

        # Log episode reward and other metrics
        writer.add_scalar("Episode Reward", episode_reward, episode)
        print(f"Episode {episode}, Reward: {episode_reward:.2f}")

        # -- Monitor plateau and adjust learning rate --
        mean_rewards.append(episode_reward)

        if len(mean_rewards) > plateau_window:
            recent_mean = np.mean(mean_rewards[-plateau_window:])
            previous_mean = np.mean(mean_rewards[-plateau_window * 2:-plateau_window])

            if abs(recent_mean - previous_mean) < plateau_threshold:  # Plateau detected, reduce learning rate
                print(f"Plateau detected at episode {episode}, reducing learning rate.")
                scheduler.step(loss)  # Use the loss for the scheduler

            # Keep only the last `plateau_window` rewards to detect plateau
            mean_rewards = mean_rewards[-plateau_window:]

    writer.close()

# ---------------------------------------------------------------------------------------------------------------

# -- Main --
if __name__ == "__main__":

    # -- Set device --
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # -- Environment --
    env_name = "Breakout-v4"
    print(f"\nEnvironment: {env_name}\n")

    # -- Create Environment --
    env = make_env(env_name)

    # -- Get input shape and number of actions --
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n

    print(f"Observation Space: {input_shape}")
    print(f"Action Space: {num_actions}\n")

    # -- Initialize model --
    rainbow_dqn_model = RainbowDQN(input_shape, num_actions).to(device)

    # Replay buffer and optimizer
    buffer = PrioritizedReplayBuffer(100000)
    optimizer = optim.Adam(rainbow_dqn_model.parameters(), lr=1e-4)

    # Train the agent
    train(env, rainbow_dqn_model, buffer, optimizer, device)

               
