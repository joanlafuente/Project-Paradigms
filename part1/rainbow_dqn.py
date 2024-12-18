
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------

import gymnasium as gym
from gymnasium.wrappers import MaxAndSkipObservation, ResizeObservation, GrayscaleObservation, FrameStackObservation, ReshapeObservation
from stable_baselines3.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, FireResetEnv, EpisodicLifeEnv
import ale_py
import cv2
from gymnasium.spaces import Box

import os
from gymnasium.spaces import Box
from torch import nn
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import collections
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import wandb
import pickle
import math
import random

# -------------------------------------------------------------------------------------------------------------
# Custom wrappers and utilities for preprocessing environment observations
# -------------------------------------------------------------------------------------------------------------

class ScaledFloatFrame(gym.ObservationWrapper):
    """
    Scales observation pixels to the range [0, 1] and converts to float32.
    """
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

def make_env(env_name):
    """
    Sets up the environment with a series of wrappers for preprocessing:
    1. MaxAndSkipObservation: Skips frames to reduce computation.
    2. ResizeObservation: Resizes observation to (84, 84).
    3. FrameStackObservation: Stacks the last 4 frames.
    4. ScaledFloatFrame: Scales pixel values to [0, 1].
    """
    env = gym.make(env_name, obs_type="grayscale")
    print("Standard Env.        : {}".format(env.observation_space.shape))

    env = MaxAndSkipObservation(env, skip=4)
    env = ResizeObservation(env, (84, 84))
    print("ResizeObservation    : {}".format(env.observation_space.shape))

    env = FrameStackObservation(env, stack_size=4)
    print("FrameStackObservation: {}".format(env.observation_space.shape))

    env = ScaledFloatFrame(env)
    print("ScaledFloatFrame     : {}".format(env.observation_space.shape))

    return env

def is_plateau(rewards, plateau_threshold=1e-2, plateau_length=100):
    """
    Detects reward plateau when the range of recent rewards is very small.
    """
    if len(rewards) < plateau_length:
        return False
    recent_rewards = rewards[-plateau_length:]
    reward_range = max(recent_rewards) - min(recent_rewards)
    return reward_range < plateau_threshold

def introduce_noise(model, noise_factor=0.1):
    """
    Adds Gaussian noise to the sigma parameters of noisy layers to explore new behaviors.
    """
    for name, param in model.named_parameters():
        if "sigma_weight" in name or "sigma_bias" in name:
            param.data += noise_factor * torch.randn_like(param.data)

def compute_loss(model, target_model, states, actions, rewards, dones, next_states, gamma=0.99, criterion=nn.MSELoss(), device="cpu"):
    """
    Computes the Bellman loss for training the DQN.
    1. Uses current model to get Q-values for actions taken.
    2. Uses target model to compute future Q-values.
    3. Applies the Bellman equation to compute expected Q-values.
    4. Computes loss using the mean squared error.
    """
    qvals = model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    actions_next = torch.argmax(model(next_states), dim=1)
    
    with torch.no_grad():
        qvals_next = target_model(next_states, noise_enabled=False)
        qvals_next = torch.tensor(
            np.array([qvals_next[i][actions_next[i]].cpu() for i in range(len(actions_next))])
        ).to(device)
    qvals_next[dones] = 0

    expected_qvals = rewards + gamma * qvals_next
    return criterion(qvals, expected_qvals)

def compute_td_error(model, target_model, states, actions, rewards, dones, next_states, gamma=0.99, device="cpu"):
    """
    Computes the temporal difference (TD) error for prioritizing experiences.
    """
    states = torch.tensor(states).to(device)
    actions = torch.tensor(actions).to(device)
    rewards = torch.tensor(rewards).to(device)
    dones = torch.tensor(dones).to(device)
    next_states = torch.tensor(next_states).to(device)

    with torch.no_grad():
        Q_values = model(states.unsqueeze(0)).gather(1, actions.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
        next_state_values = target_model(next_states.unsqueeze(0)).max(1)[0]
        next_state_values[dones] = 0.0
        expected_Q_values = next_state_values * gamma + rewards

    return (Q_values - expected_Q_values).abs().detach().item()

# -------------------------------------------------------------------------------------------------------------
# Noisy Layer: Introduces parameterized noise for exploration in DQN
# -------------------------------------------------------------------------------------------------------------

class NoisyLayer(nn.Module):
    """
    Implements a noisy linear layer as described in "Noisy Networks for Exploration".
    This replaces epsilon-greedy exploration.
    """
    def __init__(self, in_features, out_features):
        super(NoisyLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Mean parameters initialized uniformly
        mu_range = 1 / math.sqrt(in_features)
        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-mu_range, mu_range))
        self.mu_bias = nn.Parameter(torch.empty(out_features).uniform_(-mu_range, mu_range))

        # Noise parameters initialized with higher sigma for exploration
        sigma_init = 1.0 / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))

        # Buffers for noise
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        self.register_buffer("epsilon_bias", torch.zeros(out_features))

    def forward(self, x, noise_enabled=True):
        if noise_enabled:
            epsilon_in = torch.randn(self.in_features, device=x.device)
            epsilon_out = torch.randn(self.out_features, device=x.device)
            epsilon_w = torch.sign(epsilon_in) * torch.sqrt(torch.abs(epsilon_in))
            epsilon_b = torch.sign(epsilon_out) * torch.sqrt(torch.abs(epsilon_out))
            self.epsilon_weight = epsilon_out.unsqueeze(1) * epsilon_w.unsqueeze(0)
            self.epsilon_bias = epsilon_b
            weight = self.mu_weight + self.sigma_weight * self.epsilon_weight
            bias = self.mu_bias + self.sigma_bias * self.epsilon_bias
        else:
            weight = self.mu_weight
            bias = self.mu_bias

        return F.linear(x, weight, bias)

# -------------------------------------------------------------------------------------------------------------
# Dueling DQN: Combines value and advantage streams for better learning
# -------------------------------------------------------------------------------------------------------------

class DQN(nn.Module):
    """
    Implements a dueling DQN with noisy layers for better exploration.
    """
    def __init__(self, input_shape, output_shape):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=5, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(1600, 512),
            nn.LeakyReLU()
        )
        self.value_prediction = NoisyLayer(512, 1)
        self.advantage_prediction = NoisyLayer(512, output_shape)

    def forward(self, x, noise_enabled=False):
        embedding = self.net(x)
        value = self.value_prediction(embedding, noise_enabled=noise_enabled)
        advantage = self.advantage_prediction(embedding, noise_enabled=noise_enabled)
        q_values = value + advantage - advantage.mean(dim=-1).unsqueeze(-1)
        return q_values

# -------------------------------------------------------------------------------------------------------------
# Agent Class: Defines agent interaction with the environment
# -------------------------------------------------------------------------------------------------------------

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class Agent:
    def __init__(self, env, exp_replay_buffer):
        """
        Initializes the agent.
        - env: The environment the agent interacts with.
        - exp_replay_buffer: Experience replay buffer for storing experiences.
        """
        self.env = env
        self.exp_replay_buffer = exp_replay_buffer
        self.action_counts = collections.defaultdict(int)  # Tracks action usage counts.
        self._reset()

    def _reset(self):
        """
        Resets the environment and agent's state.
        """
        self.current_state = self.env.reset()[0]
        self.total_reward = 0.0

    def step(self, net, target_net, device):
        """
        Takes one step in the environment:
        - Selects action using the noisy DQN network.
        - Logs action statistics for analysis.
        - Updates experience replay buffer.
        - Resets when the environment episode ends.
        """
        done_reward = None

        # Convert the current state to tensor for inference
        state_ = np.array([self.current_state])
        state = torch.tensor(state_).to(device)

        # Get Q-values and select action with max Q-value
        q_vals = net(state, noise_enabled=True)
        _, act_ = torch.max(q_vals, dim=1)
        action = int(act_.item())

        # Update action counts for logging
        self.action_counts[action] += 1

        # Perform action in the environment
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        is_done = terminated or truncated
        self.total_reward += reward

        # Log action distribution to WandB
        wandb.log({
            f"action_{action}": self.action_counts[action] / sum(self.action_counts.values())
        })

        # Store experience in the replay buffer
        exp = Experience(self.current_state, action, reward, is_done, new_state)
        self.exp_replay_buffer.append(exp, net, target_net, device)
        self.current_state = new_state

        # If the episode is done, reset the agent
        if is_done:
            done_reward = self.total_reward
            self._reset()

        return done_reward

# -------------------------------------------------------------------------------------------------------------
# N-Step Prioritized Experience Replay
# -------------------------------------------------------------------------------------------------------------

class NStepPrioritizedExperienceReplay:
    def __init__(self, capacity, n_steps, gamma):
        """
        Initializes an N-Step Prioritized Experience Replay Buffer.
        - capacity: Max buffer size.
        - n_steps: Number of steps for N-step returns.
        - gamma: Discount factor.
        """
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
        self.priorities = collections.deque(maxlen=capacity)
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_step_buffer = collections.deque(maxlen=n_steps)

    def __len__(self):
        return len(self.buffer)

    def _get_n_step_info(self):
        """
        Computes N-step cumulative reward and next state.
        """
        reward, next_state, done = 0.0, None, False
        for idx, (state, action, r, d, next_s) in enumerate(self.n_step_buffer):
            reward += (self.gamma ** idx) * r
            next_state = next_s
            if d:  # Stop if done flag is True
                done = True
                break
        return reward, next_state, done

    def append(self, experience, model, target_model, device="cpu"):
        """
        Adds an experience to the buffer with N-step logic.
        Also calculates and assigns TD-error-based priority.
        """
        self.n_step_buffer.append(experience)
        if len(self.n_step_buffer) == self.n_steps:
            n_step_reward, next_state, done = self._get_n_step_info()
            state, action, _, _, _ = self.n_step_buffer[0]
            n_step_experience = Experience(state, action, n_step_reward, done, next_state)
            self.buffer.append(n_step_experience)

            # Compute TD error for prioritization
            td_error = compute_td_error(
                model, target_model, n_step_experience.state, n_step_experience.action,
                n_step_experience.reward, n_step_experience.done, n_step_experience.new_state, device=device
            )
            self.priorities.append(td_error)

    def sample(self, batch_size, alpha=0.6, beta=0.5, epsilon=0.01):
        """
        Samples a batch of experiences using prioritized sampling.
        """
        priorities = np.array(self.priorities) + epsilon
        probabilities = priorities ** alpha
        probabilities = probabilities / probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        weights = (1 / len(self.buffer) * 1 / probabilities[indices]) ** beta

        # Extract and prepare sampled data
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        states = torch.from_numpy(np.array(states, dtype=np.float32))
        actions = torch.from_numpy(np.array(actions, dtype=np.int64))
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32))
        dones = torch.from_numpy(np.array(dones, dtype=bool))
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32))
        weights = torch.from_numpy(np.array(weights, dtype=np.float32))

        return states, actions, rewards, dones, next_states, weights
    
# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

# Set device for training (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training constants
MEAN_REWARD_BOUND = 390.0  # Threshold for considering the environment solved
NUMBER_OF_REWARDS_TO_AVERAGE = 50  # Number of rewards to calculate moving average

# Hyperparameters
GAMMA = 0.9  # Discount factor for future rewards
BATCH_SIZE = 128  # Batch size for training
LEARNING_RATE = 0.0001  # Learning rate for optimizer
MAX_FRAMES = 1500000  # Maximum number of frames to train
EXPERIENCE_REPLAY_SIZE = 1000  # Size of the experience replay buffer
SYNC_TARGET_NETWORK = 500  # Sync target network every N frames

# Epsilon for exploration-exploitation balance in epsilon-greedy policy
EPS_START = 1.0  # Initial epsilon
EPS_DECAY = 0.99998  # Decay rate for epsilon
EPS_MIN = 0.03  # Minimum epsilon value

# Environment setup
ENV_NAME = "ALE/Frogger-v5"  # Environment name for Atari Frogger

# Parameters for plateau detection and noise adjustment
plateau_length = 100  # Number of recent rewards to check for plateau
plateau_threshold = 1e-2  # Threshold for minimal improvement to detect plateau
noise_factor = 0.1  # Initial noise factor for sigma parameters
noise_decay = 0.99  # Decay factor for noise
noise_frames = 5000  # Duration for maintaining added noise

# Flags and trackers for noise management
noise_active = False  # Indicates whether noise is currently active
noise_start_frame = 0  # Frame number when noise was introduced

# Initialize logging with Weights & Biases
wandb.init(project="Frogger", name="lr-0001_gamma_90_v5_rainbow")
wandb.config.update({
    "gamma": GAMMA,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "experience_replay_size": EXPERIENCE_REPLAY_SIZE,
    "sync_target_network": SYNC_TARGET_NETWORK,
    "eps_start": EPS_START,
    "eps_decay": EPS_DECAY,
    "eps_min": EPS_MIN
})

# Initialize environment, model, replay buffer, and agent
env = make_env(ENV_NAME)  # Create and wrap the environment
net = DQN(env.observation_space.shape, env.action_space.n).to(device)  # Main DQN
target_net = DQN(env.observation_space.shape, env.action_space.n).to(device)  # Target DQN
buffer = NStepPrioritizedExperienceReplay(EXPERIENCE_REPLAY_SIZE, n_steps=2, gamma=GAMMA)  # N-step prioritized buffer
agent = Agent(env, buffer)  # Initialize the agent

# Optimizer and loss
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss(reduction="none")  # Loss for TD error, reduction set for priorities

# Hyperparameter for training loop
total_rewards = []  # List to store total rewards per game
frame_number = 0  # Counter for total frames

# Initialize a learning rate scheduler to adjust LR on plateau
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True)

# Progress bar for tracking training progress
tbar = tqdm()

# Main training loop
while True:

    # Periodically save model checkpoints
    if frame_number % 100000 == 0:
        torch.save(net.state_dict(), "content/Model_weights/model_lr_00001_gamma_99_rainbow.pt")
        torch.save(target_net.state_dict(), "content/Model_weights/target_model_lr_00001_gamma_99_rainbow.pt")
        torch.save(optimizer.state_dict(), "content/Model_weights/optimizer_lr_00001_gamma_99_rainbow.pt")
        with open("content/Model_weights/buffer_lr_00001_gamma_99_rainbow.pkl", "wb") as f:
            pickle.dump(buffer, f)
        with open("content/Model_weights/total_rewards_lr_00001_gamma_99_rainbow.pkl", "wb") as f:
            pickle.dump(total_rewards, f)

    # Increment frame counter and decay epsilon
    frame_number += 1

    # Detect reward plateau and inject noise into sigma weights
    if is_plateau(total_rewards, plateau_threshold, plateau_length) and not noise_active:
        print(f"Plateau detected at frame {frame_number}. Introducing noise...")
        introduce_noise(net, noise_factor=noise_factor)
        noise_active = True
        noise_start_frame = frame_number

    # Decay the added noise gradually
    if noise_active and frame_number - noise_start_frame > noise_frames:
        for name, param in net.named_parameters():
            if "sigma_weight" in name or "sigma_bias" in name:
                param.data *= noise_decay  # Gradual decay of noise
        noise_active = False
        print(f"Noise decayed and reset at frame {frame_number}.")

    # Perform an environment step using the agent
    reward = agent.step(net, target_net, device=device)

    # Log and print rewards
    if reward is not None:
        total_rewards.append(reward)
        mean_reward = np.mean(total_rewards[-NUMBER_OF_REWARDS_TO_AVERAGE:])
        tbar.set_description(f"Frame:{frame_number} | Total games:{len(total_rewards)} | Mean reward: {mean_reward:.3f})")
        wandb.log({"reward_100": mean_reward, "reward": reward, "episode": len(total_rewards)}, step=frame_number)

        # Check if the environment is solved
        if mean_reward > MEAN_REWARD_BOUND:
            print(f"SOLVED in {frame_number} frames and {len(total_rewards)} games")
            break

    # Skip training until replay buffer is full
    if len(buffer) < EXPERIENCE_REPLAY_SIZE:
        continue

    # Sample a batch of experiences
    states, actions, rewards, dones, next_states, weights = buffer.sample(BATCH_SIZE)
    states, actions, rewards, dones, next_states, weights = (
        states.to(device), actions.to(device), rewards.to(device),
        dones.to(device), next_states.to(device), weights.to(device)
    )

    # Compute loss
    loss = compute_loss(net, target_net, states, actions, rewards, dones, next_states, gamma=GAMMA, criterion=criterion, device=device)
    loss = (loss * weights).mean()

    # Add sigma regularization for noisy layers
    sigma_penalty = 1e-5
    sigma_loss = sum(
        param.abs().sum()
        for name, param in net.named_parameters()
        if "sigma_weight" in name or "sigma_bias" in name
    )
    loss += sigma_penalty * sigma_loss

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Step the learning rate scheduler
    scheduler.step(mean_reward)

    # Synchronize the target network periodically
    if frame_number % SYNC_TARGET_NETWORK == 0:
        target_net.load_state_dict(net.state_dict())

wandb.finish()

# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------