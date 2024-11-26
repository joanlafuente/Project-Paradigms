import gymnasium as gym
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from collections import namedtuple, deque
import random
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from gymnasium.wrappers import MaxAndSkipObservation, ResizeObservation, FrameStackObservation
import ale_py
from stable_baselines3 import DQN
import cv2
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

class GrayScaleObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(
            low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8
        )

    def observation(self, observation):
        gray_obs = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return np.expand_dims(gray_obs, axis=-1)

# Define namedtuple for transitions
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'done', 'next_state'])

# Rainbow DQN components
class RainbowDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(RainbowDQN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512), nn.ReLU()
        )
        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, num_actions)

    def forward(self, x):
        features = self.features(x)
        fc_output = self.fc(features)
        value = self.value(fc_output)
        advantage = self.advantage(fc_output)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

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

# Teacher Guidance (Hugging Face)
def load_teacher_model():
    checkpoint = load_from_hub(repo_id="kuross/dqn-Breakout-v4", filename="dqn-Breakout-v4.zip")
    model = PPO.load(checkpoint)
    return model

def preprocess_env(env_name):

    env = gym.make(env_name)
    env = MaxAndSkipObservation(env, skip=4)
    env = ResizeObservation(env, (84, 84))
    env = GrayScaleObservation(env)  # Use the custom grayscale wrapper
    env = FrameStackObservation(env, stack_size=4)

    return env

import torch.nn.functional as F

def train(env, student_model, teacher_model, buffer, optimizer, device, num_episodes=500, batch_size=32, gamma=0.99, plateau_threshold=5, window_size=30):
    
    student_model.train()
    target_net = deepcopy(student_model)
    target_net.eval()

    target_update_freq = 1000
    epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.1, 2000
    epsilon = epsilon_start
    frame_idx = 0

    mean_rewards = []
    plateau = False

    # TensorBoard writer
    writer = SummaryWriter(log_dir="runs/rainbow_dqn_with_teacher")

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, device=device).squeeze(-1).unsqueeze(0).float()  # Shape: (1, 4, 84, 84)
        episode_reward = 0
        done = False

        while not done:
            frame_idx += 1
            epsilon = max(epsilon_end, epsilon_start - frame_idx / epsilon_decay)

            # Epsilon-greedy exploration
            if random.random() > epsilon:
                action = student_model(state).argmax(1).item()
            else:
                action = random.randint(0, env.action_space.n - 1)

            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.tensor(next_state, device=device).squeeze(-1).unsqueeze(0).float()  # Shape: (1, 4, 84, 84)
            buffer.append((state, action, reward, done, next_state), None)
            state = next_state
            episode_reward += reward

            if len(buffer) >= batch_size:
                batch, weights, indices = buffer.sample(batch_size)

                # Prepare batched tensors
                states = torch.cat(batch.state).squeeze(-1).float()  # Shape: [batch_size, 4, 84, 84]
                actions = torch.tensor(batch.action, device=device).unsqueeze(1)  # Shape: [batch_size, 1]
                rewards = torch.tensor(batch.reward, device=device)  # Shape: [batch_size]
                dones = torch.tensor(batch.done, device=device).float()  # Convert dones to float
                next_states = torch.cat(batch.next_state).squeeze(-1).float()  # Shape: [batch_size, 4, 84, 84]

                # Compute Q-values
                q_values = student_model(states).gather(1, actions).squeeze(1)
                next_q_values = target_net(next_states).max(1)[0]
                expected_q_values = rewards + gamma * next_q_values * (1 - dones)

                # Compute DQN loss
                dqn_loss = (weights * (q_values - expected_q_values.detach()) ** 2).mean()

                # If plateau detected, add teacher guidance
                if plateau:
                    with torch.no_grad():
                        teacher_probs = F.softmax(teacher_model.policy(states), dim=1)  # Teacher action probabilities
                    student_logits = student_model(states)
                    student_probs = F.softmax(student_logits, dim=1)
                    distillation_loss = F.kl_div(student_probs.log(), teacher_probs, reduction="batchmean")
                    dqn_loss += distillation_loss  # Combine with DQN loss

                optimizer.zero_grad()
                dqn_loss.backward()
                optimizer.step()

                # Log loss to TensorBoard
                writer.add_scalar("Loss", dqn_loss.item(), frame_idx)

            if frame_idx % target_update_freq == 0:
                target_net.load_state_dict(student_model.state_dict())

        # Log episode reward
        writer.add_scalar("Episode Reward", episode_reward, episode)
        print(f"Episode {episode}, Reward: {episode_reward:.2f}")

        # Monitor mean rewards and detect plateau
        mean_rewards.append(episode_reward)
        if len(mean_rewards) >= window_size:
            recent_mean = sum(mean_rewards[-window_size:]) / window_size
            if len(mean_rewards) > window_size:
                prev_mean = sum(mean_rewards[-window_size * 2 : -window_size]) / window_size
                if abs(recent_mean - prev_mean) < plateau_threshold:
                    plateau = True
                    print(f"Plateau detected at episode {episode}, enabling teacher guidance.")
            mean_rewards = mean_rewards[-window_size:]  # Keep the last `window_size` rewards

    writer.close()

# -- Main --
if __name__ == "__main__":

    # -- Set device --
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    env_name = "Breakout-v4"
    # env = gym.make(env_name, render_mode="rgb_array")
    env = preprocess_env(env_name)

    print(f"\nEnvironment: {env_name}")
    print(f"Observation Space: {env.observation_space.shape}")
    print(f"Action Space: {env.action_space.n}")

    # Preprocess the environment
    input_shape = (4, 84, 84, 3)
    num_actions = env.action_space.n

    # -- Initialize student model --
    student_model = RainbowDQN(input_shape, num_actions).to(device)
    
    # -- Initialize teacher model --
    repo_id = "kuross/dqn-Breakout-v4"
    filename = "dqn-Breakout-v4.zip"

    # Load the model from the Hugging Face Hub
    teacher_model_path = load_from_hub(repo_id=repo_id, filename=filename)
    teacher_model = DQN.load(teacher_model_path)

    print(f"\nTeacher Model: {teacher_model}")

    # Replay buffer and optimizer
    buffer = PrioritizedReplayBuffer(100000)
    optimizer = optim.Adam(student_model.parameters(), lr=1e-4)

    # Train the agent
    train(env, student_model, teacher_model, buffer, optimizer, device)
