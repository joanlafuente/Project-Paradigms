# Import necessary libraries
import gymnasium as gym
from gymnasium.spaces import Box
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torchsummary import summary
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
from gymnasium.wrappers import MaxAndSkipObservation, GrayscaleObservation, FrameStackObservation, ReshapeObservation
import ale_py
from stable_baselines3 import PPO
# import gym
from huggingface_sb3 import load_from_hub
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

# ---------------------------------------------------------------------------------------------------------------

'''Utils'''

def save_summary_to_file(model, input_size, file_path, title="Model Summary"):
    # Redirect stdout to a file
    import sys
    with open(file_path, "w") as f:
        sys.stdout = f  # Redirect to the file
        print(title)
        summary(model, input_size=input_size)  # Generate the summary
        sys.stdout = sys.__stdout__  # Reset stdout to the default

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.env.reset()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previnfo = info.copy()
        observation, _, _, _, info = self.env.step(1)
        return observation, info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        if (info["lives"] < self.previnfo["lives"]) and (info["lives"] > 0):
            observation, _, _, _, info = self.env.step(1)
        self.previnfo = info.copy()
        return observation, reward, done, truncated, info

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape=84):
        gym.ObservationWrapper.__init__(self, env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + (env.observation_space.shape[-1], )
        self.observation_space = Box(0.0, 1.0, obs_shape, dtype=np.float32)
    
    def reset(self):
        observation, info = self.env.reset()
        observation = self._observation(observation)
        return observation, info
    
    def step(self, action): 
        observation, reward, done, truncated, info = self.env.step(action)
        observation = self._observation(observation)
        return observation, reward, done, truncated, info

    def _observation(self, observation):
        observation = cv2.resize(observation, self.shape, cv2.INTER_AREA)
        return observation

# ---------------------------------------------------------------------------------------------------------------

class ScaledFloatFrame(gym.ObservationWrapper):
    def reset(self):
        observation, info = self.env.reset()
        return np.array(observation).astype(np.float32)/255.0, info
    
    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        return np.array(observation).astype(np.float32) / 255.0, reward, done, truncated, info

def make_env(env_name):

    env = gym.make(env_name, obs_type="grayscale") # Create environment with grayscale observations
    print("Standard Env.        : {}".format(env.observation_space.shape))
    
    env = MaxAndSkipObservation(env, skip=4) # Skip 4 frames
    print("MaxAndSkipObservation: {}".format(env.observation_space.shape))
    
    env = ResizeObservation(env, (84, 84)) # Resize observation to 84x84
    print("ResizeObservation    : {}".format(env.observation_space.shape))
    
    env = FrameStackObservation(env, stack_size=4) # Stack 4 frames
    print("FrameStackObservation: {}".format(env.observation_space.shape))
    
    # Apply ScaledFloatFrame last to ensure correct scaling
    env = ScaledFloatFrame(env) # Scale observation to [0, 1]
    print("ScaledFloatFrame     : {}\n".format(env.observation_space.shape))

    # Apply FireResetEnv to handle episode resets
    env = FireResetEnv(env) # Reset environment after losing a life
    print("FireResetEnv         : {}\n".format(env.observation_space.shape))
    
    return env

# ---------------------------------------------------------------------------------------------------------------

# --- Rainbow DQN components ---
class NoisyLayer(nn.Module):

    # -- Initialize the NoisyLayer --
    def __init__(self, in_features, out_features):

        super(NoisyLayer, self).__init__()
        self.in_features = in_features # Number of input features
        self.out_features = out_features # Number of output features
        
        # Initialize the mean and standard deviation for weights
        self.mu_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.zeros(out_features, in_features))
        
        # Initialize the mean and standard deviation for biases
        self.mu_bias = nn.Parameter(torch.zeros(out_features))
        self.sigma_bias = nn.Parameter(torch.zeros(out_features))

    # Initialize the noise for weights and biases
    def forward(self, x):
        # Generate random noise (epsilon) with the same shape as the weights and biases
        epsilon_w = torch.randn_like(self.mu_weight)  # Gaussian noise for weights
        epsilon_b = torch.randn_like(self.mu_bias)    # Gaussian noise for biases
        
        # Compute the noisy weights and biases by adding noise to the learned parameters
        weight = self.mu_weight + self.sigma_weight * epsilon_w
        bias = self.mu_bias + self.sigma_bias * epsilon_b
        
        # Apply the noisy weights and biases to the input x using a linear transformation
        return F.linear(x, weight, bias)

# -- Rainbow DQN model --   
class RainbowDQN(nn.Module):

    # -- Initialize the Rainbow DQN model --
    def __init__(self, input_shape, num_actions, num_atoms=51, v_min=-10.0, v_max=10.0):
        super(RainbowDQN, self).__init__()
        
        self.num_actions = num_actions # Number of actions
        self.num_atoms = num_atoms # Number of atoms for distributional RL - C51 - Catergorical DQN
        self.v_min = v_min # Minimum value for the q value distribution
        self.v_max = v_max # Maximum value for the q value distribution
        self.delta_z = (v_max - v_min) / (num_atoms - 1) # Delta z for distribution, spacing between each atom in q value distribution
        
        # Define the convolutional layers - [Feature extractor]
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
    
    # -- Forward pass of the Rainbow DQN model --
    def forward(self, x):

        features = self.features(x) # Extract features from the input
        # print(f"Features: {features.shape}")
        fc_output = self.fc(features) # Pass the features through the fully connected layer
        # print(f"FC Output: {fc_output.shape}")

        # Compute the value for each atom
        value = self.value(fc_output).view(-1, 1, self.num_atoms)  # (batch_size, 1, num_atoms)
        # print(f"Value: {value.shape}")
        
        # Compute the advantage for each atom-action pair
        advantage = self.advantage(fc_output).view(-1, self.num_actions, self.num_atoms)  # (batch_size, num_actions, num_atoms)
        # print(f"Advantage: {advantage.shape}")

        # Combine value and advantage to get Q-values
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        # print(f"Q-Values: {q_values.shape}")

        return q_values

    # -- Get the distribution of the Q-values --
    def get_distribution(self, state):

        q_values = self.forward(state) # Get Q-values from the forward pass
        dist = F.softmax(q_values, dim=2)  # Softmax over atoms

        return dist

    # -- Get the Q-value for a given state-action pair --
    def get_q_value(self, state, action):

        dist = self.get_distribution(state)
        batch_size = dist.size(0)

        return torch.sum(dist * self.get_atoms().view(1, 1, -1), dim=2)  # Sum over atoms to get the Q-value for that action at the given state

    # -- Get the atoms for the distribution --
    def get_atoms(self):

        return torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)

# ---------------------------------------------------------------------------------------------------------------

# Define namedtuple for transitions
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'done', 'next_state'])

# --- Prioritized Experience Replay ---
class PrioritizedReplayBuffer:

    # -- Initialize the PER buffer --
    def __init__(self, capacity, alpha=0.8):

        '''
        Args:
            capacity (int): Total capacity of the replay buffer
            alpha (float): Degree of prioritization (0 = uniform sampling, 1 = full prioritization)
        '''

        # -- Attributed of the PER buffer --
        self.capacity = capacity # Total capacity of the replay buffer
        self.buffer = [] # List to store the transitions
        self.priorities = np.zeros((capacity,), dtype=np.float32) # Array to store the priorities of the transitions
        self.alpha = alpha # Degree of prioritization
        self.pos = 0 # Position to store the next transition

    # -- Method to store transitions in the buffer --
    def append(self, transition, td_error):

        '''
        Args:
            transition (tuple): Tuple containing the transition (state, action, reward, done, next_state)
            td_error (float): TD-error of the transition
        '''

        max_priority = self.priorities.max() if self.buffer else 1.0 # Get the maximum priority of the buffer

        if len(self.buffer) < self.capacity: # If buffer is not full, append the transition -> Initial buffer population
            self.buffer.append(transition)
        else: # If buffer is full, overwrite the transition -> Prioritized experience replay
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_priority if td_error is None else abs(td_error) # Store the priority of the transition
        self.pos = (self.pos + 1) % self.capacity # Update the position for the next transition append

    # -- Method to sample transitions from the buffer --
    def sample(self, batch_size, beta=0.4):

        '''
        Args:
            batch_size (int): Number of transitions to sample
            beta (float): Importance sampling weight correction
        '''

        priorities = self.priorities[:len(self.buffer)] # Get the priorities of the stored transitions
        probabilities = priorities ** self.alpha # Compute the probabilities of the transitions based on the priorities * the degree of prioritization
        probabilities /= probabilities.sum() # Normalize the probabilities

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities) # Sample transitions based on the probabilities
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta) # Compute the importance sampling weights
        weights /= weights.max() # Normalize the weights by the maximum weight

        transitions = [self.buffer[idx] for idx in indices] # Get the sampled transitions
        batch = Transition(*zip(*transitions)) # Unzip the transitions

        return batch, torch.tensor(weights, dtype=torch.float32), indices

    # -- Method to update the priorities of the transitions --
    def update_priorities(self, indices, priorities):

        '''
        Args:
            indices (list): List of indices of the transitions in the buffer
            priorities (list): List of updated priorities of the transitions
        '''

        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    # -- Method to get the number of stored transitions --
    def __len__(self):
        return len(self.buffer)  # Return the number of stored transitions

# ---------------------------------------------------------------------------------------------------------------

# --- Training function with all Rainbow features ---
def train(env, student_model, teacher_model, buffer, optimizer, device, num_episodes=500, batch_size=32, gamma=0.99, target_update_freq=1000, plateau_window=30, plateau_threshold=0.01, n_steps=3):
    
    student_model.train()  # Set model to training mode

    target_net = deepcopy(student_model)  # Initialize target network with the same weights
    target_net.eval()  # Set target network to evaluation mode

    epsilon_start, epsilon_end, epsilon_decay = 0.5, 0.1, 2000  # -- Epsilon-greedy parameters --
    epsilon = epsilon_start  # Initialize epsilon
    frame_idx = 0 

    # TensorBoard writer
    writer = SummaryWriter(log_dir="runs/rainbow_dqn")

    # Initialize ReduceLROnPlateau scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7)  # Reduce learning rate on plateau

    mean_rewards = []

    for episode in range(num_episodes):

        state, _ = env.reset()
        state = torch.tensor(state, device=device).squeeze(-1).unsqueeze(0).float()  # Shape: (1, 4, 84, 84) & pass to tensor in device

        episode_reward = 0
        done = False
        n_step_buffer = deque(maxlen=n_steps)  # n-step buffer

        # Compute the 20% episode threshold to start epsilon decay
        epsilon_decay_start_episode = int(num_episodes * 0.2)

        while not done:

            frame_idx += 1
            
            # Start epsilon decay after the first 20% of episodes
            if episode >= epsilon_decay_start_episode:
                epsilon = max(epsilon_end, epsilon_start - frame_idx / epsilon_decay)  # Linearly decay epsilon

            # -- Epsilon-greedy exploration --
            if random.random() > epsilon:

                # print("Action Selection")

                # Get Q-values and compute the action with the highest value
                q_values = student_model(state)  # Shape: (batch_size, num_actions, num_atoms)
                # out_teacher = teacher_model.predict(state)
                # print(out_teacher)
                action = q_values.mean(dim=2).argmax(dim=1).item()  # Take the argmax across actions, averaged across atoms

            else:

                # print("Random Action")

                action = random.randint(0, env.action_space.n - 1)  # Choose a random action
                # out_teacher = teacher_model.predict(state)
                # print(out_teacher)

            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.tensor(next_state, device=device).squeeze(-1).unsqueeze(0).float()  # Shape: (1, 4, 84, 84)

            # Store experience in n-step buffer
            n_step_buffer.append((state, action, reward, done, next_state))
            state = next_state
            episode_reward += reward

            if len(n_step_buffer) == n_steps: # N-step transition is complete !! -> Store in replay buffer

                # Compute multi-step return
                total_reward = sum([r[2] * (gamma ** idx) for idx, r in enumerate(n_step_buffer)]) # Formula: Compute the n-step return
                done_mask = n_step_buffer[-1][3] # Get the done mask from the last transition
                buffer.append((state, action, total_reward, done_mask, next_state), None) # Append the n-step transition to the buffer

                print("N-Step Transition")

            if len(buffer) >= batch_size: # If buffer has enough samples, sample a batch and perform a gradient step
                
                batch, weights, indices = buffer.sample(batch_size)

                # Prepare batched tensors
                states = torch.cat(batch.state).squeeze(-1).float()
                actions = torch.tensor(batch.action, device=device).unsqueeze(1)
                rewards = torch.tensor(batch.reward, device=device)
                dones = torch.tensor(batch.done, device=device).float()
                next_states = torch.cat(batch.next_state).squeeze(-1).float()

                # Compute Q-values using Double DQN
                q_values = student_model(states)
                next_q_values = target_net.get_distribution(next_states)  # get_distribution gives probability over atoms
                next_actions = q_values.argmax(1, keepdim=True)  # Double DQN action selection
                next_q_values = next_q_values.gather(1, next_actions)  # Get the next state's q-values for the selected action

                # Compute the target for multi-step (n-step) updates
                rewards_expanded = rewards.unsqueeze(1).unsqueeze(2).expand(-1, student_model.num_actions, student_model.num_atoms)
                dones_expanded = dones.unsqueeze(1).unsqueeze(2).expand(-1, student_model.num_actions, student_model.num_atoms)
                targets = rewards_expanded + gamma * (1 - dones_expanded) * next_q_values

                dist = student_model.get_distribution(states)
                action_dist = dist.gather(1, actions.unsqueeze(-1).expand(-1, -1, student_model.num_atoms))
                loss = -(targets * torch.log(action_dist + 1e-6)).sum(dim=2).mean()

                # -- Backpropagate the loss --
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print("Gradient Step")

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

    # ----------------------------------------

    # -- Hyperparameters --

    episodes = 10000 # Number of episodes
    batch_size = 32 # Batch size

    gamma = 0.99 # Discount factor
    buffer_capacity = 100000 # Replay buffer capacity
    alpha_PER = 0.8 # PER degree of prioritization
    target_update_freq = 1000 # Target network update frequency
    n_steps = 3 # Number of steps for multi-step return

    plateau_window = 30 # Plateau detection window for learning rate scheduler
    plateau_threshold = 0.01 # Plateau detection threshold 

    # ----------------------------------------

    # -- Set device --
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # ----------------------------------------

    # -- Environment --
    env_name = "Breakout-v4"
    print(f"\nEnvironment: {env_name}\n")

    # -- Create Environment --
    env = make_env(env_name)

    print(env.observation_space)

    sys.exit()

    # -- Get input shape and number of actions --
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n

    print(f"Observation Space: {input_shape}")
    print(f"Action Space: {num_actions}\n")
    
    '''
    env = make_atari_env('BreakoutNoFrameskip-v4')
    env = VecFrameStack(eval_env, n_stack=4)

    # -- Get input shape and number of actions --
    input_shape = eval_env.observation_space.shape
    input_shape = (input_shape[-1],) + input_shape[:-1] # Change shape to ({frame_stack}, {height}, {width})
    num_actions = eval_env.action_space.n

    print(f"Observation Space: {input_shape}")
    print(f"Action Space: {num_actions}\n")
    '''

    # ----------------------------------------

    # -- Initialize student model --
    rainbow_dqn_model = RainbowDQN(input_shape, num_actions).to(device)

    # -- Initialize teacher model --
    checkpoint = load_from_hub(repo_id="mrm8488/a2c-BreakoutNoFrameskip-v4", filename="a2c-BreakoutNoFrameskip-v4.zip")
    teacher_model = PPO.load(checkpoint)

    # -- Print model summaries & arquitectures --

    '''
    student_model_summary_path = "rainbow_dqn_model_summary.txt"
    teacher_model_summary_path = "ppo_teacher_model_summary.txt"

    dummy_input = torch.tensor(input_shape)
    save_summary_to_file(
        model=rainbow_dqn_model,
        input_size=dummy_input, 
        file_path=student_model_summary_path,
        title="Summary of Student Model (RainbowDQN)"
    )

    with open(teacher_model_summary_path, "w") as f:
        f.write("Summary of Teacher Model (PPO):\n\n")
        f.write(str(teacher_model.policy) + "\n")
        # Optionally add a detailed summary
        f.write("\nDetailed Summary:\n")
        sys.stdout = f  # Redirect to the file
        summary(teacher_model.policy, input_size=(1,) + dummy_input)
        sys.stdout = sys.__stdout__  # Reset stdout

    sys.exit()
    '''

    # ----------------------------------------

    # Replay buffer and optimizer
    buffer = PrioritizedReplayBuffer(capacity=buffer_capacity, alpha=alpha_PER)
    optimizer = optim.Adam(rainbow_dqn_model.parameters(), lr=1e-4)

    # Train the agent
    train(
        env, # Environment
        rainbow_dqn_model, # Student model: Rainbow DQN
        teacher_model, # Teacher model
        buffer, 
        optimizer, 
        device, 
        num_episodes=episodes, 
        batch_size=batch_size, 
        gamma=gamma, 
        target_update_freq=target_update_freq, 
        plateau_window=plateau_window, 
        plateau_threshold=plateau_threshold, 
        n_steps=n_steps
        )

               
