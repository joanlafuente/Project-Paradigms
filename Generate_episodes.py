import gymnasium as gym
from gymnasium.wrappers import MaxAndSkipObservation, ResizeObservation, FrameStackObservation
import ale_py 

import numpy as np
import matplotlib.pyplot as plt

ENV_NAME = "ALE/Breakout-v5"
PATH2SAVE = "/Users/jlafuente/Desktop/Paradigms of machine learning/Episodes"

class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


def make_env(env_name):
    env = gym.make(env_name, obs_type="grayscale", render_mode="rgb_array")
    print("Standard Env.        : {}".format(env.observation_space.shape))
    env = MaxAndSkipObservation(env, skip=2)
    print("MaxAndSkipObservation: {}".format(env.observation_space.shape))
    env = ResizeObservation(env, (84, 84))
    print("ResizeObservation    : {}".format(env.observation_space.shape))
    env = FrameStackObservation(env, stack_size=5)
    print("FrameStackObservation: {}".format(env.observation_space.shape))
    env = ScaledFloatFrame(env)
    print("ScaledFloatFrame     : {}".format(env.observation_space.shape))
    return env

transitions = []
env = make_env(ENV_NAME)
obs = env.reset()
prev_obs = obs
total_reward = 0
for _ in range(5000):
    action = input("Enter action: ")
    if action == "a":
        action = 3
    elif action == "d":
        action = 2
    elif action == "w":
        action = 1
    elif action == "s":
        action = 0
    else:
        action = 0
    plt.close()
    obs, reward, done, truncated, info = env.step(int(action))
    total_reward += reward
    transitions.append((prev_obs, action, reward))
    if done or truncated:
        break
    plt.imshow(env.render())
    plt.axis("off")
    plt.show(block=False)
    prev_obs = obs

env.close()

episode_name = F"reward_{total_reward}_num_episodes_{len(transitions)}"
import pickle
with open("{}/{}.pkl".format(PATH2SAVE, episode_name), "wb") as f:
    pickle.dump(transitions, f)