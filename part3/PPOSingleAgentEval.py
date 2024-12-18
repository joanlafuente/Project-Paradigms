import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from stable_baselines3 import PPO
import gymnasium as gym
import ale_py
from stable_baselines3.common.env_util import make_vec_env
import supersuit as ss
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder

import imageio

import numpy as np
import torch
from tqdm import tqdm
import random

import sys
import yaml

class Wrapper(gym.Wrapper):
    """
    Wrapper for the environment using the supersuit library.
    We use the predefine wrapers for this part of the project.
    """
    def __init__(self, env, frame_stack):
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, frame_stack, stack_dim=0)
        env = ss.dtype_v0(env, dtype=np.float32)
        env = ss.normalize_obs_v0(env, env_min=0, env_max=1)

        super().__init__(env)


if __name__ == "__main__":
    if len(sys.argv) < 2:
       print("Please provide the name of the yaml file")
       exit()

    # Getting the name of the yaml file
    name_yaml = sys.argv[1]
    # name_yaml = "Pacman2"
    
    # Loading the setup
    config = yaml.safe_load(open(f"./Setups/{name_yaml}.yaml"))

    # Generating the vector environment (multiple environments running in parallel)
    env = make_vec_env(config["env_name"], n_envs=config["Environment"]["n_envs"], seed=config["seed"], 
                    wrapper_class=Wrapper,
                    wrapper_kwargs={"frame_stack": config["Environment"]["frame_stack"]}, 
                    env_kwargs={"mode": config["Environment"]["mode"], "difficulty": config["Environment"]["difficulty"]})


    # Using the parameters from the config file to create the model
    if config["ModelParams"]["policy_kwargs"] == "None":
        config["ModelParams"]["policy_kwargs"] = {}
    
    # Loading the model
    model = PPO.load(f"./Runs/{name_yaml}/{name_yaml}")
    
    print("Model loaded. \n eEvaluating.")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, deterministic=False)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

    # Saving the evaluation
    with open(f"./Runs/{name_yaml}/eval.txt", "w") as f:
        f.write(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

    # Generating a single enviroment to make the video
    env = make_vec_env(config["env_name"], n_envs=1, seed=config["seed"], 
                wrapper_class=Wrapper,
                wrapper_kwargs={"frame_stack": config["Environment"]["frame_stack"]}, 
                env_kwargs={"mode": 0, "difficulty": config["Environment"]["difficulty"]})
    env = VecVideoRecorder(env, f"./Runs/{name_yaml}/videos", record_video_trigger=lambda x: x == 0, video_length=10000, name_prefix=name_yaml)

    # Generate the video of the agent playing
    accum_reward = 0
    obs = env.reset()
    for iter in range(10000):
        action, _ = model.predict(obs, deterministic=False)
        obs, rew, done, _ = env.step(action)
        accum_reward += rew
        if done:
            break

    env.close()
    

