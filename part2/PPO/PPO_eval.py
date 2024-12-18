from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
import ale_py
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder

import imageio

import numpy as np
import torch
from tqdm import tqdm
import random

import sys
import yaml

if __name__ == "__main__":
    if len(sys.argv) < 2:
       print("Please provide the name of the yaml file")
       exit()

    # Getting the name of the yaml file
    name_yaml = sys.argv[1]
    
    # Loading the setup
    config = yaml.safe_load(open(f"./Setups/{name_yaml}.yaml"))

    # Generating multiple environments running in parallel for the evaluation
    env = make_atari_env(config["env_name"], n_envs=config["Environment"]["n_envs"], wrapper_kwargs={"frame_skip": config["Environment"]["frame_skip"], "terminal_on_life_loss": False})
    env = VecFrameStack(env, n_stack=config["Environment"]["n_stack"])


    # Using the parameters from the config file to create the model
    if config["ModelParams"]["policy_kwargs"] == "None":
        config["ModelParams"]["policy_kwargs"] = {}
    
    # Loading the model
    model = PPO.load(f"./Runs/{name_yaml}/{name_yaml}")
    
    print("Model loaded. \n Evaluating.")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

    # Save the evaluation
    with open(f"./Runs/{name_yaml}/eval.txt", "w") as f:
        f.write(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
    
    # Generate a single environment for the video
    env = make_atari_env(config["env_name"], n_envs=1, wrapper_kwargs={"frame_skip": config["Environment"]["frame_skip"], "terminal_on_life_loss": False})
    env = VecFrameStack(env, n_stack=config["Environment"]["n_stack"])
    
    # Video wrapper
    env = VecVideoRecorder(env, f"./Runs/{name_yaml}/videos", record_video_trigger=lambda x: x == 0, video_length=10000, name_prefix=name_yaml)

    # Generate the video
    accum_reward = 0
    obs = env.reset()
    for iter in range(10000):
        action, _ = model.predict(obs, deterministic=False)
        obs, rew, done, _ = env.step(action)
        accum_reward += rew
        if done:
            break

    env.close()
    

