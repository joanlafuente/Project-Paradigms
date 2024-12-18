from stable_baselines3 import PPO
import gymnasium as gym
import ale_py
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_vec_env
import supersuit as ss
import numpy as np

import sys
import yaml
import os

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

    # Loading the setup
    config = yaml.safe_load(open(f"./Setups/{name_yaml}.yaml"))

    # Generating the environment with the wrappers and the parameters from the config file
    # It consists of multimple environments running in parallel
    env = make_vec_env(config["env_name"], n_envs=config["Environment"]["n_envs"], seed=config["seed"], 
                       wrapper_class=Wrapper,
                       wrapper_kwargs={"frame_stack": config["Environment"]["frame_stack"]}, 
                       env_kwargs={"mode": config["Environment"]["mode"], "difficulty": config["Environment"]["difficulty"]})
    obs = env.reset()

    # Using the parameters from the config file to create the model
    if config["ModelParams"]["policy_kwargs"] == "None":
        config["ModelParams"]["policy_kwargs"] = None

    # Creating the model
    model = PPO(env=env, 
                tensorboard_log=config["logs_dir"],
                seed=config["seed"],
                device=config["device"],
                **config["ModelParams"],
                )
    
    # Training the model for the predefined number of timesteps at the config file
    model.learn(total_timesteps=config["total_timesteps"], log_interval=config["log_interval"], tb_log_name=name_yaml, progress_bar=True)

    # Saving the model
    os.makedirs(f"./Runs/{name_yaml}", exist_ok=True)
    model.save(f"./Runs/{name_yaml}/{name_yaml}")

    # Closing the environment
    env.close()