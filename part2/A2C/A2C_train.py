# Import necessary libraries
from stable_baselines3 import A2C
# from stable_baselines3.common.policies import CnnPolicy
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
# from stable_baselines3.common.vec_env import SyncVectorEnv
import gymnasium as gym
from ale_py import ALEInterface
import ale_py
import yaml
import numpy as np
import argparse
import os
import sys

"""
# ----------------------------------------------------------------------------

# Custom reward wrapper to modify MsPacman reward

from stable_baselines3.common.vec_env import VecEnvWrapper

class MsPacmanVecRewardWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self.previous_lives = None

    def reset(self):
        obs = self.venv.reset()
        self.previous_lives = self._get_lives()
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        current_lives = self._get_lives()
        
        # Apply custom rewards
        custom_rewards = []
        for i, reward in enumerate(rewards):
            if current_lives[i] < self.previous_lives[i]:
                reward -= 50  # Penalize for losing a life
            custom_rewards.append(reward)
        self.previous_lives = current_lives
        return obs, np.array(custom_rewards), dones, infos

    def _get_lives(self):
        # Access lives for each environment in the vectorized setup
        return [env.unwrapped.ale.lives() for env in self.venv.envs]
        
# ----------------------------------------------------------------------------

"""

def parse_args():
    parser = argparse.ArgumentParser(description="Train an A2C model with MsPacman environment and custom rewards.")
    parser.add_argument('--model_name', type=str, default="A2C", help='Model type to use (default: A2C)')
    parser.add_argument('--env_name', type=str, default="MsPacman", help='Environment name to use')
    parser.add_argument('--yaml_file', type=str, default="MsPacman", help='YAML file to use for training')
    return parser.parse_args()

# ----------------------------------------------------------------------------

if __name__ == "__main__":


    # Register the ALE environment
    ale = ALEInterface()
    # ale.loadROM('/export/fhome/pmlai02/miniconda3/envs/gym/lib/python3.10/site-packages/AutoROM/roms/.bin')  
    gym.register_envs(ale_py)

    # -------------------------------------------------
    # Parse command-line arguments
    args = parse_args()
    name_yaml = args.yaml_file
    config = yaml.safe_load(open(f"/fhome/pmlai02/project_repo/part2/A2C/Setups/{name_yaml}.yaml"))
    os.makedirs("./trained_models", exist_ok=True)
    os.makedirs(f"./trained_models/{name_yaml}", exist_ok=True)

    # -------------------------------------------------
    # Generate multiple environments running in parallel
    raw_env = make_atari_env(
        env_id=config["env_name"],  
        n_envs=config["Environment"]["n_envs"],
        seed=config["seed"],
        wrapper_kwargs={"frame_skip": config["Environment"]["frame_skip"]}
    )
    raw_env = VecFrameStack(raw_env, n_stack=config["Environment"]["n_stack"])

    # raw_env = MsPacmanVecRewardWrapper(raw_env)
    
    # -------------------------------------------------

    # Prepare the config data
    if config["ModelParams"]["policy_kwargs"] == "None":
        config["ModelParams"]["policy_kwargs"] = {}
    elif isinstance(config["ModelParams"]["policy_kwargs"], str):
        # Convert string to dictionary
        config["ModelParams"]["policy_kwargs"] = eval(config["ModelParams"]["policy_kwargs"])
    
    # Whether to use step lr scheduler
    squeduler = config["ModelParams"].pop("lr_squeduler", False)
    if squeduler:
        def lr_squeduler(initial_value: float):
            """
            Step learning rate schedule.

            :param initial_value: Initial learning rate.
            :return: schedule that computes
            current learning rate depending on remaining progress
            """
            def func(progress_remaining: float):
                """
                Progress will decrease from 1 (beginning) to 0.

                Until 0.85 progress remaining, learning rate will be 100% of initial value.
                Until 0.5 progress remaining, learning rate will be 25% of initial value.
                Until 0.3 progress remaining, learning rate will be 10% of initial value.
                Until 0.1 progress remaining, learning rate will be 5% of initial value.

                :param progress_remaining:
                :return: current learning rate
                """

                if 0.85 < progress_remaining:
                    return initial_value
                elif 0.5 < progress_remaining <= 0.85:
                    return initial_value * 0.25
                elif 0.3 < progress_remaining <= 0.5:
                    return initial_value * 0.1
                elif 0.1 < progress_remaining <= 0.3:
                    return initial_value * 0.05
                else:
                    return initial_value * 0.01
            return func
        lr = config["ModelParams"].pop("learning_rate")
        

    else:
        def lr_squeduler(initial_value: float):
            """
            Constant lr squedule.

            :param initial_value: Initial learning rate.
            :return: schedule that computes
            current learning rate depending on remaining progress
            """
            def func(progress_remaining: float):
                """
                Return the initial value of the learning rate.
                """
                return initial_value
            return func
        lr = config["ModelParams"].pop("learning_rate")
    
    # Initialize the A2C model with the config parameters
    A2C_model = A2C(
        env=raw_env,
        tensorboard_log=config["logs_dir"],
        learning_rate=lr_squeduler(lr),
        seed=config["seed"],
        device=config["device"],
        **config["ModelParams"],
        ) 
                
    # -------------------------------------------------
    # Train the model with the specified number of timesteps
    train_timesteps = config["total_timesteps"]
    A2C_model.learn(
        total_timesteps=train_timesteps,
        log_interval=config["log_interval"],
        tb_log_name=name_yaml,
        reset_num_timesteps=True
    )

    # Save the model
    model_path = f"./trained_models/{name_yaml}/{args.model_name}_{args.env_name}"
    A2C_model.save(model_path)

    # Close the environment
    raw_env.close()
