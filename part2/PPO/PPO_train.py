from stable_baselines3 import PPO
import gymnasium as gym
import ale_py
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

import sys
import yaml
import os


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Please provide the name of the yaml file")
        exit()

    # Getting the name of the yaml file
    name_yaml = sys.argv[1]

    # Loading the setup
    config = yaml.safe_load(open(f"./Setups/{name_yaml}.yaml"))

    # Generating the environment
    env = make_atari_env(config["env_name"], n_envs=config["Environment"]["n_envs"], seed=config["seed"], wrapper_kwargs={"frame_skip": config["Environment"]["frame_skip"]})
    env = VecFrameStack(env, n_stack=config["Environment"]["n_stack"])

    # Using the parameters from the config file to create the model
    if config["ModelParams"]["policy_kwargs"] == "None":
        config["ModelParams"]["policy_kwargs"] = None

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

                Until 0.75 progress remaining, learning rate will be 100% of initial value.
                Until 0.5 progress remaining, learning rate will be 50% of initial value.
                Until 0.3 progress remaining, learning rate will be 25% of initial value.
                Until 0.1 progress remaining, learning rate will be 10% of initial value.
                After 0.1 progress remaining, learning rate will be 5% of initial value.

                :param progress_remaining:
                :return: current learning rate
                """

                if 0.75 < progress_remaining:
                    return initial_value
                elif 0.5 < progress_remaining <= 0.75:
                    return initial_value * 0.5
                elif 0.3 < progress_remaining <= 0.5:
                    return initial_value * 0.25
                elif 0.1 < progress_remaining <= 0.3:
                    return initial_value * 0.1
                else:
                    return initial_value * 0.05
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
                """
                return initial_value
            return func
        lr = config["ModelParams"].pop("learning_rate")
        
    # Initialize the PPO model with the parameters from the config file
    model = PPO(env=env, 
                tensorboard_log=config["logs_dir"],
                seed=config["seed"],
                device=config["device"],
                learning_rate=lr_squeduler(lr),
                **config["ModelParams"],
                )
    
    # Training the model for the predefined number of timesteps at the config file
    model.learn(total_timesteps=config["total_timesteps"], log_interval=config["log_interval"], tb_log_name=name_yaml, progress_bar=True)
    
    # Saving the model
    os.makedirs(f"./Runs/{name_yaml}", exist_ok=True)
    model.save(f"./Runs/{name_yaml}/{name_yaml}")

    env.close()