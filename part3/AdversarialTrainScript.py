from warnings import resetwarnings
from stable_baselines3 import PPO
import torch
import gymnasium as gym
import ale_py
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.env_util import make_vec_env
import supersuit as ss
import numpy as np
from pettingzoo.utils import agent_selector


from gymnasium.spaces import Discrete, Box

from pettingzoo.atari import pong_v3


import sys
import yaml
import os

class MultiAgentPong(gym.Env):
    """
    Environment for the adversarial training.
    This environment is baseon on the pettingzoo library.

    This enviroment allows the training of a model vs an already trained model as an adversary.
    """
    def __init__(self, model=None, agent_id=None):
        """
        Args:
            model: The model that will be used as oponent for the adversarial training.
            agent_id: The id of the agent that will not be trained.
        """
        # Wrapper for the environment using the supersuit library.
        # We use the predefine wrapers for this part of the project.
        env = pong_v3.env(num_players=2)
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 4, stack_dim=0)
        env = ss.dtype_v0(env, dtype=np.float32)
        env = ss.normalize_obs_v0(env, env_min=0, env_max=1)

        # Storing the necesary variables
        self.pong_env = env
        self.model = model
        self.agent_id = agent_id

        # Defining the action and observation space
        self.action_space = Discrete(6)
        self.observation_space = Box(0, 1, shape=(4, 84, 84), dtype=np.float32)

        # Getting the possible agents
        self.agents = self.pong_env.possible_agents
        print(self.agents)
        # Obtaining the id of the agent to train
        if agent_id is not None:
            if agent_id == 0:
                self.training_agent_id = 1
            else:
                self.training_agent_id = 0

        # Creating the agent selector to change between the agents
        self.agent_selector = agent_selector(self.agents)

    def reset(self, **kwargs):
        self.pong_env.reset()
        obs, reward, termination, truncation, info = self.pong_env.last()
        return obs, info

    def step(self, action):
        """
        Function to make a step in the environment.
        Args:
            action: The action to take in the environment, by the model being trained.
        """
        # Iterating over the two agents
        reward2return = 0
        for _ in range(2):
            # Selecting the agent
            agent = self.agent_selector.next()

            # Getting the observation and information about the environment
            obs, reward, termination, truncation, info = self.pong_env.last()

            # Making the step in the environment if the game is not over
            if not (termination or truncation):
                if agent == self.agents[0]: 
                    # If the agent is the already trained model, use the model to make the prediction
                    # Otherwise use the action given as input
                    if self.agent_id == 0:
                        act = self.model.predict(obs)[0]
                    else:
                        act = action
                else:
                    # If the agent is the already trained model, use the model to make the prediction
                    # Otherwise use the action given as input
                    if self.agent_id == 1:
                        act = self.model.predict(obs)[0]
                    else:
                        act = action

                # Making the step in the environment
                self.pong_env.step(act)

                # Getting the reward of the agent being trained
                reward2return += self.pong_env.rewards[self.agents[self.training_agent_id]]

        # Getting the observation and information about the environment
        if not (termination or truncation):
            obs2return, _, termination2return, truncation2return, info2return = self.pong_env.last()
        else:
            # If the game is over, return the last observation and information
            obs2return, termination2return, truncation2return, info2return = obs, termination, truncation, info

        return obs2return, reward2return, termination2return, truncation2return, info2return

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please provide the name of the yaml file")
        exit()

    # Getting the name of the yaml file
    name_yaml = sys.argv[1]

    # Checking if the cuda is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Cuda is available: {torch.cuda.is_available()}")

    # Loading the setup
    config = yaml.safe_load(open(f"./Setups/{name_yaml}.yaml"))

    # Using the parameters from the config file to create the model
    if config["ModelParams"]["policy_kwargs"] == "None":
        config["ModelParams"]["policy_kwargs"] = None

    # Creating a dummy environment
    env = MultiAgentPong(model=None, agent_id=None)
    # Making the enviroment vectorized
    env = make_vec_env(MultiAgentPong, n_envs=config["Environment"]["n_envs"])
    
    # Creating the models
    model_0 = PPO(env=env,
                tensorboard_log=config["logs_dir"],
                seed=config["seed"],
                device=device,
                **config["ModelParams"],
                )
    model_1 = PPO(env=env,
                tensorboard_log=config["logs_dir"],
                seed=config["seed"],
                device=device,
                **config["ModelParams"],
                )

    # Store the weight to init the adversarial training
    os.makedirs(f"./Runs/{name_yaml}", exist_ok=True)
    model_0.save(f"./Runs/{name_yaml}/{name_yaml}_0")
    model_1.save(f"./Runs/{name_yaml}/{name_yaml}_1")


    for i in range(config["Adversial"]["n_iter"]):
        # If the iteration is odd, train the model 0
        if ((i % 2) != 0):
            print("Model 0")
            # Loading the model 1 as opponent
            model_1 = PPO.load(f"./Runs/{name_yaml}/{name_yaml}_1")

            # Creating a vectorized environment with the model 1 as opponent
            env = make_vec_env(MultiAgentPong, n_envs=config["Environment"]["n_envs"],
                            wrapper_class=None,
                            wrapper_kwargs=None,
                            env_kwargs={"model": model_1, "agent_id": 1}
                            )                        
            
            # Loading the model 0 to train
            model_0 = PPO.load(f"./Runs/{name_yaml}/{name_yaml}_0",
                env=env,
                tensorboard_log=config["logs_dir"],
                seed=config["seed"],
                device=device,
                **config["ModelParams"],
                )
            
            # Training the model 0 for the specified number of timesteps
            model_0.learn(total_timesteps=config["Adversial"]["timesteps_adv"], log_interval=config["log_interval"], tb_log_name=name_yaml, progress_bar=True)
            # Saving the model 0
            model_0.save(f"./Runs/{name_yaml}/{name_yaml}_0")
        else:
            # If the iteration is even, train the model 1
            print("Model 1")
            # Loading the model 0 as opponent
            model_0 = PPO.load(f"./Runs/{name_yaml}/{name_yaml}_0")
            
            # Creating a vectorized environment with the model 0 as opponent
            env = make_vec_env(MultiAgentPong, n_envs=config["Environment"]["n_envs"],
                            wrapper_class=None,
                            wrapper_kwargs=None,
                            env_kwargs={"model": model_0, "agent_id": 0}
                            )

            # Loading the model 1 to train
            model_1 = PPO.load(f"./Runs/{name_yaml}/{name_yaml}_1",
                env=env,
                tensorboard_log=config["logs_dir"],
                seed=config["seed"],
                device=device,
                **config["ModelParams"],
                )
            
            # Training the model 1 for the specified number of timesteps
            model_1.learn(total_timesteps=config["Adversial"]["timesteps_adv"], log_interval=config["log_interval"], tb_log_name=name_yaml, progress_bar=True)

            # Saving the model 1
            model_1.save(f"./Runs/{name_yaml}/{name_yaml}_1")

        # Saving the model every 5 iterations
        if ((i % 5) == 0):
            model_0.save(f"./Runs/{name_yaml}/{name_yaml}_0_iter_{i}")
            model_1.save(f"./Runs/{name_yaml}/{name_yaml}_1_iter_{i}")
    
    # Closing the environment
    env.close()