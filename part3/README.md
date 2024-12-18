# Part 3: Multi Agent Pong

This part of the project focuses on training agents adversially to play the **Pong** game using **PPO**.

---


## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Project Description](#project-description)
3. [How to execute the code](#how-to-execute-the-code)
4. [Results and Comparisons](#results-and-comparisons)

---

## Environment Setup

To install the necessary dependencies, use the `environment.yml` file provided in this repository. The file contains all the libraries required for this project.

### Steps:
1. Install Miniconda or Anaconda.
2. Run the following command to create the environment:
   ```bash
   conda env create -f environment_part3.yml
    ```
3. Activate the environment:
   ```bash
   conda activate environment_part3
    ```
# Project Description

This project aims to train adversially reinforcement learning agents to play the **Pong** game using stable baselines3PPO.

The agents are trained to maximize the reward by scoring more points than the opponent.

---

# How to execute the code

## Single Agent

To train and evaluate an agent with the default gymnasium opponent it can be done with the files ``PPOSingleAgentTrain.py`` and ``PPOSingleAgentEval.py``.

- ``PPOSingleAgentTrain.py``: To train an agent vs the default oponent at gymnasium Pong.
- ``PPOSingleAgentEval.py``: To evaluate an agent vs the default oponent at gymnasium Pong.

Both files require a setup name of ``.yaml`` situated on the ``/Setups`` folder.

For instance, to train the PPO model with the best hyperparamers out of the ones tested is:

```bash
   conda activate environment_part3
   python PPOSingleAgentTrain.py Pong_baseline
```

The training sript will log on a tensorboard file located at ``/log_dir``, and will store the trained weights on ``/Runs/name_yaml``. 

The evaluation script will store a video and the evaluation statistics at ``/Runs/name_yaml``.

## Multi Agent
To train and evaluate an agent with our adversial approach on the petting zoo environment, can be done with the following files ``AdversarialTrainScript.py``, ``PPOEvalAdversarialSingleAgent.py`` and ``MultiAgentEvaluation.ipynb``.

- ``AdversarialTrainScript.py``: To train the right and left agents using our adversial approach.
-  ``PPOEvalAdversarialSingleAgent.py``: To evaluate a trained right agent vs the default oponent at gymnasium Pong.
- ``MultiAgentEvaluation.ipynb``:  To evaluate and Generate videos of matches between the trained right and left agents.

Both ``.py`` files require a setup name of ``.yaml`` situated on the ``/Setups`` folder. We provide one example in ``Pong_baseline_adversarial_v5.yaml``.

For instance, to train a PPO model on the adversarial aproach you can run:

```bash
   conda activate environment_part3
   python AdversarialTrainScript.py Pong_baseline_adversarial_v5
```

The training sript will log on a tensorboard file located at ``/log_dir``, and will store the trained weights on ``/Runs/name_yaml``. 

The evaluation script will store a video and the evaluation statistics at ``/Runs/name_yaml``.

---

## Trained Agents

Note that the best model weights can be seen at Pong_baseline_adversarial_v5_0.zip & Pong_baseline_adversarial_v5_1.zip 

## Visual Demonstration

At ``MultiAgentPong.mp4`` can be seen an example of a match between our two agents trained adversially.

---

## Key Observations

### Adversarial Training
- **Dynamic Learning**: Both agents improved iteratively by competing against each other, resulting in longer and more complex rallies.
- **Strategy Development**: Agents developed advanced techniques such as:
   - Increasing ball speed by hitting it with paddle edges.
   - Predicting opponent movements for better positioning.
   - Manipulating ball trajectories to make it harder for the opponent to respond.

### Generalization
- The adversarially trained agents generalized well to diverse scenarios, unlike agents trained only against the default opponent.

---

## Results Summary

### Comparison of Results

| **Metric**                         | **Single Agent (Default Opponent)** | **Adversarial Agents**        |
|------------------------------------|------------------------------------|--------------------------------|
| **Average Reward**                 | 19.4                               | -4.6 (vs Default Opponent)     |
| **Win Rate (Adversarial Matches)** | N/A                                | **90%** (Left Agent Wins)      |
| **Average Points Scored**          | 19.4                               | 20.5 (Left) / 14.92 (Right)    |

- **Single Agent**: Trained to exploit the weaknesses of the default opponent, achieving high scores.
- **Adversarial Agents**: Demonstrated competitive and strategic gameplay but required further fine-tuning for general performance.

---

## Conclusion

This section implemented adversarial training for **Pong** using **Proximal Policy Optimization (PPO)**:
- Agents improved iteratively, learning advanced strategies and creating dynamic policies.
- The adversarial training approach produced agents capable of generalizing to various gameplay situations.
- Results highlight the robustness of PPO for multi-agent reinforcement learning scenarios.

Further improvements include:
1. Extending training cycles for better convergence.
2. Fine-tuning noise and exploration strategies to improve adaptability against unseen opponents.

