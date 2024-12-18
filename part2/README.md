# Part 2: Ms. Pac-Man ALE Environment

This part of the project focuses on solving the **Ms. Pac-Man** game using two reinforcement learning algorithms: **Proximal Policy Optimization (PPO)** and **Advantage Actor-Critic (A2C)**.

---

## Project Overview

Ms. Pac-Man is a game where the player controls Ms. Pac-Man to:
- Navigate through a maze.
- Eat pellets while avoiding ghosts.
- Use power pellets to eat ghosts temporarily.

The environment has **8 discrete actions**:
- **0**: NOOP  
- **1**: UP  
- **2**: RIGHT  
- **3**: LEFT  
- **4**: DOWN  
- **5**: UPRIGHT  
- **6**: UPLEFT  
- **7**: DOWNRIGHT  
- **8**: DOWNLEFT  

---

## Preprocessing

Both models use the same preprocessing pipeline:
1. **No-op reset**: Random no-ops at environment start.
2. **Frame Skipping**: Skips 4 frames to reduce computation.
3. **Life Wrapper**: Simulates game end on life loss.
4. **Image Resizing**: Resizes to 84x84 pixels and converts to grayscale.
5. **Reward Clipping**: Scales rewards to the range \([-1, 0, 1]\).
6. **Frame Stacking**: Stacks the last 4 frames for motion awareness.

---

## Implemented Algorithms

### 1. PPO (Proximal Policy Optimization)

- **Description**:  
  PPO optimizes the policy directly using gradient ascent with a clipping mechanism to ensure stable updates.

- **Hyperparameter Search**:
  - Learning Rates: `0.00025`, `0.00075`, `0.000075`
  - Discount Factors: `0.9`, `0.95`, `0.99`

- **Best Configuration**:
  - Learning Rate: **0.00025**
  - Discount Factor: **0.9**

- **Results**:
  - **Average Reward**: 4403.3  
  - **Standard Deviation**: 2190.21  

The trained PPO agent successfully completes the first maze and often progresses significantly into the second screen.

### 2. A2C (Advantage Actor-Critic)

- **Description**:  
  A2C combines policy-based and value-based methods to optimize the actor (policy) and critic (value estimation).

- **Hyperparameter Search**:
  - Learning Rates: `0.007`, `0.001`, `0.00025`, `0.00005`
  - Discount Factors: `0.9`, `0.99`

- **Best Configuration**:
  - Learning Rate: **0.001**
  - Discount Factor: **0.9**

- **Results**:
  - **Average Reward**: 2146.5  
  - **Standard Deviation**: 785.015  

The A2C agent learned a path-based strategy but struggled to avoid ghosts reliably.

---

## Results Comparison

| Metric               | **PPO**          | **A2C**         |
|-----------------------|------------------|-----------------|
| **Average Reward**    | 4403.3           | 2146.5          |
| **Standard Deviation**| 2190.21          | 785.015         |

- **PPO** outperformed A2C, achieving almost double the average reward.  
- The PPO agent demonstrated better stability and exploration during training.

---

## How to Run the Code

Follow these steps to train or evaluate the agents:

1. **Activate the Conda Environment**:
   ```bash
   conda env create -f environment_part2.yml
   conda activate enviroment_part2
    ´´´
2. **Train the Agents**:

   - **Train A2C**:
     Run the training script for the A2C agent:
     ```bash
     cd A2C
     python A2C_train.py --model_name A2C --env_name MsPacman --yaml_file {name_yaml_file}
     ```

   - **Train PPO**:
     Run the training script for the PPO agent:
     ```bash
     cd PPO
     python PPO_train.py {name_yaml_file}
     ```

3. **Evaluate the Agents**:

   - **Evaluate A2C**:
     Use the A2C evaluation script to test the trained model:
     ```bash
     cd A2C
     python A2C_eval.py {name_yaml_file}
     ```

   - **Evaluate PPO**:
     Use the PPO evaluation script to test the trained model:
     ```bash
     cd PPO
     python PPO_eval.py {name_yaml_file}
     ```

4. **Logs and Results**:
   - Training and evaluation logs are stored in:
     - `A2C/log_dir/`
     - `PPO/log_dir/`

   - Best model weights and outputs can be found in:
     - `A2C/Best Model Results/`
     - `PPO/Best Model Results/`

---

## Ms. Pac-Man Agent Evaluation

### PPO Agent
[Click to view the PPO agent video](videos/PPO_Pacman.mp4)

### A2C Agent
[Click to view the A2C agent video](videos/A2C_Pacman.mp4)

---

## Key Observations

### PPO vs. A2C:
- **PPO**:
  - Provides superior performance, achieving higher rewards with better stability and exploration.
- **A2C**:
  - Converges faster initially but produces less stable results and performs worse overall.

### Preprocessing:
- Using a **frame skip of 4** and **reward clipping** significantly reduced training time and improved performance.

### Training Dynamics:
- PPO's **clipping mechanism** helped avoid large policy updates, leading to better generalization and stability.

---

## Summary

| **Algorithm** | **Strengths**                      | **Weaknesses**                  |
|---------------|------------------------------------|---------------------------------|
| **PPO**       | Stable updates, better long-term rewards | Slower convergence initially    |
| **A2C**       | Faster initial convergence         | Lower overall performance       |

---

The **PPO algorithm** is the preferred solution for solving the **Ms. Pac-Man environment** due to its **superior performance** and **robustness**.
