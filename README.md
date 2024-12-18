# Machine Learning Paradigms Project

This project focuses on solving three reinforcement learning tasks using advanced techniques and models. Each part of the project is located in its respective folder.

Authors: Joan Lafuente & Adrián García

---

## Table of Contents

1. [Part 1: Frogger ALE Environment](#part-1-frogger-ale-environment)
2. [Part 2: Ms. Pac-Man ALE Environment](#part-2-ms-pac-man-ale-environment)
3. [Part 3: Pong World Tournament](#part-3-pong-world-tournament)

---

## Part 1: Frogger ALE Environment

In this part, we implemented and compared two Deep Q-Learning algorithms:
1. **DQN + Extensions**:  
   - Double DQN  
   - Dueling DQN  
   - Prioritized Experience Replay  

2. **Rainbow DQN**:  
   - Double DQN, Dueling DQN  
   - N-Step Prioritized Replay  
   - Noisy Layers with Gaussian Noise Injection  

### Results:
- **DQN + Extensions** achieves better long-term performance.
- **Rainbow DQN** learns faster but plateaus earlier.

For detailed results and visualizations, refer to the `part1/` folder.

---

## Part 2: Ms. Pac-Man ALE Environment

We implemented two policy gradient algorithms to solve the **Ms. Pac-Man** game:
1. **PPO (Proximal Policy Optimization)**: Achieves the highest performance with stable learning.  
2. **A2C (Advantage Actor-Critic)**: Performs well but is outperformed by PPO.  

### Results:
- PPO achieves an average reward of **4403.3**.
- A2C achieves an average reward of **2146.5**.

For hyperparameter tuning and training results, refer to the `part2/` folder.

---

## Part 3: Pong World Tournament

In this part, we developed an **adversarial training** method to train two agents to compete in the **Pong** game:
- **Algorithm**: PPO (Proximal Policy Optimization)  
- Agents are trained alternately, improving through adversarial gameplay.

### Results:
- The agents demonstrate advanced strategies like ball effects and speed control.
- Generalized policies were observed, with **left agent** achieving a win rate of **90\%**.

For training methodology and evaluation results, see the `part3/` folder.

---

