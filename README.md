# Q-Learning for Gridworld Navigation

A comprehensive implementation and analysis of Q-Learning applied to a 3×4 stochastic gridworld problem with detailed hyperparameter experiments.

## 📋 Table of Contents
- [Problem Description](#problem-description)
- [Implementation](#implementation)
- [Experiments & Results](#experiments--results)
- [Installation & Usage](#installation--usage)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)

## 🎯 Problem Description

### Gridworld Environment
- **Grid Size:** 3 rows × 4 columns (1-indexed)
- **Start State:** (1,1) - Bottom-left corner
- **Goal State:** (4,3) - Top-right corner [Reward: +1]
- **Trap State:** (4,2) - Middle-right [Reward: -1 or -200]
- **Wall State:** (2,2) - Blocked cell
- **Step Cost:** -0.04 for all non-terminal states

### Stochastic Transitions
- **80%** - Move in intended direction
- **10%** - Move perpendicular left
- **10%** - Move perpendicular right
- Collision with walls results in staying in the same position

### Actions
- 0: Up (North)
- 1: Right (East)
- 2: Down (South)
- 3: Left (West)

## 🛠 Implementation

### Online Q-Learning Algorithm
The agent learns through direct interaction with the environment using the Q-learning update rule:

```
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
```

**Components:**
- **Q-table:** Dictionary storing state-action values
- **ε-greedy policy:** Balances exploration vs exploitation
- **Temporal difference learning:** Updates after each action

### Default Hyperparameters
- Learning rate (α): 0.1
- Discount factor (γ): 0.99
- Exploration rate (ε): 0.1
- Training episodes: 1000

## 🔬 Experiments & Results

### Experiment 1: Effect of Learning Rate (α)

Testing α = {0.01, 0.1, 0.5, 1.0} with fixed γ=0.99, ε=0.1

| Learning Rate (α) | Final Avg Reward | Observation |
|-------------------|------------------|-------------|
| 0.01 | 0.7364 | Slow but stable learning |
| 0.1 | 0.7152 | Optimal balance |
| 0.5 | 0.5992 | Too aggressive updates |
| 1.0 | 0.5020 | Unstable, overwrites old info |

**Key Insight:** Lower learning rates achieve better final performance. α=0.1 provides good balance between learning speed and stability.

**Trap Avoidance Analysis (State 3,2):**
- All agents correctly learn to avoid moving right toward trap
- Higher α values show more extreme Q-values (-1.0) for trap direction
- Lower α values show gradual learning (-0.086 to -0.633)

### Experiment 2: Effect of Discount Factor (γ)

Testing γ = {0.5, 0.8, 0.95, 0.99} with fixed α=0.1, ε=0.1

| Discount Factor (γ) | Final Avg Reward | Future Vision |
|---------------------|------------------|---------------|
| 0.5 | 0.6516 | Short-sighted |
| 0.8 | 0.6760 | Medium-term |
| 0.95 | 0.7324 | Long-term |
| 0.99 | 0.7432 | Far-sighted (best) |

**Key Insight:** All discount factors produce identical policies, but higher γ values achieve better cumulative rewards by properly valuing future returns.

### Experiment 3: Effect of Exploration Rate (ε)

Testing ε = {0.0, 0.1, 0.3, 0.5} with fixed α=0.1, γ=0.99

| Exploration Rate (ε) | Final Avg Reward | Policy Quality |
|----------------------|------------------|----------------|
| 0.0 | 0.7052 | Good (but lucky) |
| 0.1 | 0.6760 | Optimal balance |
| 0.3 | 0.5584 | Suboptimal actions |
| 0.5 | 0.3284 | Highly degraded |

**Key Insight:** ε=0.1 provides best exploration-exploitation trade-off. Higher values prevent policy convergence; lower values risk local optima.

### Experiment 4: Q-Value vs Policy Convergence

**Research Question:** Which converges first - Q-values or policy?

**Answer:** **POLICY CONVERGES FIRST**
- Policy converged: Episode 130
- Q-values stabilized: Episode 300
- Gap: 170 episodes

**Explanation:**
- Policy depends only on *relative* Q-values (which action is best)
- Exact Q-value magnitudes continue refining after policy stabilizes
- This is expected behavior in stochastic environments

### Experiment 5: Trap Penalty Comparison (-1 vs -200)

| Trap Penalty | Episode 100 | Episode 1000 | Policy Difference |
|--------------|-------------|--------------|-------------------|
| -1 | 0.367 | 0.729 | Standard risk avoidance |
| -200 | -19.554 | -1.328 | Ultra-conservative |

**Key Differences:**
- **State (3,2) Policy:**
  - Penalty -1: Move UP (direct avoidance)
  - Penalty -200: Move LEFT (maximum distance from trap)
- **Learning Curve:** Extreme penalty causes catastrophic early episodes
- **Final Behavior:** More risk-averse, takes longer safer paths

**Does it make sense?** YES - Higher penalty logically produces more conservative behavior, trading efficiency for safety.

## 📦 Installation & Usage

### Requirements
```bash
numpy
matplotlib
```

### Installation
```bash
git clone https://github.com/btvvardhan/Gridworld_Qlearning.git
cd gridworld-qlearning
pip install -r requirements.txt
```

### Quick Start
```python
from gridworld import GridWorld
from qlearning_agent import QLearningAgent

# Create environment
env = GridWorld(trap_penalty=-1)

# Create and train agent
agent = QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)
rewards = agent.train(num_episodes=1000, verbose=True)

# Visualize results
agent.visualize_policy()
agent.plot_learning_curve(rewards)
```

### Run All Experiments
```python
from experiments import *

# Experiment 1: Learning rate
rewards_dict, agents_dict = experiment_learning_rate(env)
analyze_trap_qvalues(agents_dict, env)

# Experiment 2: Discount factor
rewards_dict, agents_dict = experiment_discount_factor(env)
visualize_gamma_policies(agents_dict, env)

# Experiment 3: Exploration rate
rewards_dict, agents_dict = experiment_exploration_rate(env)
compare_epsilon_policies(agents_dict, env)

# Experiment 4: Convergence analysis
agent, q_changes, policy_changes, episodes = experiment_convergence_analysis(env)

# Experiment 5: Trap penalty comparison
env_large = GridWorld(trap_penalty=-200)
agent_normal, agent_large, rewards_normal, rewards_large = experiment_trap_penalty(env, env_large)
```

## 🎓 Key Findings

### Question 1: How do hyperparameters affect learning?

1. **Learning Rate (α):**
   - Lower values (0.01-0.1) achieve better final performance
   - Higher values cause instability and poor convergence
   - Recommended: α = 0.1

2. **Discount Factor (γ):**
   - Higher values better account for long-term rewards
   - All tested values produce identical policies in this small gridworld
   - Recommended: γ = 0.99

3. **Exploration Rate (ε):**
   - Critical for policy quality
   - Too high: never settles on good policy
   - Too low: risk of local optima
   - Recommended: ε = 0.1

### Question 2: Does Q-value or policy converge first?

**POLICY CONVERGES FIRST** (170 episodes earlier)

**Why?**
- Policy only needs to know which action is *best* (relative ordering)
- Q-values must estimate precise expected returns (absolute values)
- Small Q-value fluctuations don't affect policy if action ranking unchanged
- This is mathematically expected in stochastic environments

### Question 3: Effect of extreme penalty changes

Increasing trap penalty from -1 to -200:
- Creates more **risk-averse** behavior
- Agent takes **longer but safer** paths
- **Learning is harder** - early catastrophic failures
- Policy makes sense: higher stakes → more conservative strategy

## 📁 Project Structure

```
gridworld-qlearning/
│
├── gridworld.py              # Environment implementation
├── qlearning_agent.py        # Q-Learning agent
├── experiments.py            # All experiments and analysis functions
├── Assignment.ipynb               # Jupyter notebook with all experiments
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 📊 Visualizations

The project includes comprehensive visualizations:
- **Policy grids** with directional arrows
- **Learning curves** showing reward progression
- **Q-value comparisons** at critical states
- **Convergence plots** tracking policy and Q-value changes
- **Side-by-side policy comparisons** for different hyperparameters

## 🔍 Implementation Details

### GridWorld Class
- Configurable trap penalty
- Stochastic state transitions
- Boundary and wall collision handling
- Visualization methods

### QLearningAgent Class
- Dictionary-based Q-table
- ε-greedy action selection
- Online Q-value updates
- Policy extraction and visualization
- Learning curve plotting

### Experiments Module
- 5 comprehensive experiments
- Automated training and comparison
- Statistical analysis functions
- Visualization utilities

## 📝 References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.)
- Watkins, C. J., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3-4), 279-292.

## 👥 Authors

[Teja Vishnu Vardhan Boddu]
Vijayramsriram Sathananthan


---

**Note:** This implementation uses online Q-learning where the agent learns through direct environment interaction without a pre-existing model. All experiments are reproducible with fixed random seeds (seed=42).
