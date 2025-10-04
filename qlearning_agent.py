import numpy as np
import matplotlib.pyplot as plt
import random

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        Q-Learning Agent for Online Learning

        Parameters:
        -----------
        env : GridWorld environment
        alpha : float (learning rate)
            How much to update Q-values (0 to 1)
        gamma : float (discount factor)
            How much to value future rewards (0 to 1)
        epsilon : float (exploration rate)
            Probability of taking random action (0 to 1)
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize Q-table as dictionary: Q[state][action] = value
        self.Q = {}
        self._initialize_q_table()

    def _initialize_q_table(self):
        """Initialize Q-values to 0 for all state-action pairs"""
        for row in range(1, self.env.rows + 1):
            for col in range(1, self.env.cols + 1):
                state = (col, row)
                if self.env.is_valid_state(state) and not self.env.is_terminal(state):
                    self.Q[state] = {action: 0.0 for action in self.env.actions}

    def get_action(self, state):
        """
        Choose action using epsilon-greedy policy

        Returns:
        --------
        action : int (0=Up, 1=Right, 2=Down, 3=Left)
        """
        if self.env.is_terminal(state):
            return None

        # Exploration: random action with probability epsilon
        if random.random() < self.epsilon:
            return random.choice(self.env.actions)

        # Exploitation: choose action with highest Q-value
        q_values = [self.Q[state][a] for a in self.env.actions]
        max_q = max(q_values)

        # If multiple actions have same max Q, choose randomly among them
        best_actions = [a for a in self.env.actions if self.Q[state][a] == max_q]
        return random.choice(best_actions)

    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value using Q-learning formula:
        Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
        """
        if self.env.is_terminal(state):
            return

        # Get max Q-value for next state
        if self.env.is_terminal(next_state):
            max_next_q = 0  # Terminal state has no future value
        else:
            max_next_q = max(self.Q[next_state].values())

        # Q-learning update
        current_q = self.Q[state][action]
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        self.Q[state][action] = current_q + self.alpha * td_error

    def train(self, num_episodes=1000, max_steps_per_episode=100, verbose=True):
        """
        Train the agent using online Q-learning

        Parameters:
        -----------
        num_episodes : int
            Number of episodes to train
        max_steps_per_episode : int
            Maximum steps per episode (to avoid infinite loops)
        verbose : bool
            Whether to print progress

        Returns:
        --------
        episode_rewards : list
            Total reward obtained in each episode
        """
        episode_rewards = []

        if verbose:
            print("="*60)
            print("ONLINE Q-LEARNING TRAINING")
            print("="*60)
            print(f"Hyperparameters: α={self.alpha}, γ={self.gamma}, ε={self.epsilon}")
            print(f"Training for {num_episodes} episodes...\n")

        for episode in range(num_episodes):
            # Reset environment to start state
            state = self.env.reset()
            total_reward = 0
            steps = 0

            # Run one episode
            while not self.env.is_terminal(state) and steps < max_steps_per_episode:
                # Choose action
                action = self.get_action(state)

                # Take action in environment
                next_state, reward, done = self.env.step(action)

                # Update Q-value
                self.update_q_value(state, action, reward, next_state)

                # Update state and counters
                state = next_state
                total_reward += reward
                steps += 1

            episode_rewards.append(total_reward)

            # Print progress every 100 episodes
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode+1:4d}: Avg Reward (last 100) = {avg_reward:7.3f}")

        if verbose:
            print("\n✓ Training Complete!")

        return episode_rewards

    def get_policy(self):
        """
        Extract the learned policy (best action for each state)

        Returns:
        --------
        policy : dict
            policy[state] = best_action
        """
        policy = {}
        for state in self.Q:
            best_action = max(self.Q[state], key=self.Q[state].get)
            policy[state] = best_action
        return policy

    def visualize_policy(self):
        """Visualize the learned policy with arrows"""
        fig, ax = plt.subplots(figsize=(10, 7))

        # Draw grid
        for i in range(self.env.rows + 1):
            ax.axhline(i, color='black', linewidth=2)
        for j in range(self.env.cols + 1):
            ax.axvline(j, color='black', linewidth=2)

        policy = self.get_policy()
        arrow_dict = {0: '↑', 1: '→', 2: '↓', 3: '←'}

        # Fill cells
        for row in range(1, self.env.rows + 1):
            for col in range(1, self.env.cols + 1):
                state = (col, row)
                plot_col = col - 1
                plot_row = row - 1

                if state == self.env.wall_state:
                    ax.add_patch(plt.Rectangle((plot_col, plot_row), 1, 1, color='gray'))
                    ax.text(plot_col + 0.5, plot_row + 0.5, 'WALL', ha='center', va='center',
                           fontsize=10, fontweight='bold')
                elif state == self.env.goal_state:
                    ax.add_patch(plt.Rectangle((plot_col, plot_row), 1, 1, color='lightgreen'))
                    ax.text(plot_col + 0.5, plot_row + 0.5, 'GOAL\n+1', ha='center', va='center',
                           fontsize=14, fontweight='bold')                                    
                elif state == self.env.trap_state:
                    ax.add_patch(plt.Rectangle((plot_col, plot_row), 1, 1, color='lightcoral'))
                    penalty_text = f'{self.env.trap_reward:.0f}' if self.env.trap_reward != -1 else '-1'
                    ax.text(plot_col + 0.5, plot_row + 0.5, f'TRAP\n{penalty_text}', ha='center', va='center',
                        fontsize=14, fontweight='bold')
                elif state in policy:
                    arrow = arrow_dict[policy[state]]
                    ax.text(plot_col + 0.5, plot_row + 0.5, arrow, ha='center', va='center',
                           fontsize=35, fontweight='bold', color='blue')

        ax.set_xlim(0, self.env.cols)
        ax.set_ylim(0, self.env.rows)
        ax.set_aspect('equal')
        ax.set_xticks(range(self.env.cols + 1))
        ax.set_yticks(range(self.env.rows + 1))
        ax.set_xticklabels([1, 2, 3, 4, 5])
        ax.set_yticklabels([1, 2, 3, 4])
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        ax.set_title('Learned Policy (Online Q-Learning)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def visualize_q_values(self, state):
        """
        Print Q-values for a specific state

        Parameters:
        -----------
        state : tuple
            State to display Q-values for
        """
        if state not in self.Q:
            print(f"State {state} not in Q-table")
            return

        print(f"\nQ-values for state {state}:")
        print("-" * 40)
        for action in self.env.actions:
            action_name = self.env.action_names[action]
            q_value = self.Q[state][action]
            print(f"  {action_name:6s} (action {action}): {q_value:8.4f}")

        best_action = max(self.Q[state], key=self.Q[state].get)
        print(f"\nBest action: {self.env.action_names[best_action]}")

    def plot_learning_curve(self, episode_rewards):
        """
        Plot the learning curve (rewards over episodes)

        Parameters:
        -----------
        episode_rewards : list
            Rewards obtained in each episode
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot raw rewards
        ax.plot(episode_rewards, alpha=0.3, color='blue', label='Episode Reward')

        # Plot smoothed rewards (moving average)
        window = 50
        if len(episode_rewards) >= window:
            smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(episode_rewards)), smoothed,
                   color='red', linewidth=2, label=f'Moving Average (window={window})')

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title('Learning Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
