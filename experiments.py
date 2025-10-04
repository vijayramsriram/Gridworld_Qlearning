"""
experiments.py
Experiment functions for hyperparameter analysis
"""

import numpy as np
import copy
from qlearning_agent import QLearningAgent
import matplotlib.pyplot as plt


def experiment_learning_rate(env, alphas=[0.01, 0.1, 0.5, 1.0], num_episodes=1000):
    """
    Experiment 1: Test different learning rates
    
    Returns:
    --------
    all_rewards : dict
        {alpha: list of episode rewards}
    all_agents : dict
        {alpha: trained agent}
    """
    print("="*70)
    print("EXPERIMENT 1: EFFECT OF LEARNING RATE (α)")
    print("="*70)
    
    all_rewards = {}
    all_agents = {}
    
    for alpha in alphas:
        print(f"\nTraining with α={alpha}...")
        agent = QLearningAgent(env, alpha=alpha, gamma=0.99, epsilon=0.1)
        rewards = agent.train(num_episodes=num_episodes, verbose=False)
        
        all_rewards[alpha] = rewards
        all_agents[alpha] = agent
        
        final_avg = np.mean(rewards[-100:])
        print(f"  Final avg reward: {final_avg:.4f}")
    
    return all_rewards, all_agents


def experiment_discount_factor(env, gammas=[0.5, 0.8, 0.95, 0.99], num_episodes=1000):
    """
    Experiment 2: Test different discount factors
    
    Returns:
    --------
    all_rewards : dict
    all_agents : dict
    """
    print("="*70)
    print("EXPERIMENT 2: EFFECT OF DISCOUNT FACTOR (γ)")
    print("="*70)
    
    all_rewards = {}
    all_agents = {}
    
    for gamma in gammas:
        print(f"\nTraining with γ={gamma}...")
        agent = QLearningAgent(env, alpha=0.1, gamma=gamma, epsilon=0.1)
        rewards = agent.train(num_episodes=num_episodes, verbose=False)
        
        all_rewards[gamma] = rewards
        all_agents[gamma] = agent
        
        final_avg = np.mean(rewards[-100:])
        print(f"  Final avg reward: {final_avg:.4f}")
    
    return all_rewards, all_agents


def experiment_exploration_rate(env, epsilons=[0.0, 0.1, 0.3, 0.5], num_episodes=1000):
    """
    Experiment 3: Test different exploration rates
    
    Returns:
    --------
    all_rewards : dict
    all_agents : dict
    """
    print("="*70)
    print("EXPERIMENT 3: EFFECT OF EXPLORATION RATE (ε)")
    print("="*70)
    
    all_rewards = {}
    all_agents = {}
    
    for epsilon in epsilons:
        print(f"\nTraining with ε={epsilon}...")
        agent = QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=epsilon)
        rewards = agent.train(num_episodes=num_episodes, verbose=False)
        
        all_rewards[epsilon] = rewards
        all_agents[epsilon] = agent
        
        final_avg = np.mean(rewards[-100:])
        print(f"  Final avg reward: {final_avg:.4f}")
    
    return all_rewards, all_agents


def experiment_convergence(env, num_episodes=2000, check_interval=10):
    """
    Experiment 4: Track Q-value vs Policy convergence
    
    Returns:
    --------
    agent : trained agent
    q_changes : list of Q-value changes
    policy_changes : list of policy changes
    episode_numbers : list of episode numbers
    """
    print("="*70)
    print("EXPERIMENT 4: Q-VALUE vs POLICY CONVERGENCE")
    print("="*70)
    
    agent = QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)
    
    q_changes = []
    policy_changes = []
    episode_numbers = []
    
    prev_q = None
    prev_policy = None
    
    print(f"Training for {num_episodes} episodes...\n")
    
    for episode in range(num_episodes):
        # Run one episode
        state = env.reset()
        steps = 0
        max_steps = 100
        
        while not env.is_terminal(state) and steps < max_steps:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            steps += 1
        
        # Track changes
        if (episode + 1) % check_interval == 0:
            current_q = copy.deepcopy(agent.Q)
            current_policy = agent.get_policy()
            
            if prev_q is not None:
                # Q-value change
                total_q_diff = 0
                count = 0
                for state in current_q:
                    for action in current_q[state]:
                        diff = abs(current_q[state][action] - prev_q[state][action])
                        total_q_diff += diff
                        count += 1
                
                avg_q_change = total_q_diff / count if count > 0 else 0
                q_changes.append(avg_q_change)
                
                # Policy change
                num_policy_changes = 0
                for state in current_policy:
                    if state in prev_policy:
                        if current_policy[state] != prev_policy[state]:
                            num_policy_changes += 1
                
                policy_changes.append(num_policy_changes)
                episode_numbers.append(episode + 1)
            
            prev_q = current_q
            prev_policy = current_policy
        
        if (episode + 1) % 200 == 0:
            if q_changes:
                print(f"Episode {episode+1}: Q-change={q_changes[-1]:.6f}, Policy changes={policy_changes[-1]}")
    
    print("\n✓ Training complete!")
    
    # Analysis
    policy_converged_idx = None
    for i in range(len(policy_changes) - 10):
        if all(pc == 0 for pc in policy_changes[i:i+10]):
            policy_converged_idx = i
            break
    
    q_converged_idx = next((i for i, change in enumerate(q_changes) if change < 0.01), None)
    
    print("\n" + "="*70)
    print("CONVERGENCE RESULTS:")
    print("="*70)
    
    if policy_converged_idx is not None:
        print(f"Policy converged at episode: {episode_numbers[policy_converged_idx]}")
    
    if q_converged_idx is not None:
        print(f"Q-values stabilized at episode: {episode_numbers[q_converged_idx]}")
    
    if policy_converged_idx is not None and q_converged_idx is not None:
        if episode_numbers[policy_converged_idx] < episode_numbers[q_converged_idx]:
            print("\n→ POLICY CONVERGED FIRST! ✓")
        else:
            print("\n→ Q-VALUES CONVERGED FIRST!")
    
    return agent, q_changes, policy_changes, episode_numbers


def experiment_trap_penalty(env_normal, env_large, num_episodes=1000):
    """
    Experiment 5: Compare trap penalties (-1 vs -200)
    
    Returns:
    --------
    agent_normal : agent trained with penalty -1
    agent_large : agent trained with penalty -200
    rewards_normal : episode rewards for normal penalty
    rewards_large : episode rewards for large penalty
    """
    print("="*70)
    print("EXPERIMENT 5: TRAP PENALTY COMPARISON (-1 vs -200)")
    print("="*70)
    
    # Train with penalty -1
    print("\nTraining with trap penalty = -1...")
    agent_normal = QLearningAgent(env_normal, alpha=0.1, gamma=0.99, epsilon=0.1)
    rewards_normal = agent_normal.train(num_episodes=num_episodes, verbose=False)
    print(f"Final avg reward: {np.mean(rewards_normal[-100:]):.4f}")
    
    # Train with penalty -200
    print("\nTraining with trap penalty = -200...")
    agent_large = QLearningAgent(env_large, alpha=0.1, gamma=0.99, epsilon=0.1)
    rewards_large = agent_large.train(num_episodes=num_episodes, verbose=False)
    print(f"Final avg reward: {np.mean(rewards_large[-100:]):.4f}")
    
    # Compare policies
    policy_normal = agent_normal.get_policy()
    policy_large = agent_large.get_policy()
    
    differences = []
    action_names = ['Up', 'Right', 'Down', 'Left']
    
    print("\n" + "="*70)
    print("POLICY COMPARISON:")
    print("="*70)
    
    for state in policy_normal:
        if state in policy_large:
            if policy_normal[state] != policy_large[state]:
                differences.append((state, 
                                  action_names[policy_normal[state]], 
                                  action_names[policy_large[state]]))
    
    if len(differences) == 0:
        print("✓ Policies are IDENTICAL!")
    else:
        print(f"✗ Policies differ at {len(differences)} state(s):")
        for state, action1, action2 in differences:
            print(f"  State {state}: -1→{action1}, -200→{action2}")
    
    return agent_normal, agent_large, rewards_normal, rewards_large



def analyze_trap_qvalues(agents_dict, env):
    """
    Analyze Q-values at state (3,2) - the state next to the trap
    This shows how different learning rates affect risk assessment
    """
    print("="*70)
    print("ANALYSIS: Q-VALUES AT STATE (3,2) - NEXT TO TRAP")
    print("="*70)
    print("State (3,2) is directly left of the trap (4,2)")
    print("Action 'Right' leads to trap with 80% probability")
    print("="*70)
    
    state = (3, 2)
    alphas = [0.01, 0.1, 0.5, 1.0]
    action_names = ['Up', 'Right', 'Down', 'Left']
    
    # Prepare data for plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Collect Q-values for each alpha
    qvalues_by_alpha = {}
    for alpha in alphas:
        agent = agents_dict[alpha]
        qvalues = [agent.Q[state][action] for action in range(4)]
        qvalues_by_alpha[alpha] = qvalues
        
        print(f"\nα = {alpha}:")
        print("-" * 40)
        for action, qval in enumerate(qvalues):
            marker = "  ← TRAP!" if action == 1 else ""
            print(f"  {action_names[action]:6s}: {qval:8.4f}{marker}")
        
        best_action = np.argmax(qvalues)
        print(f"  Best action: {action_names[best_action]}")
    
    # Plot 1: Q-values comparison as grouped bar chart
    x = np.arange(len(action_names))
    width = 0.2
    colors = ['blue', 'green', 'red', 'purple']
    
    for i, (alpha, color) in enumerate(zip(alphas, colors)):
        offset = width * (i - 1.5)
        qvals = qvalues_by_alpha[alpha]
        bars = ax1.bar(x + offset, qvals, width, label=f'α={alpha}', 
                      color=color, alpha=0.7)
        
        # Highlight the trap action (Right)
        bars[1].set_edgecolor('red')
        bars[1].set_linewidth(3)
    
    ax1.set_xlabel('Action', fontsize=12)
    ax1.set_ylabel('Q-value', fontsize=12)
    ax1.set_title('Q-values at State (3,2) for Different α', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(action_names)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add annotation for trap action
    ax1.text(1, ax1.get_ylim()[0], '↑\nTRAP', ha='center', va='bottom', 
            fontsize=10, color='red', fontweight='bold')
    
    # Plot 2: Comparison of Right action (trap direction) Q-value
    right_qvalues = [qvalues_by_alpha[alpha][1] for alpha in alphas]
    bars2 = ax2.bar([str(a) for a in alphas], right_qvalues, color=colors, alpha=0.7)
    ax2.set_xlabel('Learning Rate (α)', fontsize=12)
    ax2.set_ylabel('Q-value for "Right" (toward trap)', fontsize=12)
    ax2.set_title('How α Affects Learning to Avoid Trap', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars2, right_qvalues):
        height = bar.get_height()
        y_pos = height if height > 0 else height
        va = 'bottom' if height > 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:.3f}', ha='center', va=va, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    
    # Check policy differences
    print(f"\n{'='*70}")
    print("POLICY COMPARISON AT STATE (3,2):")
    print(f"{'='*70}")
    
    for alpha in alphas:
        agent = agents_dict[alpha]
        policy = agent.get_policy()
        best_action = policy[state]
        action_name = action_names[best_action]
        
        risk_level = "SAFE" if best_action != 1 else "RISKY"
        print(f"α={alpha:4}: Chooses {action_name:6s} [{risk_level}]")


def visualize_gamma_policies(agents_dict, env):
    """
    Visualize policies learned with different gamma values
    """
    print("\n" + "="*70)
    print("VISUALIZING POLICIES FOR DIFFERENT γ VALUES")
    print("="*70)
    
    gammas = [0.5, 0.8, 0.95, 0.99]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    arrow_dict = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    
    for idx, gamma in enumerate(gammas):
        ax = axes[idx]
        agent = agents_dict[gamma]
        policy = agent.get_policy()
        
        # Draw grid
        for i in range(env.rows + 1):
            ax.axhline(i, color='black', linewidth=2)
        for j in range(env.cols + 1):
            ax.axvline(j, color='black', linewidth=2)
        
        # Fill cells
        for row in range(1, env.rows + 1):
            for col in range(1, env.cols + 1):
                state = (col, row)
                plot_col = col - 1
                plot_row = row - 1
                
                if state == env.wall_state:
                    ax.add_patch(plt.Rectangle((plot_col, plot_row), 1, 1, color='gray'))
                elif state == env.goal_state:
                    ax.add_patch(plt.Rectangle((plot_col, plot_row), 1, 1, color='lightgreen'))
                    ax.text(plot_col + 0.5, plot_row + 0.5, '+1', ha='center', va='center', 
                           fontsize=12, fontweight='bold')
                elif state == env.trap_state:
                    ax.add_patch(plt.Rectangle((plot_col, plot_row), 1, 1, color='lightcoral'))
                    ax.text(plot_col + 0.5, plot_row + 0.5, '-1', ha='center', va='center', 
                           fontsize=12, fontweight='bold')
                elif state in policy:
                    arrow = arrow_dict[policy[state]]
                    ax.text(plot_col + 0.5, plot_row + 0.5, arrow, ha='center', va='center', 
                           fontsize=28, fontweight='bold', color='blue')
        
        ax.set_xlim(0, env.cols)
        ax.set_ylim(0, env.rows)
        ax.set_aspect('equal')
        ax.set_title(f'Policy with γ={gamma}', fontsize=13, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle('Policy Comparison: Different Discount Factors', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def compare_epsilon_policies(agents_dict, env):
    """
    Visualize policies learned with different epsilon values
    """
    print("\n" + "="*70)
    print("VISUALIZING POLICIES FOR DIFFERENT ε VALUES")
    print("="*70)
    
    epsilons = [0.0, 0.1, 0.3, 0.5]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    arrow_dict = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    
    for idx, epsilon in enumerate(epsilons):
        ax = axes[idx]
        agent = agents_dict[epsilon]
        policy = agent.get_policy()
        
        # Draw grid
        for i in range(env.rows + 1):
            ax.axhline(i, color='black', linewidth=2)
        for j in range(env.cols + 1):
            ax.axvline(j, color='black', linewidth=2)
        
        # Fill cells
        for row in range(1, env.rows + 1):
            for col in range(1, env.cols + 1):
                state = (col, row)
                plot_col = col - 1
                plot_row = row - 1
                
                if state == env.wall_state:
                    ax.add_patch(plt.Rectangle((plot_col, plot_row), 1, 1, color='gray'))
                elif state == env.goal_state:
                    ax.add_patch(plt.Rectangle((plot_col, plot_row), 1, 1, color='lightgreen'))
                    ax.text(plot_col + 0.5, plot_row + 0.5, '+1', ha='center', va='center', 
                           fontsize=12, fontweight='bold')
                elif state == env.trap_state:
                    ax.add_patch(plt.Rectangle((plot_col, plot_row), 1, 1, color='lightcoral'))
                    ax.text(plot_col + 0.5, plot_row + 0.5, '-1', ha='center', va='center', 
                           fontsize=12, fontweight='bold')
                elif state in policy:
                    arrow = arrow_dict[policy[state]]
                    ax.text(plot_col + 0.5, plot_row + 0.5, arrow, ha='center', va='center', 
                           fontsize=28, fontweight='bold', color='blue')
        
        ax.set_xlim(0, env.cols)
        ax.set_ylim(0, env.rows)
        ax.set_aspect('equal')
        
        title = f'Policy with ε={epsilon}'
        if epsilon == 0.0:
            title += ' (No Exploration)'
        elif epsilon == 0.1:
            title += ' (Recommended)'
        elif epsilon == 0.5:
            title += ' (Too Much!)'
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle('Policy Comparison: Different Exploration Rates', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def experiment_convergence_analysis(env):
    """
    Track both Q-values and policy changes over training to determine
    which converges first.
    
    KEY QUESTION: Does the policy stabilize before Q-values, or vice versa?
    
    HYPOTHESIS: Policy should converge first because it only depends on
    which action has the HIGHEST Q-value, not the exact Q-value magnitude.
    """
    print("="*70)
    print("EXPERIMENT 4: Q-VALUE vs POLICY CONVERGENCE")
    print("="*70)
    print("Tracking changes in Q-values and policy over 2000 episodes")
    print("Parameters: α=0.1, γ=0.99, ε=0.1")
    print("="*70)
    
    # Create agent
    agent = QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)
    
    # Storage for tracking
    q_changes = []
    policy_changes = []
    episode_numbers = []
    
    prev_q = None
    prev_policy = None
    
    num_episodes = 2000
    check_interval = 10  # Check every 10 episodes
    
    print(f"\nTraining for {num_episodes} episodes...")
    print("Tracking convergence every 10 episodes...\n")
    
    # Training loop with tracking
    for episode in range(num_episodes):
        # Run one episode
        state = env.reset()
        steps = 0
        max_steps = 100
        
        while not env.is_terminal(state) and steps < max_steps:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            steps += 1
        
        # Track changes every check_interval episodes
        if (episode + 1) % check_interval == 0:
            # Get current Q-values and policy
            current_q = copy.deepcopy(agent.Q)
            current_policy = agent.get_policy()
            
            if prev_q is not None:
                # Calculate Q-value change (average absolute difference)
                total_q_diff = 0
                count = 0
                for state in current_q:
                    for action in current_q[state]:
                        diff = abs(current_q[state][action] - prev_q[state][action])
                        total_q_diff += diff
                        count += 1
                
                avg_q_change = total_q_diff / count if count > 0 else 0
                q_changes.append(avg_q_change)
                
                # Calculate policy change (number of states where best action changed)
                num_policy_changes = 0
                for state in current_policy:
                    if state in prev_policy:
                        if current_policy[state] != prev_policy[state]:
                            num_policy_changes += 1
                
                policy_changes.append(num_policy_changes)
                episode_numbers.append(episode + 1)
            
            prev_q = current_q
            prev_policy = current_policy
        
        # Print progress
        if (episode + 1) % 200 == 0:
            if q_changes:
                print(f"Episode {episode+1:4d}: Q-change={q_changes[-1]:.6f}, "
                      f"Policy changes={policy_changes[-1]}")
    
    print("\n✓ Training complete!")
    
    # Plot convergence
    print("\nGenerating convergence plots...")
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Q-value changes over time
    ax1.plot(episode_numbers, q_changes, linewidth=2, color='blue', marker='o', markersize=2)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Average Q-value Change', fontsize=12)
    ax1.set_title('Q-value Convergence (Lower = More Stable)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale to see small changes
    
    # Add convergence threshold line
    ax1.axhline(y=0.01, color='red', linestyle='--', linewidth=2, label='Threshold (0.01)')
    ax1.axhline(y=0.001, color='orange', linestyle='--', linewidth=2, label='Threshold (0.001)')
    ax1.legend()
    
    # Plot 2: Policy changes over time
    ax2.plot(episode_numbers, policy_changes, linewidth=2, color='red', marker='s', markersize=3)
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Number of Policy Changes', fontsize=12)
    ax2.set_title('Policy Convergence (Zero = Fully Converged)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add reference line at 0
    ax2.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Fully Converged')
    ax2.legend()
    
    # Plot 3: Combined view (normalized)
    # Normalize both to 0-1 scale for comparison
    q_changes_norm = np.array(q_changes) / max(q_changes) if max(q_changes) > 0 else np.array(q_changes)
    policy_changes_norm = np.array(policy_changes) / max(policy_changes) if max(policy_changes) > 0 else np.array(policy_changes)
    
    ax3.plot(episode_numbers, q_changes_norm, linewidth=2, color='blue', label='Q-value changes (normalized)')
    ax3.plot(episode_numbers, policy_changes_norm, linewidth=2, color='red', label='Policy changes (normalized)')
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Normalized Change (0-1)', fontsize=12)
    ax3.set_title('Combined View: Q-value vs Policy Convergence', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    # Analysis
    print("\n" + "="*70)
    print("CONVERGENCE ANALYSIS:")
    print("="*70)
    
    # Find when policy converges (first time it reaches 0 changes and stays there)
    policy_converged_idx = None
    for i in range(len(policy_changes) - 10):  # Check if stable for 10 checks
        if all(pc == 0 for pc in policy_changes[i:i+10]):
            policy_converged_idx = i
            break
    
    if policy_converged_idx is not None:
        policy_converged_episode = episode_numbers[policy_converged_idx]
        print(f"\n✓ POLICY converged at episode: {policy_converged_episode}")
        print(f"  (Reached 0 changes and remained stable)")
    else:
        print(f"\n✗ POLICY did not fully converge (still changing)")
    
    # Find when Q-values converge (change < 0.01)
    q_converged_idx_01 = next((i for i, change in enumerate(q_changes) if change < 0.01), None)
    q_converged_idx_001 = next((i for i, change in enumerate(q_changes) if change < 0.001), None)
    
    if q_converged_idx_01 is not None:
        q_converged_episode_01 = episode_numbers[q_converged_idx_01]
        print(f"\n✓ Q-VALUES stabilized (change < 0.01) at episode: {q_converged_episode_01}")
    else:
        print(f"\n✗ Q-VALUES did not stabilize to < 0.01")
    
    if q_converged_idx_001 is not None:
        q_converged_episode_001 = episode_numbers[q_converged_idx_001]
        print(f"✓ Q-VALUES highly stabilized (change < 0.001) at episode: {q_converged_episode_001}")
    else:
        print(f"✗ Q-VALUES did not stabilize to < 0.001")
    
    # Compare convergence
    print("\n" + "="*70)
    print("ANSWER TO QUESTION 2:")
    print("="*70)
    
    if policy_converged_idx is not None and q_converged_idx_01 is not None:
        if policy_converged_episode < q_converged_episode_01:
            print(f"\n→ POLICY CONVERGED FIRST! ✓")
            print(f"\n  Policy converged at:  Episode {policy_converged_episode}")
            print(f"  Q-values stable at:   Episode {q_converged_episode_01}")
            print(f"  Difference:           {q_converged_episode_01 - policy_converged_episode} episodes")
        else:
            print(f"\n→ Q-VALUES CONVERGED FIRST!")
            print(f"\n  Q-values stable at:   Episode {q_converged_episode_01}")
            print(f"  Policy converged at:  Episode {policy_converged_episode}")
    
    
    final_policy = agent.get_policy()
    action_names = ['Up', 'Right', 'Down', 'Left']
    
    for row in range(env.rows, 0, -1):
        for col in range(1, env.cols + 1):
            state = (col, row)
            if state == env.goal_state:
                print(f"  {state}: GOAL", end="")
            elif state == env.trap_state:
                print(f"  {state}: TRAP", end="")
            elif state == env.wall_state:
                print(f"  {state}: WALL", end="")
            elif state in final_policy:
                action = final_policy[state]
                print(f"  {state}: {action_names[action]}", end="")
        print()  # New row
    
    agent.visualize_policy()
    
    return agent, q_changes, policy_changes, episode_numbers



    
