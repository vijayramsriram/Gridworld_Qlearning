import numpy as np
import random
import matplotlib.pyplot as plt

# Define the gridworld environment
class GridWorld:
    def __init__(self, trap_penalty=-1.0):
        """
        GridWorld environment with configurable trap penalty
        
        Parameters:
        -----------
        trap_penalty : float
            Penalty for entering trap state (default: -1.0)
            Can be changed to test different scenarios (e.g., -200.0)
        """
        # Grid layout (1-indexed, col and row start from 1):
        # Row 3 (top):    [(1,3), (2,3), (3,3), (4,3)=GOAL]
        # Row 2:          [(1,2), (2,2)=WALL, (3,2), (4,2)=TRAP]  
        # Row 1 (bottom): [(1,1)=START, (2,1), (3,1), (4,1)]
        
        # Using (col, row) format, 1-indexed
        self.rows = 3
        self.cols = 4
        
        # Special positions (col, row) - 1-indexed
        self.start_state = (1, 1)  # Bottom-left
        self.goal_state = (4, 3)   # Top-right (+1)
        self.trap_state = (4, 2)   # Middle-right (configurable penalty)
        self.wall_state = (2, 2)   # Blocked cell
        
        # Rewards
        self.step_cost = -0.04
        self.goal_reward = 1.0
        self.trap_reward = trap_penalty  # Configurable!
        
        # Current state
        self.state = self.start_state
        
        # Actions: 0=Up, 1=Right, 2=Down, 3=Left
        self.actions = [0, 1, 2, 3]
        self.action_names = ['Up', 'Right', 'Down', 'Left']
        
        # Transition probabilities
        self.intended_prob = 0.8
        self.perpendicular_prob = 0.1
    
    def reset(self):
        """Reset environment to start state"""
        self.state = self.start_state
        return self.state
    
    def is_terminal(self, state):
        """Check if state is terminal (goal or trap)"""
        return state == self.goal_state or state == self.trap_state
    
    def is_valid_state(self, state):
        """Check if state is valid (within bounds and not wall)"""
        col, row = state
        # Check bounds (1-indexed: col 1-4, row 1-3)
        if col < 1 or col > self.cols or row < 1 or row > self.rows:
            return False
        if state == self.wall_state:
            return False
        return True
    
    def get_next_state(self, state, action):
        """Get next state given current state and action (deterministic)"""
        col, row = state
        
        # Action effects: 0=Up, 1=Right, 2=Down, 3=Left
        if action == 0:  # Up (increase row)
            next_state = (col, row + 1)
        elif action == 1:  # Right (increase col)
            next_state = (col + 1, row)
        elif action == 2:  # Down (decrease row)
            next_state = (col, row - 1)
        elif action == 3:  # Left (decrease col)
            next_state = (col - 1, row)
        else:
            next_state = state
        
        # If next state is invalid (wall or out of bounds), stay in current state
        if not self.is_valid_state(next_state):
            return state
        
        return next_state
    
    def get_reward(self, state):
        """Get reward for being in a state"""
        if state == self.goal_state:
            return self.goal_reward
        elif state == self.trap_state:
            return self.trap_reward
        else:
            return self.step_cost
    
    def step(self, action):
        """
        Execute action with stochastic transitions
        80% intended direction, 10% perpendicular left, 10% perpendicular right
        """
        if self.is_terminal(self.state):
            return self.state, 0, True
        
        # Determine actual action taken (with stochasticity)
        rand = random.random()
        
        if rand < self.intended_prob:  # 80% - intended direction
            actual_action = action
        elif rand < self.intended_prob + self.perpendicular_prob:  # 10% - perpendicular left
            # Perpendicular left: Up->Left, Right->Up, Down->Right, Left->Down
            actual_action = (action + 3) % 4
        else:  # 10% - perpendicular right
            # Perpendicular right: Up->Right, Right->Down, Down->Left, Left->Up
            actual_action = (action + 1) % 4
        
        # Get next state
        next_state = self.get_next_state(self.state, actual_action)
        reward = self.get_reward(next_state)
        done = self.is_terminal(next_state)
        
        self.state = next_state
        return next_state, reward, done
    
    def get_all_states(self):
        """Get all valid non-terminal states"""
        states = []
        for row in range(1, self.rows + 1):
            for col in range(1, self.cols + 1):
                state = (col, row)
                if self.is_valid_state(state) and not self.is_terminal(state):
                    states.append(state)
        return states
    
    def visualize_grid(self):
        """Visualize the gridworld"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Draw grid
        for i in range(self.rows + 1):
            ax.axhline(i, color='black', linewidth=2)
        for j in range(self.cols + 1):
            ax.axvline(j, color='black', linewidth=2)
        
        # Fill cells (convert 1-indexed to 0-indexed for plotting)
        for row in range(1, self.rows + 1):
            for col in range(1, self.cols + 1):
                state = (col, row)
                # For plotting: subtract 1 from col
                # For row: row 1 should be at bottom (y=0), row 3 at top (y=2)
                plot_col = col - 1
                plot_row = row - 1  # Now row 1 -> 0, row 2 -> 1, row 3 -> 2
                
                if state == self.wall_state:
                    ax.add_patch(plt.Rectangle((plot_col, plot_row), 1, 1, color='gray'))
                    ax.text(plot_col + 0.5, plot_row + 0.5, 'WALL', ha='center', va='center', 
                           fontsize=10, fontweight='bold')
                elif state == self.goal_state:
                    ax.add_patch(plt.Rectangle((plot_col, plot_row), 1, 1, color='lightgreen'))
                    ax.text(plot_col + 0.5, plot_row + 0.5, 'GOAL\n+1', ha='center', va='center', 
                           fontsize=14, fontweight='bold')
                elif state == self.trap_state:
                    ax.add_patch(plt.Rectangle((plot_col, plot_row), 1, 1, color='lightcoral'))
                    penalty_text = f'{self.trap_reward:.0f}' if self.trap_reward != -1 else '-1'
                    ax.text(plot_col + 0.5, plot_row + 0.5, f'TRAP\n{penalty_text}', ha='center', va='center', 
                           fontsize=14, fontweight='bold')
                elif state == self.start_state:
                    ax.text(plot_col + 0.5, plot_row + 0.5, 'START', ha='center', va='center', 
                           fontsize=12, fontweight='bold', color='blue')
                else:
                    ax.text(plot_col + 0.5, plot_row + 0.5, f'{state}', ha='center', va='center', 
                           fontsize=8, color='gray')
        
        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_aspect('equal')
        ax.set_xticks(range(self.cols + 1))
        ax.set_yticks(range(self.rows + 1))
        
        # Set labels to show 1-indexed values
        ax.set_xticklabels([1, 2, 3, 4, 5])
        ax.set_yticklabels([1, 2, 3, 4])
        
        ax.set_xlabel('Column (1-4)', fontsize=12)
        ax.set_ylabel('Row (1-3)', fontsize=12)
        title = f'GridWorld: Trap Penalty = {self.trap_reward}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()



# Test the environment
if __name__ == "__main__":
    print("="*60)
    print("GRIDWORLD ENVIRONMENT TEST")
    print("="*60)

    env = GridWorld()

    print(f"\nGrid size: {env.rows} x {env.cols}")
    print(f"Start state: {env.start_state}")
    print(f"Goal state: {env.goal_state} (reward: +{env.goal_reward})")
    print(f"Trap state: {env.trap_state} (reward: {env.trap_reward})")
    print(f"Wall state: {env.wall_state}")
    print(f"Step cost: {env.step_cost}")
    print(f"\nActions: 0=Up, 1=Right, 2=Down, 3=Left")

    # Visualize the grid
    env.visualize_grid()

  